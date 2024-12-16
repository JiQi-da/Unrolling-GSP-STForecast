import torch
import torch.nn as nn
from lib.graph_learning_module import GNNExtrapolation, FeatureExtractor, GraphLearningModule
from lib.admm_block import ADMMBlock
from lib.backup_modules import layer_norm_on_data, layer_recovery_on_data, find_k_nearest_neighbors
from torch.nn.parameter import Parameter

class UnrollingModel(nn.Module):
    def __init__(self, num_blocks, device, 
                 T, t_in,
                 n_heads,
                 signal_channels,
                 feature_channels,
                 k_hop, 
                 GNN_alpha=0.2,
                 graph_sigma=6,
                 graph_info = {
                     'n_nodes': None,
                     'u_edges': None,
                     'u_dist': None,
                 },
                 ADMM_info = {
                 'ADMM_iters':30,
                 'CG_iters': 3,
                 'PGD_iters': 3,
                 'mu_u_init':3,
                 'mu_d1_init':3,
                 'mu_d2_init':3,
                 },
                 use_norm=False,
                 use_dist_conv=True
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.device = device
        self.T = T
        self.t_in = t_in
        self.n_heads = n_heads
        self.use_norm = use_norm

        # define a graph connection pattern
        self.kNN = find_k_nearest_neighbors(graph_info['u_edges'], graph_info['u_dist'], k_hop)
        self.linear_extrapolation = GNNExtrapolation(graph_info['n_nodes'], t_in, T, self.kNN, n_heads, device, graph_sigma)

        self.model_blocks = nn.ModuleList([])

        self.skip_connection_weights = Parameter(torch.ones((num_blocks,), device=self.device) * 0.95, requires_grad=True)

        for i in range(self.num_blocks):
            self.model_blocks.append(nn.ModuleDict(
                {
                    'feature_extractor': FeatureExtractor(
                        n_in=signal_channels,
                        n_out=feature_channels,
                        n_nodes=graph_info['n_nodes'],
                        n_heads=n_heads,
                        kNN=self.kNN,
                        device=device,
                        sigma=graph_sigma,
                        alpha=GNN_alpha,
                        use_dist_conv=use_dist_conv
                    ),
                    'ADMM_block': ADMMBlock(
                        T=T,
                        n_nodes=graph_info['n_nodes'],
                        n_heads=n_heads,
                        n_channels=signal_channels,
                        kNN=self.kNN,
                        device=device,
                        ADMM_info=ADMM_info
                    ),
                    'graph_learning_module': GraphLearningModule(
                        T=T,
                        n_nodes=graph_info['n_nodes'],
                        kNN=self.kNN,
                        n_heads=n_heads,
                        device=device,
                        n_channels=feature_channels,
                    )
                }
            ))
        self.y_norm_shape = [self.t_in, graph_info['n_nodes'], signal_channels]
        self.norm_shape = [self.T, graph_info['n_nodes'], signal_channels]
    
    def regularized_terms(self, x, t=None): # TODO: save as the same operators
        '''
        Notice that for here, x stands for the true sequence, the full observation, the ideal x. Just for testing and validation.
    Inputs:
        u_model: unrolling models (T)
        x (torch.Tensor): in (B, T, n_nodes, n_channels)
    Return:
        regularized term (x L^u x, ||Ld x||_2, ||L_d x||_1)
    '''
        assert not self.training, 'only on validation and test'
        B = x.size(0)
        if self.use_norm:
            x, mean, std = layer_norm_on_data(x, self.norm_shape)

        x_norm_list = []
        x_Lu_norm_list = []
        Ldx_l2_list = []
        Ldx_l1_list = []
        with torch.no_grad():
            for i in range(self.num_blocks):
                t_block = self.model_blocks[i]

                feature_extractor = t_block['feature_extractor']
                graph_learn = t_block['graph_learning_module']
                admm_block = t_block['ADMM_block']
                # mus
                # mu_u = admm_block.mu_u.max()
                # mu_d1 = admm_block.mu_d1.max()
                # mu_d2 = admm_block.mu_d2.max()
                # passing forward
                # features, regenerate from original signal
                features = feature_extractor(x, self.T, self.GNN_graph) # in (batch, T, n_nodes, n_heads, n_out)
            
                u_ew, d_ew = graph_learn(features)

                admm_block.u_ew = u_ew
                admm_block.d_ew = d_ew

                # cauculate norms, x in (B, T, n_nodes, 1)

                # pass on the module, caucluate norms
                p = self.skip_connection_weights[i]

                x_norm_list.append(torch.norm(x, dim=0))
                Lu_norm = torch.sqrt((x * admm_block.apply_op_Lu(x)).sum([1,2,3])) # L2 norm
                x_Lu_norm_list.append(Lu_norm)
                Ld_l2 = torch.norm(admm_block.apply_op_Ldr(x))
                Ld_l1 = torch.norm(admm_block.apply_op_Ldr(x), p=1)
                Ldx_l2_list.append(Ld_l2)
                Ldx_l2_list.append(Ld_l1)
                # pass on
                x_old = x
                x_new = admm_block(x, self.t_in)
                x = p * x_new + (1-p) * x_old
                
            # # calculate mean of norms0
            # x_norm = torch.cat(x_norm_list, dim=0).mean()
            # Lu_norm = torch.cat(x_Lu_norm_list, dim=0).mean()
            # Ldx_l1 = torch.cat(Ldx_l1_list, dim=0).mean()
            # Ldx_l2 = torch.cat(Ldx_l2_list, dim=0).mean()
        # organize the feature dicts
        
        return torch.Tensor(x_norm_list), torch.Tensor(x_Lu_norm_list), torch.Tensor(Ldx_l1_list), torch.Tensor(Ldx_l2_list) # in (n_blocks, B)
    
    def clamp_param(self, alpha_max, beta_max):
        for i in range(self.num_blocks):
            transformer_block = self.model_blocks[i]
            graph_learn:GraphLearningModule = transformer_block['graph_learning_module']
            # graph_learn.multiM.data = torch.clamp(graph_learn.multiM.data, 0.0)
            # graph_learn.multiQ1.data = torch.clamp(graph_learn.multiQ1.data, 0.0)
            # graph_learn.multiQ2.data = torch.clamp(graph_learn.multiQ2.data, 0.0)
            admm_block:ADMMBlock = transformer_block['ADMM_block']
            admm_block.alpha_x.data = torch.clamp(admm_block.alpha_x.data, 0.0, alpha_max)
            admm_block.beta_x.data = torch.clamp(admm_block.beta_x.data, 0.0, beta_max)
            admm_block.alpha_zu.data = torch.clamp(admm_block.alpha_zu.data, 0.0, alpha_max)
            admm_block.beta_zu.data = torch.clamp(admm_block.beta_zu.data, 0.0, beta_max)
            admm_block.alpha_zd.data = torch.clamp(admm_block.alpha_zd.data, 0.0, alpha_max)
            admm_block.beta_zd.data = torch.clamp(admm_block.beta_zd.data, 0.0, beta_max)

            #W admm_block.epsilon.data = torch.clamp(admm_block.epsilon.data, 0.0, 0.2)


    def forward(self, y):
        '''
        y in (batch, t, n_nodes, signal_channels)
        '''
        # linear extrapolation
        t = y.size(1)
        if self.use_norm:
            y, mean, std = layer_norm_on_data(y, self.y_norm_shape)
            
        output = self.linear_extrapolation(y)
        # print('output', output.size())

        for i in range(self.num_blocks):
            output_old = output
            transformer_block = self.model_blocks[i]
            feature_extractor = transformer_block['feature_extractor']
            graph_learn = transformer_block['graph_learning_module']
            admm_block = transformer_block['ADMM_block']
            # learn features
            features = feature_extractor(output) # in (batch, T, n_nodes, n_heads, n_out)
            # print('features', features.size())
            u_ew, d_ew = graph_learn(features)
            # print('max in weights', get_max_in_dict(u_ew), get_max_in_dict(d_ew))
            # print('u_ew', u_ew[0].shape)
            # print('d_ew', d_ew[0].shape)
            admm_block.u_ew = u_ew
            admm_block.d_ew = d_ew
            output_new = admm_block(output, t) # in (batch, T, n_nodes, signal_channels)
            # skip connections
            p = self.skip_connection_weights[i]
            output = p * output_new + (1-p) * output_old
        if self.use_norm:
            output = layer_recovery_on_data(output, self.norm_shape, mean, std)
        return nn.ReLU()(output)            # 

def get_max_in_dict(ew:dict):
    maxlist = []
    for v in ew.values():
        maxlist.append(v.max())
    return max(maxlist)
