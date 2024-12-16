import torch
from tqdm import tqdm
import os
from dataloader import TrafficDataset
from torch.utils.data import DataLoader


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_admm_parameter(model, i):
    cgd_dict, mu_dict, rho_dict, mx_dict = {}, {}, {}, {}
    transformer_block = model.model_blocks[i]
    graph_learn = transformer_block['graph_learning_module']
    mx_dict['M'] = graph_learn.multiM.data
    mx_dict['Q1'] = graph_learn.multiQ1.data
    mx_dict['Q2'] = graph_learn.multiQ2.data

    admm_block = transformer_block['ADMM_block']
    cgd_dict['alpha_x'] = admm_block.alpha_x.data
    cgd_dict['beta_x'] = admm_block.beta_x.data
    cgd_dict['alpha_zu'] = admm_block.alpha_zu.data
    cgd_dict['beta_zu'] = admm_block.beta_zu.data
    cgd_dict['alpha_zd'] = admm_block.alpha_zd.data
    cgd_dict['beta_zd'] = admm_block.beta_zd.data

    mu_dict['mu_u'] = admm_block.mu_u.data
#     mu_dict['mu_d1'] = admm_block.mu_d1.data
    mu_dict['mu_d2'] = admm_block.mu_d2.data

    # rho_dict['rho'] = admm_block.rho.data
    rho_dict['rho_u'] = admm_block.rho_u.data
    rho_dict['rho_d'] = admm_block.rho_d.data
    return cgd_dict, mu_dict, rho_dict, mx_dict

def diff_dicts(dict1:dict, dict2):
    keys = dict1.keys()
    d_dict = {}
    for k in keys:
        d_dict[k] = torch.norm(dict1[k] - dict2[k])
    return d_dict

def disp_dict(d:dict):
    minmaxdict = {}
    for k in d.keys():
        minmaxdict[k] = (d[k].max(), d[k].min())
    return minmaxdict

def regularized_terms(model, x, y, t=None): # TODO: save as the same operators
        '''
        Notice that for here, x stands for the true sequence, the full observation, the ideal x. Just for testing and validation.
    Inputs:
        u_model: unrolling models (T)
        x (torch.Tensor): in (B, T, n_nodes, n_channels)
    Return:
        regularized term (x L^u x, ||Ld x||_2, ||L_d x||_1)
    '''
        assert not model.training, 'only on validation and test'
        B = x.size(0)
        # x = x.unsqueeze(-2).repeat(1,1,1,4,1)

        x_l2_list = []
        x_l1_list = []
        x_Lu_norm_list = []
        Ldx_l2_list = []
        Ldx_l1_list = []
        with torch.no_grad():
            multi_x = x.unsqueeze(-2).repeat(1,1,1,model.n_heads,1)
            print(y.shape)
            output = model.linear_extrapolation(y , model.GNN_graph)

            for i in range(model.num_blocks):
                t_block = model.model_blocks[i]

                feature_extractor = t_block['feature_extractor']
                graph_learn = t_block['graph_learning_module']
                admm_block = t_block['ADMM_block']
                # mus
                # mu_u = admm_block.mu_u.max()
                # mu_d1 = admm_block.mu_d1.max()
                # mu_d2 = admm_block.mu_d2.max()
                # passing forward
                # features, regenerate from original signal
                features = feature_extractor(output, model.T, model.GNN_graph) # in (batch, T, n_nodes, n_heads, n_out)
            
                u_ew, d_ew = graph_learn(features)
                # print('u_ew, d_ew', u_ew.size(), d_ew.size())

                admm_block.u_ew = u_ew
                admm_block.d_ew = d_ew

                # cauculate norms, x in (B, T, n_nodes, 1)

                # pass on the module, caucluate norms
                p = model.skip_connection_weights[i]
                # if i == 0:
                #     multi_x = x.unsqueeze(-2).repeat(1,1,1,4,1)
                # else:
                #     multi_x = output.unsqueeze(-2).repeat(1,1,1,4,1)
                x_l2_list.append(torch.norm(multi_x.view(B, -1), dim=1))
                x_l1_list.append(torch.norm(multi_x.view(B, -1), p=1, dim=1))
                Lu_norm = torch.sqrt((multi_x * admm_block.apply_op_Lu(multi_x)).sum((1,2,3,4))) # L2 norm
                x_Lu_norm_list.append(Lu_norm)
                Ldx = admm_block.apply_op_Ldr(multi_x).view(B, -1)
                Ld_l2 = torch.norm(Ldx, dim=1)
                Ld_l1 = torch.norm(Ldx, dim=1, p=1)
                Ldx_l2_list.append(Ld_l2)
                Ldx_l1_list.append(Ld_l1)

                output_old = output
                output_new = admm_block(output, model.t_in)
                output = p * output_new + (1-p) * output_old
                
            # calculate mean of norms
            # print(Ldx_l1_list)
            # print(x_norm_list[0].shape, len(x_norm_list))
            x_l2 = torch.stack(x_l2_list, dim=1).mean(0)
            x_l1 = torch.stack(x_l1_list, dim=1).mean(0)
            Lu_norm = torch.stack(x_Lu_norm_list, dim=1).mean(0)
            Ldx_l1 = torch.stack(Ldx_l1_list, dim=1).mean(0)
            Ldx_l2 = torch.stack(Ldx_l2_list, dim=1).mean(0)
            # print(x_norm.size())
        # organize the feature dicts
        
        return x_l2, x_l1, Lu_norm, Ldx_l1, Ldx_l2

def eval_smoothness(model, val_loader):
    model.eval()
    # n_heads = model.n_heads
    with torch.no_grad():
        x_l2_sum, x_l1_sum, Lu_norm_sum, Ld_l1_sum, Ld_l2_sum = torch.zeros((model.num_blocks,), device=model.device), torch.zeros((model.num_blocks,), device=model.device), torch.zeros((model.num_blocks,), device=model.device), torch.zeros((model.num_blocks,), device=model.device), torch.zeros((model.num_blocks,), device=model.device)
        for y, x in tqdm(val_loader):
            y, x = y.to(model.device), x.to(model.device)
            x_l2, x_l1, Lu_norm, Ldx_l1, Ldx_l2 = regularized_terms(model, x, y)
            # print(x_norm, Lu_norm, Ldx_l1, Ldx_l2)
            print(x_l2.shape)
            x_l2_sum += x_l2
            x_l1_sum += x_l1
            Lu_norm_sum += Lu_norm
            Ld_l1_sum += Ldx_l1
            Ld_l2_sum += Ldx_l2
    norm_dict = {}  
    norm_dict['||x||_2'] = x_l2_sum / len(val_loader)
    norm_dict['||x||_1'] = x_l1_sum / len(val_loader)
    norm_dict['sqrt(x^T L^u x)'] = Lu_norm_sum / len(val_loader)
    norm_dict['||L^d x||_1'] = Ld_l1_sum / len(val_loader)
    norm_dict['||L^d x||_2'] = Ld_l2_sum / len(val_loader)
    norm_dict['Ld_l1/x_l1'] = norm_dict['||L^d x||_1'] / norm_dict['||x||_1']
    norm_dict['Ld_l2/x_l2'] = norm_dict['||L^d x||_2'] / norm_dict['||x||_2']
    norm_dict['Lu_l2/x_l2'] = norm_dict['sqrt(x^T L^u x)'] / norm_dict['||x||_2']
    return norm_dict
     
model_pretrained_path = 'UnrollingForecasting/MainExperiments/models/v2{experiment_name}/PEMS03/MSE_5b24_4h_6f/val_5.pth'

if model_pretrained_path is not None:
    model = torch.load(model_pretrained_path, map_location={'cuda:0':'cuda:1'})
    print(model.linear_extrapolation)
    model.device = 'cuda:1'
    for module in model.children():
        if hasattr(module, 'device'):
            module.device = 'cuda:1'

    for moduledict in model.model_blocks:
        for key, module in moduledict.items():
            if hasattr(module, 'device'):
                module.device = 'cuda:1'
    model.to('cuda:1')
# model_10 = torch.load('/mnt/qij/UnrollingForecasting/UnrollingForecasting/MainExperiments/models/v2/PEMS03/4b_4h_6f/val_10.pth')

batch_size = 16
learning_rate = 1e-3
num_epochs = 30
num_workers = 4
dataset_name = 'PEMS03'
# load datasets
data_folder = os.path.join('MainExperiments/datasets/PEMS0X_data/', dataset_name)
graph_csv = dataset_name + '.csv'
data_file = dataset_name + '.npz'
if dataset_name == 'PEMS03':
    id_file = dataset_name + '.txt'
else:
    id_file = None



val_set = TrafficDataset(data_folder, graph_csv, data_file, 12, 6, 3, 'val', id_file=id_file)
val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)

cgd_dict, mu_dict, rho_dict, mx_dict = get_admm_parameter(model, 1)
print('Q1', mx_dict['Q1'].min(), mx_dict['Q1'].max(), torch.isnan(mx_dict['Q1']).any())
print('Q2', mx_dict['Q2'].min(), mx_dict['Q2'].max(), torch.isnan(mx_dict['Q2']).any())
print('M', mx_dict['M'].min(), mx_dict['M'].max(), torch.isnan(mx_dict['M']).any())
print('mu', disp_dict(mu_dict))
# print(model_5)
print('val', eval_smoothness(model, val_loader))
# print('val_10', eval_smoothness(model_10, val_loader))




