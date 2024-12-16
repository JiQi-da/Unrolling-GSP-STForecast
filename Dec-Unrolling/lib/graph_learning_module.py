import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math

import pandas as pd
import numpy as np
import torch.nn.functional as F
import networkx as nx
from lib.backup_modules import k_hop_neighbors, LR_guess, find_k_nearest_neighbors
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller


# node embedding
class GNNExtrapolation(nn.Module):
    '''GNN extrapolation
    '''
    def __init__(self, n_nodes, t_in, T, kNN, n_heads, device, sigma):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.kNN = kNN
        assert T > t_in, 't_in > T'
        # model in markovian
        # self.MLP = nn.Sequential(nn.Linear(t_in * n_heads, hidden_size), nn.ReLU(), nn.Linear(hidden_size, T - t_in), nn.ReLU())
        self.shrink = nn.Sequential(nn.Linear(t_in * n_heads, T - t_in), nn.ReLU())
        self.sigma = sigma
        
    def forward(self, x):
        # signals in (Batch, T, n_nodes, n_channels)?
        B, t_in, n_nodes, n_channels = x.size()
        # aggregation
        agg, _ = graph_aggregation(x, self.kNN, self.n_heads, self.device, self.sigma) # in (B, t_in, N, n_heads, n_channels)
        agg = agg.permute(0,2,4,1,3).reshape(B, n_nodes, n_channels, -1) # in (B, N, n_channels, t_in * n_heads)
        y = self.shrink(agg).permute(0,3,1,2)
        return torch.cat([x, y], dim=1)
##################################################################################


def graph_aggregation(x:torch.Tensor, nearest_neighbors:dict, n_heads, device, sigma=6):
    B, T, n_nodes, n_in = x.size(0), x.size(1), x.size(2), x.size(-1)
    lambda_ = torch.arange(1, n_heads + 1, 1, dtype=torch.float, device=device) / n_heads
    agg = torch.zeros((B, T, n_nodes, n_heads, n_in), device=device)
    dist_agg = torch.zeros((n_nodes, n_heads), device=device)
    for i in range(n_nodes):
        neighbors = nearest_neighbors[i][0]
        dists = torch.tensor(nearest_neighbors[i][1], device=device)
        weights = torch.exp(- (dists[:,None] ** 2) * lambda_ / (sigma ** 2)) # in (k, n_heads)
        if x.ndim < 5:
            agg[:,:,i] = (weights[:,:,None] * x[:,:,neighbors,None,:]).sum(2) # in (B, T, k, n_heads, n_in)
        else:
            # print(x[:,:,neighbors,:,:].shape, weights.shape, neighbors)
            agg[:,:,i] = (weights[:,:,None] * x[:,:,neighbors,:,:]).sum(2)
        dist_agg[i] = (weights * dists[:,None]).sum(0)
    return agg, dist_agg # in (B, T, N, n_heads, n_in), (N, n_heads)

class GraphConvolutionLayer(nn.Module):
    def __init__(self, n_in, n_out, n_nodes, n_heads, kNN, device, sigma=6, alpha=0.2, use_dist_conv=False):
        super().__init__()
        self.use_dist_conv = use_dist_conv
        if self.use_dist_conv:
            self.fc = nn.Linear(n_in + 1, n_out)
        else:
            self.fc = nn.Linear(n_in, n_out)
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.device = device
        self.kNN = kNN # dict: {i:nodeList, distList}
        self.sigma = sigma
        self.relu = nn.ReLU()
        self.alpha = alpha
        # self.use_dist_conv = use_dist_conv

    def forward(self, x):
        # aggregate
        B, T = x.size(0), x.size(1)
        agg, dist_agg = graph_aggregation(x, self.kNN, self.n_heads, self.device, self.sigma) # in (B, T, N, n_heads, n_in), (N, n_heads)
        # add next layer, agg x_i(t+1)
        if self.use_dist_conv:
            dist_agg = dist_agg[None, None, :,:].repeat(B, T, 1, 1).unsqueeze(-1)
            agg = torch.cat((agg, dist_agg), -1)
        # else: use Laplacian embedding (spatial embeddings)

        agg[:,1:] = (1 - self.alpha) * agg[:,1:] + self.alpha * agg[:,:-1]

        out = self.fc(agg)
        return self.relu(out)

class FeatureExtractor(nn.Module):
    def __init__(self, n_in, n_out, n_nodes, n_heads, kNN, device, n_layers=3, sigma=6, alpha=0.2, use_dist_conv=False):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kNN = kNN
        self.sigma = sigma
        self.alpha = alpha
        self.use_dist_conv = use_dist_conv
        self.n_layers = n_layers

        self.input_layer = GraphConvolutionLayer(n_in, n_out, n_nodes, n_heads, self.kNN, device, sigma, alpha, True)
        # GNN layers
        if self.n_layers > 1:
            self.GNN = nn.Sequential(*[GraphConvolutionLayer(n_out, n_out, n_nodes, n_heads, self.kNN, device, sigma, alpha, False)
                for i in range(n_layers - 1)
            ])

    def forward(self, x):
        out = self.input_layer(x)
        # print('GCN 1:', out.size())
        if self.n_layers == 1:
            return out
        else:
            return self.GNN(out)
# return k-hop edges 

class GraphLearningModule(nn.Module):
    '''
    learning the directed and undirected weights from features
    '''
    def __init__(self, T, n_nodes, kNN, n_heads, device, n_channels=None, sigma=6, Q1_init=1.2, Q2_init=0.8, M_init=1.5) -> None:
        '''
        Args:
            u_edges (torch.Tensor) in (n_edges, 2) # nodes regularized
            u_dist (torch.Tensor) in (n_edges)
        We construct d_edges by hand with n_nodes
        '''
        super().__init__()
        self.T = T
        self.n_nodes = n_nodes
        self.device = device
        # construct d_edges, d_dist
        self.kNN = kNN
        # multi_heads
        self.n_heads = n_heads

        # self.n_features = n_features # feature channels
        self.n_channels = n_channels
        self.n_out = self.n_out = (self.n_channels + 1) // 2
        # define multiM, multiQs
        self.Q1_init = Q1_init
        self.Q2_init = Q2_init
        self.M_init = M_init
        q_form = torch.zeros((self.n_heads, self.n_out, self.n_channels), device=self.device)
        q_form[:,:, :self.n_out] = torch.diag_embed(torch.ones((self.n_heads, self.n_out), device=self.device))
        # all variables shared across time
        self.multiQ1 = Parameter(q_form * self.Q1_init, requires_grad=True)
        self.multiQ2 = Parameter(q_form * self.Q2_init, requires_grad=True)
        self.multiM = Parameter(torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device)) * self.M_init, requires_grad=True) # in (n_heads, n_channels, n_channels)

    def undirected_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
        Returns:
            u_edges in (-1, T, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)
        weights = {}
        degrees = torch.zeros((B, T, self.n_nodes, self.n_heads), device=self.device)
        node_js = {}
        for node_i in range(self.n_nodes):
            node_js[node_i] = self.kNN[node_i][0] # (k)
            df = features[:,:, node_i].unsqueeze(2) - features[:,:, node_js[node_i]] # in (B, T, k, n_heads, n_channels)
            Mdf = torch.einsum('hij, btehj -> btehi', self.multiM, df) # in (B, T, k, n_head, n_channels)
            weights[node_i] = torch.exp(- (Mdf ** 2).sum(-1)) # in (B, T, k, n_heads)
            degrees[:,:,node_i] = weights[node_i].sum(2) # > 0 (B, T, n_head)
        
        for node_i in range(self.n_nodes):
            # symmetric normalization
            degree_i, degree_j = degrees[:,:,node_i], degrees[:, :, node_js[node_i]] # (B, T, n_head), (B, T, k, n_head)
            weights[node_i] = weights[node_i] / (torch.sqrt(degree_i[:,:,None,:]) * torch.sqrt(degree_j))

        return weights # dict, {i: (B, T, k, n_heads)}

    def directed_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_features)
        Return:
            u_edges in (-1, T-1, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)
        weights = {}
        # only compute in-degrees
        for node_j in range(self.n_nodes):
            node_is = self.kNN[node_j][0]
            features_i = features[:,:-1, node_is] # in (B, T-1, n_heads, n_in)
            features_j = features[:,1:, node_j] # in (B, T-1, k, n_heads, n_in)
            Q_i = torch.einsum('hij, btehj -> btehi', self.multiQ1, features_i)
            Q_j = torch.einsum('hij, bthj -> bthi', self.multiQ2, features_j)
            # TODO: assertation
            assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({self.multiQ2.max().item():.4f}, {self.multiQ2.min().item():.4f}; features in ({features_j.max().item()}, {features_j.min().item()}))'
            assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({features_i.max()}, {features_i.min()})'
            weight_j = torch.exp(- (Q_i * Q_j[:,:,None,:,:]).sum(-1)) # in (B, T-1, k, n_heads)
            # print(weight_j.max(), weight_j.min())
            degree = weight_j.sum(2) # in (B, T-1, )
            # print(degree.max(), degree.min())
            weights[node_j] = weight_j / degree.unsqueeze(2)
        
        return weights

    def forward(self, features=None):
        '''
        return u_ew and d_ew
        '''
        # print('features', features)
        assert features is not None, 'feature cannot be none'
        return self.undirected_graph_from_features(features), self.directed_graph_from_features(features)
        
# u_edges = torch.Tensor([[0,1], [1,0], [1,2], [2,1]]).type(torch.long)
# glm = GraphLearningModule(1, 3, u_edges, torch.Tensor([1,1,2,2]), initialize=True, device='cpu', n_heads=1)
# print(glm.undirected_graph_from_distance())