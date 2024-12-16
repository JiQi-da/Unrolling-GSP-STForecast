import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math

import pandas as pd
import numpy as np
import torch.nn.functional as F
import networkx as nx
from lib.backup_modules import k_hop_neighbors, LR_guess
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller


# node embedding
class GNNExtrapolation(nn.Module):
    '''GNN extrapolation
    '''
    def __init__(self, n_nodes, t_in, T, u_edges, n_heads, device, hidden_size=3):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.d_edges = torch.cat([u_edges, torch.arange(0, n_nodes, dtype=torch.long).unsqueeze(1).repeat(1,2)], dim=0)
        assert T > t_in, 't_in > T'
        # model in markovian
        # self.MLP = nn.Sequential(nn.Linear(t_in * n_heads, hidden_size), nn.ReLU(), nn.Linear(hidden_size, T - t_in), nn.ReLU())
        self.shrink = nn.Sequential(nn.Linear(t_in * n_heads, T - t_in), nn.ReLU())

    def graph_convolution(self, x, d_ew):
        '''
        x in (n_nodes, n_channel)
        u_edges in (n_edges, 2)
        d_ew (normalized) in (n_edges, n_heads)
        return AH
        '''
        # d_ew.to(self.device)
        n_edges = self.d_edges.size(0)
        n_channels = x.size(-1)
        node_i, node_j = self.d_edges[:,0], self.d_edges[:,1]
        
        if x.ndim == 4:
            B, T = x.size(0), x.size(1)
            holder = torch.zeros((B, T, self.n_nodes, self.n_nodes, self.n_heads, n_channels), device=self.device)
            # print('x[:,:,node_i].size', x[:,:,node_i,None,:].size())
            # print(holder[:,:, node_i, node_j].device, x[:,:,node_i,None,:].device, d_ew[:,:,None].device)
            holder[:,:, node_i, node_j] = d_ew[:,:,None] * x[:,:,node_i,None,:]
            return holder.sum(2)
        else:
            assert x.ndim == 2, 'x.ndim must be 2 or 4'
            holder = torch.zeros((self.n_nodes, self.n_nodes, self.n_heads, n_channels), device=self.device)
            holder[node_i, node_j] = d_ew[:,:,None] * x[node_i] # in (n_edges, n_heads, n_channels)
            return holder.sum(0) # + x.unsqueeze(-2)
        
    def forward(self, x, d_ew):
        # signals in (Batch, T, n_nodes, n_channels)?
        B, t, n_nodes, n_channels = x.size()
        n_heads = d_ew.size(-1)
        y = self.graph_convolution(x, d_ew).permute(0,2,4,1,3).reshape(B, n_nodes, n_channels, -1) # in (B, n_nodes, n_channels, T * n_heads)
        y = self.shrink(y).permute(0,3,1,2) # self.
        return torch.cat([x, y], dim=1)


# original graph construction for GNN inputs
def GNN_graph_construction(n_nodes, n_heads, u_edges, u_dist, device, sigma=6): # fixed graph for all GNN module
    # multi_head lambdas
    u_dist = u_dist.to(device)
    lambda_ = torch.arange(1, n_heads + 1, 1, dtype=torch.float, device=device) / n_heads
    # cosntruct d_edges, d_dist
    d_edges = torch.cat([u_edges, torch.arange(0, n_nodes, dtype=torch.long).unsqueeze(1).repeat(1,2)], dim=0)
    d_dist = torch.cat([u_dist, torch.zeros((n_nodes,), device=device)], dim=0)
    # same as graph construction but with self loops
    n_edges = d_edges.size(0)
    d_ew = torch.exp(-(d_dist[:,None] ** 2) * lambda_/ (sigma ** 2)) #+ bar_u)
    node_i, node_j = d_edges[:,0], d_edges[:,1]
    holder = torch.zeros((n_nodes, n_nodes, n_heads), device=device)
    holder[node_i, node_j] = d_ew
    in_degree, out_degree = holder.sum(0), holder.sum(1)
    # regularization
    inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=device) / in_degree, torch.zeros((1,), device=device))
    inv_out_degree = torch.where(out_degree > 0, torch.ones((1,), device=device) / out_degree, torch.zeros((1,), device=device))
    # regularization
    return d_ew * torch.sqrt(inv_in_degree[node_j]) * torch.sqrt(inv_out_degree[node_i])

class FeatureExtractor(nn.Module):
    '''
        - extract feature using the signals of the adjacent nodes, which is a shallow GNN
        - using undirected graph: 2 GCNs on t and t + 1, local convolutions
    '''
    def __init__(self, n_in, n_out, n_nodes, n_heads, u_edges, u_dist, device, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.n_nodes = n_nodes
        self.n_in = n_in
        self.n_out = n_out
        # self.u_edges = u_edges # in (n_edges, 2)
        self.d_edges = torch.cat([u_edges, torch.arange(0, n_nodes, dtype=torch.long).unsqueeze(1).repeat(1,2)], dim=0)
        # add self connection
        # self.u_dist = u_dist # in (n_edges)
        self.d_dist = torch.cat([u_dist, torch.zeros((n_nodes,), device=device)], dim=0)
        self.device = device
        self.n_heads = n_heads
        # self.sigma = sigma
        # self.lambda_u = torch.arange(1, self.n_heads + 1, 1, dtype=torch.float, device=self.device) / self.n_heads # fixed multi-graph
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha # 0.1?
        assert alpha >= 0 and alpha < 1, 'alpha must be in [0,1)'
    
    def graph_convolution(self, x, d_ew):
        '''
        x in (B, T, n_nodes, n_channels) or (n_nodes, n_channel)
        u_edges in (n_edges, 2)
        d_ew (normalized) in (n_edges, n_heads)
        return AH
        '''
        n_edges = self.d_edges.size(0)
        n_channels = x.size(-1)
        node_i, node_j = self.d_edges[:,0], self.d_edges[:,1]
        
        if x.ndim == 4:
            B, T = x.size(0), x.size(1)
            holder = torch.zeros((B, T, self.n_nodes, self.n_nodes, self.n_heads, n_channels), device=self.device)
            holder[:,:, node_i, node_j] = d_ew[:,:,None] * x[:,:,node_i,None,:]
            return holder.sum(2)
        else:
            assert x.ndim == 2, 'x.ndim must be 2 or 4'
            holder = torch.zeros((self.n_nodes, self.n_nodes, self.n_heads, n_channels), device=self.device)
            holder[node_i, node_j] = d_ew[:,:,None] * x[node_i] # in (n_edges, n_heads, n_channels)
            return holder.sum(0) # + x.unsqueeze(-2)
        
    def dist_convolution(self, d_ew):
        '''
        weighted sum of distance, including itself
        u_ew in (n_edges, n_head)
        '''
        node_i, node_j = self.d_edges[:,0], self.d_edges[:,1]
        # distance holder
        holder = torch.zeros((self.n_nodes, self.n_nodes, self.n_heads), device=self.device)
        holder[node_i, node_j] = self.d_dist[:,None] * d_ew
        # convolution at in neighbors
        return holder.sum(0)

    def forward(self, x, T, d_ew): # TODO: multi-layer GCNs?
        '''
        x in (B, T, n_nodes, n_in)
        return features in (B, T, n_nodes, n_head, n_channels)
        '''
        x_conv = self.graph_convolution(x, d_ew) # in (B, T, n_nodes, n_heads, n_in - 1)
        dist_conv = self.dist_convolution(d_ew) # in (n_nodes, n_heads)
        # concatanate, simple merge
        if x.ndim == 4:
            B, t = x.size(0), x.size(1)
            x_feature = torch.cat([x_conv, dist_conv.unsqueeze(-1)[None, None, :,:,:].repeat(B, t, 1,1,1)], dim=-1) # in (B, T, )
        else:
            assert x.ndim == 2, 'x.ndim must be 2 or 4'
            x_feature = torch.cat([x_conv, dist_conv], dim=-1)
        # FC for weights
        x_out = self.fc(x_feature) # in (B, T, n_nodes, n_heads, n_channels)
        if x.ndim == 4:
            # linear extrapolation to T, for primal input
            if x.size(1) < T:
                x_out = LR_guess(x_out, T, self.device)
            # differential in time, using all neighbors
            x_out[:,1:] = self.alpha * x_out[:,:-1] + (1 - self.alpha) * x_out[:,1:]
        # sigmoid as a threshold
        x_out = self.sigmoid(x_out) # they surely need normalization
        return x_out

# return k-hop edges 

class GraphLearningModule(nn.Module):
    '''
    learning the directed and undirected weights from features
    '''
    def __init__(self, T, n_nodes, u_edges, n_heads, device, initialize:bool=False, u_dist=None, n_channels=None, sigma=6, Q1_init=1.2, Q2_init=0.8, M_init=1.5, k=2) -> None:
        '''
        Args:
            u_edges (torch.Tensor) in (n_edges, 2) # nodes regularized
            u_dist (torch.Tensor) in (n_edges)
        We construct d_edges by hand with n_nodes
        '''
        super().__init__()
        self.T = T
        self.n_nodes = n_nodes
        self.initialize = initialize
        self.device = device
        # construct d_edges, d_dist
        self.u_edges = u_edges
        self.d_edges = k_hop_neighbors(n_nodes, u_edges, k) # torch.cat([u_edges, torch.arange(0, n_nodes, dtype=torch.long).unsqueeze(1).repeat(1,2)], dim=0)
        # multi_heads
        self.n_heads = n_heads

        if self.initialize:
            assert u_dist is not None, 'udist should be tensors'
            # multihead initialization, shared across time
            self.lambda_u = Parameter(torch.arange(1, self.n_heads + 1, 1, dtype=torch.float, device=self.device) / self.n_heads, requires_grad=True)
            self.lambda_d = Parameter(torch.arange(1, self.n_heads + 1, 1, dtype=torch.float, device=self.device) / self.n_heads, requires_grad=True)
            self.u_dist = u_dist.to(device)
            self.d_dist = torch.cat([self.u_dist, torch.zeros((n_nodes,), device=device)], dim=0)
            self.sigma = sigma
        else:
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
        node_i, node_j = self.u_edges[:,0], self.u_edges[:,1]
        f_i, f_j = features[:, :, node_i], features[:,:,node_j] # in (B, T, n_edges, n_head, n_channels)
        df = f_j - f_i
        Mdf = F.relu(torch.einsum('hij, btehj -> btehi', self.multiM, df)) # in (B, T, n_edges, n_head, n_channels)
        weights = torch.exp(-(Mdf ** 2).sum(-1)) # in (B, T, n_edges, n_head)
        # print('undirected weights (unnormalized)', weights.max(), weights.min())
        holder = torch.zeros((B, T, self.n_nodes, self.n_nodes, self.n_heads), device=self.device)
        holder[:,:,node_i, node_j] = weights
        in_degree, out_degree = holder.sum(2), holder.sum(3)
        inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=self.device) / in_degree, torch.zeros((1,), device=self.device))
        # print('indegree', in_degree.size())
        inv_out_degree = torch.where(out_degree > 0, torch.ones((1,), device=self.device) / out_degree, torch.zeros((1,), device=self.device))
        return weights * torch.sqrt(inv_in_degree[:,:,node_j]) * torch.sqrt(inv_out_degree[:,:,node_i]) # in (B, T, n_edges, n_heads)
        # return (weights * inv_in_degree[:,:,node_j] + weights * inv_out_degree[:,:,node_i]) / 2

    def directed_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_features)
        Return:
            u_edges in (-1, T-1, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)
        node_i, node_j = self.d_edges[:,0], self.d_edges[:,1]
        features_i, features_j = features[:,:-1,node_i], features[:,1:, node_j] # in (B, (T-1), n_edges, n_head, n_in)

        # Q shared across layers
        Q_i = torch.einsum("hij, btehj -> btehi", self.multiQ1, features_i) # in (B, T*n_edges, n_heads, n_out)
        Q_j = torch.einsum("hij, btehj -> btehi", self.multiQ2, features_j)
        # Q_i = F.relu(torch.einsum("hij, btehj -> btehi", self.multiQ1, features_i)) # in (B, T*n_edges, n_heads, n_out)
        # Q_j = F.relu(torch.einsum("hij, btehj -> btehi", self.multiQ2, features_j))
        assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({self.multiQ2.max().item():.4f}, {self.multiQ2.min().item():.4f}; features in ({features_j.max().item()}, {features_j.min().item()}))'
        assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({features_i.max()}, {features_i.min()})'

        # TODO: Q not shared across layers? multiQ in (T, n_head, n_out, n_in) 
        # Q_j = torch.einsum("thij, btehj -> btehi", self.multiQ1, features_j) # in (B, T, n_edges, n_head, n_out)
        # Q_i = torch.einsum("thij, btehj -> btehi", self.multiQ2, features_i) 
        # # Q_j = Q @ f_j
        # bar_Q = (Q_j * Q_i).sum(-1).mean()
        # print(bar_Q)
        weights = torch.exp(-(Q_j * Q_i).sum(-1)) # un-normalized, in (B, (T-1), n_edges, n_head)
        # normalize on graph
        assert not torch.isnan(weights).any(), 'weights has NaN values'
        assert (not torch.isinf(weights).any()) and (not torch.isinf(-weights).any()), 'weight has inf values'

        holder = torch.zeros((B, T-1, self.n_nodes, self.n_nodes, self.n_heads), device=self.device)
        # compute the in_degrees
        holder[:,:, node_i, node_j] = weights
        in_degrees = holder.sum(2) # in (B, T-1, n_nodes(j), n_heads)
        inv_in_degrees = torch.where(in_degrees > 0, torch.ones((1), device=self.device) / in_degrees, torch.zeros((1), device=self.device))
        return weights * inv_in_degrees[:, :, node_j]

    def forward(self, features=None):
        '''
        return u_ew and d_ew
        '''
        if self.initialize:
            return self.undirected_graph_from_distance(), self.directed_graph_from_distance()
        else:
            # print('features', features)
            assert features is not None, 'feature cannot be none'
            return self.undirected_graph_from_features(features), self.directed_graph_from_features(features)
        
# u_edges = torch.Tensor([[0,1], [1,0], [1,2], [2,1]]).type(torch.long)
# glm = GraphLearningModule(1, 3, u_edges, torch.Tensor([1,1,2,2]), initialize=True, device='cpu', n_heads=1)
# print(glm.undirected_graph_from_distance())