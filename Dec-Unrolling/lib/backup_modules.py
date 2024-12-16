import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math

import pandas as pd
import numpy as np
import torch.nn.functional as F
import networkx as nx
import heapq
import matplotlib.pyplot as plt

class SimpleLinearExtrapolation(nn.Module):
    def __init__(self, n_nodes, t_in, T):
        super().__init__()
        self.t_in = t_in
        self.T = T
        assert T > t_in, 't_in > T'
        self.fc = nn.Linear(n_nodes, (T - t_in) * n_nodes)
        self.relu = nn.ReLU()
    def forward(self, x):
        # signals in (Batch, T, n_nodes, n_channels)?
        B, t, n_nodes, n_channels = x.size()
        y = self.fc(x[:,-1].transpose(-1,-2)).reshape(B, n_channels, -1, n_nodes) # in (B, n_channels, t, nodes)
        y = self.relu(y.permute(0,2,3,1))
        y = torch.cat([x, y], dim=1)
        return y
    
# primal prediction with linear extrapolation
    

def laplacian_embeddings(k, n_nodes, edges, u_dist, device, sigma=6, eps=1e-6):
    assert k > 0 and k < n_nodes, f'0 < k < {n_nodes}'
    # compute adjs
    adj = torch.zeros((n_nodes, n_nodes), device=device)
    for i in range(edges.size(0)):
        adj[edges[i,0], edges[i,1]] = math.exp(- u_dist[i] ** 2 / sigma ** 2)
    # eigenvalues
    diagonals = adj.sum(0)
    diagonal_x = torch.sqrt(diagonals[:,None] * diagonals[None,:])
    laplacian = torch.eye(n_nodes).to(device) - adj / diagonal_x
    L, Q = torch.linalg.eigh(laplacian)
    # histogram of laplacian 
    print('non_zero eigenvalues', (L>eps).sum())
    # TODO: smallest k eigenvectors?
    index = torch.topk(L, k, largest=False).indices
    Q_topk = Q[:, index]
    if Q_topk.is_complex():
        Q_topk = Q_topk.real
    return Q_topk

class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, k, n_nodes, edges, u_dist, sigma, device, tid_dim=10, diw_dim=2, use_t_emb=True):
        super().__init__()
        self.k = k
        self.n_nodes = n_nodes
        self.edges = edges
        self.u_dist = u_dist.to(device)
        self.sigma = sigma
        self.device = device
        # unchanged spatial embedding information
        self.spatial_emb = laplacian_embeddings(self.k, self.n_nodes, self.edges, self.u_dist, self.device, self.sigma) # in (n_nodes, k)
        self.use_t_emb = use_t_emb
        self.tid_dim = tid_dim
        self.diw_dim = diw_dim
        if use_t_emb:
            self.time_in_day_emb = nn.Embedding(12*24, tid_dim)
            self.day_in_week_emb = nn.Embedding(7, diw_dim)

    def forward(self, x, t_list=None):
        '''
        x in (B, T, n_nodes, 1)
        t in (B, T) t[batch, i] = t_i
        return (B, T, n_nodes, Dx + Ds + Dt)
        '''
        B, T = x.size(0), x.size(1)
        # add spatial embeddings
        output = torch.cat([x, self.spatial_emb[None, None,:,:].repeat(B, T, 1,1)], dim=-1)
        # temporal embeddings:
        if self.use_t_emb:
            assert t_list is not None, 't_list should not be None'
            time_of_day = t_list % (12 * 24)
            day_of_week = (t_list // (12 * 24)) % 7
            tid_emb = self.time_in_day_emb(time_of_day)
            diw_emb = self.day_in_week_emb(day_of_week)
            t_emb = torch.cat([tid_emb, diw_emb], dim=-1) # in (B, T, tid_dim + diw_dim)
            output = torch.cat([output, t_emb[:,:,None,:].repeat(1,1,self.n_nodes, 1)], dim=-1)
        return output

def LR_guess(y, T, device): # actually we won't use them
    '''
    A simple linear regression model for primal guess of the x
    regression function:
        y = W @ t + b, min_w ||y - W @ t||, data groups = batch
    Args:
        y (torch.tensor) in (B, t, n_nodes, n_heads, n_channels)
        T (int): time
        device (torch.device)
    '''
    # T = self.T
    B, t, n_nodes, n_channels = y.size()
    if t == 0:
        return torch.zeros((B, T, n_nodes, n_channels), device=device)
    elif t == 1:
        return y.repeat(1,T,1,1,1)
    else:
        y1 = y.transpose(0,1).reshape(t, -1) # in (t, F)
        x1 = torch.arange(0, t, 1).type(torch.float).to(device) # in (t,)
        bar_x =  (t-1) / 2
        bar_y = y1.mean(0)
        # print(x1.dtype, y1.dtype, bar_x)
        # print(y1.T @ x1)
        w = (t * y1.T @ x1 - x1.sum() * y1.sum(0)) / (t * x1.dot(x1) - (x1.sum()) ** 2)
        b = bar_y - bar_x * w
        # print('w', w)
        x_out = torch.arange(t, T, 1).type(torch.float).to(device)
        y_out = torch.cat([y1, x_out[:,None] * w + b], 0).view(T, B, n_nodes, n_channels).transpose(0,1)
        # [print(y_out.shape)
        return y_out   

def k_hop_neighbors(n_nodes, edges:torch.Tensor, k):
    # 创建有向图
    edges = edges.detach().cpu().numpy()
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 用于存储新的边
    new_edges = set()

    # 遍历每个节点
    for node in range(n_nodes):
        # 找到k-hop邻居
        k_hop = set(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())
        # 为每个k-hop邻居添加边
        for neighbor in k_hop:
            new_edges.add((node, neighbor))

    # 转换为numpy数组
    new_edges_array = np.array(list(new_edges))

    return torch.LongTensor(new_edges_array) # (n_edges, 2)

def visualise_graph(edges:torch.Tensor, distances:torch.Tensor, dataset_name, fig_name):
    edges = edges.detach().cpu().numpy()
    dist = distances.detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(len(edges)): 
        G.add_edge(edges[i, 0], edges[i, 1], weight=dist[i])

    pos = nx.spring_layout(G)  # 使用弹簧布局来定位节点
    nx.draw(G, pos, with_labels=False, node_size=7, node_color='lightblue', arrowsize=2)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=2)

    plt.title(dataset_name)
    plt.savefig(fig_name, dpi=800)


def find_k_nearest_neighbors(edges:torch.Tensor, distances:torch.Tensor, k):
    '''
    return: [dict] {node_i: [(node_j1, d1), ..., (node_jk, dk)]}
    '''
    edges = edges.detach().cpu().numpy()
    dist = distances.detach().cpu().numpy()
    graph = nx.DiGraph()
    for i in range(len(edges)): 
        graph.add_edge(edges[i, 0], edges[i, 1], weight=dist[i])
    nearest_neighbors = {}

    for node in graph.nodes: # 使用 Dijkstra 算法计算从当前节点出发的最短路径 
        distances = nx.single_source_dijkstra_path_length(graph, node) # 将结果按距离排序，并获取最近的 N 个节点 
        closest_nodes = heapq.nsmallest(k, distances.items(), key=lambda x: x[1]) # 存储结果 
        nearest_neighbors[node] = ([i for (i,_) in closest_nodes], [j for (_,j) in closest_nodes]) 
    return nearest_neighbors


def layer_norm_on_data(x:torch.Tensor, norm_shape):
    norm_dims = len(norm_shape)
    # print(norm_shape, x.shape[-norm_dims:])
    assert torch.Size(norm_shape) == x.shape[-norm_dims:], f'get {x[-norm_dims].size()} for {norm_shape}'
    dims = list(range(x.ndim - norm_dims, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
    std = torch.sqrt(mean_x2 - mean ** 2 + 1e-6)
    x_norm = (x - mean) / std
    return x_norm, mean, std

def layer_recovery_on_data(x, norm_shape, gain, bias):
    x_norm, _, _ = layer_norm_on_data(x, norm_shape)
    return x_norm * bias + gain