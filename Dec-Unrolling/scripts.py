'''
This file is the scripts for all the codes
'''

'''
Task 1 Check all the connections in graph
'''
from dataloader import physical_graph, TrafficDataset
import numpy as np
import torch

def get_degrees(u_edges:torch.Tensor):
    '''
    u_edges: ndarray, in (n_edges, 2), already bidirectional
    '''
    n_edges = u_edges.size(0)
    degrees = np.zeros((n_edges,), dtype=int)
    for i in range(n_edges):
        degrees[u_edges[i,0]] += 1
        # degrees[u_edges[i,1]] += 1
    return degrees

'''
Task 2 Get the K-nearest neighbors for connection
'''
import torch
import heapq
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt

# 构建有向图

# 可视化有向图
def visualise_graph(G, dataset_name, fig_name):
    pos = nx.spring_layout(G)  # 使用弹簧布局来定位节点
    nx.draw(G, pos, with_labels=False, node_size=7, node_color='lightblue', arrowsize=2)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=2)

    plt.title(dataset_name)
    plt.savefig(fig_name, dpi=800)

def find_k_nearest_neighbors(edges:torch.Tensor, distances:torch.Tensor, k):
    edges = edges.detach().cpu().numpy()
    dist = distances.detach().cpu().numpy()
    graph = nx.DiGraph()
    for i in range(len(edges)): 
        graph.add_edge(edges[i, 0], edges[i, 1], weight=dist[i])
    
    visualise_graph(graph, 'PeMS04', 'PeMS04.png')
    nearest_neighbors = {}
    # print('nodes', graph.nodes)

    for node in graph.nodes: # 使用 Dijkstra 算法计算从当前节点出发的最短路径 
        distances = nx.single_source_dijkstra_path_length(graph, node) # 将结果按距离排序，并获取最近的 N 个节点 
        closest_nodes = heapq.nsmallest(k, distances.items(), key=lambda x: x[1]) # 存储结果 
        # nearest_neighbors[node] = closest_nodes 
        nearest_neighbors[node] = ([i for (i,_) in closest_nodes], [j for (_,j) in closest_nodes])
    return nearest_neighbors


from utils import *
from collections import Counter

batch_size = 12
num_workers = 4
dataset_dir = './datasets/PEMS0X_data/'
experiment_name = 'k_hop'
dataset_name = 'PEMS04'
T = 12
t_in = 6
stride = 3

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers)

u_edges, u_dist = train_set.u_edges, train_set.u_distance
n_nodes = train_set.n_nodes
print(u_edges.shape, u_dist.shape)

degrees = get_degrees(u_edges)
print(degrees.max(), degrees.min())
print(Counter(degrees))

kNN = find_k_nearest_neighbors(u_edges, u_dist, 8)
kNN_dict2 = {}
for key in kNN.keys():
    kNN_dict2[key] = ([i for (i,_) in kNN[key]], [j for (_,j) in kNN[key]])
print([len(kNN[i]) for i in range(n_nodes)])
print(kNN_dict2)


'''
Task 4: convolution/aggregation with K neighbors
'''
import torch.nn as nn

class GraphConvolutionLayer(nn.Module):
    '''
    convolution for feature extractor
    multiple GCN
    '''
    def __init__(self, n_nodes, n_in, n_out, kNN_dict, device, sigma=0.6):
        super().__init__()
        self.n_nodes = n_nodes
        self.kNN_dict = kNN_dict
        self.sigma = sigma
        self.device = device
        self.fc = nn.Linear(n_in, n_out)
    
    def conv(self, x):
        '''
        x (torch.Tensor): [B, T, n_nodes, n_in]
        '''
        B, T, n_nodes = x.size(0), x.size(1), x.size(2)
        conv_out = torch.zeros_like(x, device=self.device)
        for i in range(n_nodes):
            neighbors, dists = self.kNN_dict[i][0], torch.Tensor(self.kNN_dict[i][1]) # in (k)
            weights = torch.exp(-dists ** 2/ self.sigma ** 2)
            conv_out[:,:,i] = (weights * x[:,:,neighbors]).sum(2)
        return conv_out
    
    def forward(self, x):
        '''
        x (torch.Tensor): [B, T, n_nodes, n_in]
        '''
        conv_out = self.conv(x)
        return nn.ReLu()(self.fc(conv_out))

class GraphLearningModule(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

# import networkx as nx
# import heapq

# def find_nearest_neighbors(graph, N):
#     nearest_neighbors = {}

#     for node in graph.nodes:
#         # 使用 Dijkstra 算法计算从当前节点出发的最短路径
#         distances = nx.single_source_dijkstra_path_length(graph, node)
#         # 将结果按距离排序，并获取最近的 N 个节点
#         closest_nodes = heapq.nsmallest(N, distances.items(), key=lambda x: x[1])
#         # 存储结果
#         nearest_neighbors[node] = closest_nodes

#     return nearest_neighbors

# # 创建一个带有权重的有向图
# G = nx.DiGraph()
# edges = [
#     ('A', 'B', 1.5),
#     ('A', 'C', 2.0),
#     ('B', 'C', 1.0),
#     ('B', 'D', 2.2),
#     ('C', 'D', 1.8),
#     ('C', 'E', 2.1),
#     ('D', 'E', 1.3)
# ]

# # 添加边到图中
# for edge in edges:
#     G.add_edge(edge[0], edge[1], weight=edge[2])

# # 找出每个点最近的 N 个邻居
# N = 3
# nearest_neighbors = find_nearest_neighbors(G, N)

# for node, neighbors in nearest_neighbors.items():
#     print(f"{node}: {neighbors}")

