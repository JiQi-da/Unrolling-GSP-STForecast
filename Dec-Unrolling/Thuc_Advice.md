1. Sparse Graph operation, consider explore the graph a little.
    - The holder variable is so wasteful, some exploration in the data
    show that the node degrees is much smaller than the n_nodes. 
    So an NxN matrix for the holder variable is too much.
    - Consider create a edge_rank variable along with d_edges that contains rank of the edges connect to node i. 
    max_rank should be the largest node degrees (or percentile 90)
    Or consider K-nearest neighbor to bound the max_node_degree so that max_rank is bounded
    Given that max_rank, the holder matrix become (N x max_rank). This will substantially reduce memory footprint
    - Quite a lot of nodes have degree 1, this need serious consideration.
    - I can see that \PEMS0X_data\PEMS03 dataset have max degree 4, and about 300 nodes, 
    so 300x4 is much better than 300x300.

2. Feature Extractor module needs some rework. I strongly suggest something simple first.

    - Currently, I can see that \PEMS0X_data\PEMS03 dataset have max degree 4, 
    so my suggestion is to spent some time explore what kind of NN will work best,

    - I recommend to start first with a simple Feed forward Neural that concatenate 
    all K neighbor features and output the features of each nodes:
        features_out[i] = FNN(concat([features_in[j] for all nodes j connect to node i]))
        FNN is simple multiple layers of linear and relu activation, 
        you can bring in some tricks (layer norm, skip connect etc)

    - I also suggest to follow the GCN implementation in this paper https://arxiv.org/pdf/1609.02907v4, 
    look at equation 2, what they do is:
        + Given the features H_l, first is the normalized graph transform operation (D^{-1/2} A D^{-1/2})
        + Then follow by a linear transformation
        + Then an activation function ReLU 
     + Currently, the script use Sigmoid, why ?
     + Multiple layers of GCN is also important, the code only have 1 layer.  

    - Look into graphSage: https://arxiv.org/pdf/1706.02216 for other possibility for last resort.
     