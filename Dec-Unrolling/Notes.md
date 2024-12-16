# Discussion on Dec 5

## Advice on graph
1. Look into the connections of the real edges
    - Thuc: most nodes are under degree 4, while a large number of nodes are of degree 2
    - My suggestion: use a *Dijkstra Algorithm* to find the shortest K paths for each nodes and define connections

2. Feature extractor: max degree 4, just concatenate all the features as an embedding $[{x_j}]$, start with basic models

3. GCN extraction: (1) use multi-layer or multi-connection, choose multilayer; (2) use ReLU as activation

4. Fully directed graph ($A\rightarrow B$ and $B\rightarrow A$ is different)


## Advice on training

1. Fix the graph parameters and train the ADMM layers to see whether it will converge

2. Fix the ADMM layers (as the optimal solutions) and train the Graph Learning Modules

# Work plan and results from Dec 14-19
Graphs:
- [x] use a Dijkstra Algorithm to find the shortest $k$ paths for each node
- [x] extend the $k$-connection to GCN feature extractor (and distance convolution, laplacian embeddings)
- [x] Graph Learning Module with the $k$-connection
- [ ] edit the multiplications in ADMM block
- [ ] test and debug

> Why no concatenate? some leave-out points, 3/4 loops

Training:
- [] Reuse the fixed ADMM with proper parameter


# Discussion on Dec 19