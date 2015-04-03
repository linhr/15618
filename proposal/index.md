---
layout: index
---

# Project Proposal

## SUMMARY

We are going to implement a GPU-based parallel eigendecomposition algorithm (the Lanczos algorithm) for spectral analysis of large graphs.

## BACKGROUND

Eigendecomposition is a basic routine in many linear algebra libraries. It is also useful for graph analysis. Given an undirected graph \\(G = (V, E)\\), the spectral graph theory tries to gain insights into the graph structure by considering the eigenvalues and eigenvectors of the adjacency matrix \\(A\\) or the Laplacian matrix \\(L = D - A\\) where \\(D\\) is a diagonal matrix with element \\(d\_{ii} = \sum\_{j = 1}^n a\_{ij}\\). For unweighted graphs, if \\((i, j) \in E\\), then \\(a\_{ij} = 1\\). Otherwise \\(a\_{ij} = 0\\). For weighted graphs, the element \\(a\_{ij}\\) encodes the weights for the edge \\((i, j)\\). Some typical problems in spectral analysis involve solving the following equations for eigenvalue \\(\lambda\\) and eigenvector \\(\mathbf{f}\\).

*   \\(A\mathbf{f} = \lambda \mathbf{f}\\)
*   \\(L\mathbf{f} = \lambda \mathbf{f}\\)
*   \\(L\mathbf{f} = \lambda D\mathbf{f}\\), or \\(D^{-1}L\mathbf{f} = \lambda\mathbf{f}\\)

Therefore, the key is to solve the eigendecomposition problem \\[M\mathbf{f} = \lambda \mathbf{f}\\] for some matrices \\(M\\). As this project focus on the area of graph analysis, we restrict ourselves to the case when \\(M\\) is symmetric.

The spectral graph theory has many applications. The first few components of the eigenvector can be viewed as a low dimensional representation of vertices in the graph. This dimension reduction procedure followed by a k-means clustering forms the basis for spectral clustering [2]. Also spectral analysis can be used for image segmentation, if we convert an image to a similarity graph where each vertex corresponds to a pixel in the image and edge weights measure the similarity of adjacent pixels [3].

The Lanczos algorithm [1] is one of the classic algorithms for eigendecomposition. It is an iterative update to the eigenvector and eigenvalues. Each step involves matrix-vector multiplications, which can be parallelized on GPU. Also, it would be interesting if we can visualize the low dimensional representation produced from eigendecomposition, and rendering such a visualization can naturally be done in parallel on GPU.

## CHALLENGES

The graphs we are interested in are generally sparse, so a basic challenge for our project is to optimize the algorithm for sparse matrices. We plan to design a suitable way to store the graph structure and implement efficient sparse matrix-vector multiplication, the key operation in the Lanczos algorithm. Different parts of the graph may have different sparsity patterns, so we may need to resolve issues such as work imbalance.

## RESOURCES

* We refer to some existing work on parallel Lanczos algorithm [5, 6] as our starting point. As these papers do not provide much details on the sparse matrix operations, we need to come up with our own solution to fit into the general framework.
* We plan to use GPUs on GHC machines for our experiments.
* We may use the [cuBLAS](https://developer.nvidia.com/cuBLAS) library for some basic linear algebra operations.

## GOALS & DELIVERABLES

We plan to achieve an efficient GPU-based Lanczos algorithm implementation for eigendecomposition that can deal with large sparse matrices. We hope that our solution can be at least 5x faster on large graphs (matrices) than CPU-based solutions (such as the one provided in MATLAB) on GHC machines, because from most papers we have read we see a speedup of at least 5x compared with the MATLAB implementation.

We plan to conduct thorough experiments on the performance for different kinds of graphs (generated graphs, social networks, similarity graph created from some datasets), and show the speedup graphs comparing our solution with other CPU-based solutions and GPU-based solutions. We hope that our performance will be comparable to the one in the CUDA Toolkit [cuSOLVER](https://developer.nvidia.com/cusolver), which means that we have done a really good job.

We also plan to use CUDA to visualize the eigenvectors we have computed. If we have time we may try to implement parallel spectral clustering or image segmentation based on our eigensolver.

## SCHEDULE

### April 3rd - April 9th

Implement the Lanczos algorithm for dense graphs on GPU with CUDA and test on small toy graphs or matrices to verify the correctness.

### April 10th - April 16th

Implement sparse matrix-vector multiplication on GPU with CUDA and integrate it into our Lanczos algorithm so that we can deal with large sparse graphs or matrices.

### April 17th - April 23rd

Experiment on our algorithms with large real-world graphs and do some performance fine-tuning. Compare our solution to some well-known CPU-based solutions such as those in Python SciPy and MATLAB.

### April 24th - April 30th

Apply our solution to some real-world problems such as graph visualization or image segmentation.

### May 1st - May 7th

Finalize our report. Also keep some buffer time in case some of our previous tasks takes too long.

### May 8th - May 11th

Prepare for the final presentation.

## PLATFORM CHOICE

We choose to use the GPU on GHC machines, because that is enough for interesting graphs with reasonable sizes, and we want to focus on speeding up eigensolvers on a single machine with GPUs in this project.

## REFERENCES

[1] Lanczos, C. (1950). An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. Journal of Research of the National Bureau of Standards, 45(4), 255-282.

[2] Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416.

[3] Weiss, Y. (1999). Segmentation using eigenvectors: a unifying view. In Computer vision, 1999. The proceedings of the seventh IEEE international conference on (Vol. 2, pp. 975-982). IEEE.

[4] Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation. Neural computation, 15(6), 1373-1396.

[5] Matam, K. K., & Kothapalli, K. (2011, March). GPU accelerated Lanczos algorithm with applications. In Advanced Information Networking and Applications (WAINA), 2011 IEEE Workshops of International Conference on (pp. 71-76). IEEE.

[6] Wu, K., & Simon, H. (1999). A parallel Lanczos method for symmetric generalized eigenvalue problems. Computing and Visualization in Science, 2(1), 37-46.
