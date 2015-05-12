---
layout: index
---

# Project Final Report

## SUMMARY

We implemented a parallel program to calculate matrix eigenvalues in CUDA on GPU using the Lanczos algorithm. We also implemented efficient matrix-vector multiplication in CUDA which serves as a basis for our eigenvalue computation.

## BACKGROUND

Eigendecomposition is an important procedure for graph spectral analysis, where we investigate the eigenvalues (and eigenvectors) for some matrix \\(M\\) (e.g., adjacency matrix) associated with an undirected graph \\(G = (V, E)\\). Eigendecomposition not only helps us gain insights into the structure of large graphs (e.g., social networks), but also has many interesting applications such as spectral clustering and image segmentation. Specifically, in this project we consider the eigenvalue calculation problem for a symmetric real matrix \\(M\\) and find \\(\lambda\\) such that \\[M\mathbf{x} = \lambda \mathbf{x}\\] for some vector \\(\mathbf{x}\\). Most of the time we are only interested in the largest few eigenvalues of \\(M\\).

As is shown in the following figure, one way to perform eigendecomposition is to first transform the matrix \\(M\\) into a symmetric tridiagonal matrix \\(T\_m\\) and then find the eigenvalues of \\(T\_m\\), for which efficient algorithms exists [1]. The size \\(m\\) of \\(T\_m\\) depends on the number of eigenvalues and desired precision we want to obtain. As we increase \\(m\\), the eigenvalues of \\(T\_m\\) converge to those of \\(M\\).

![Eigendecomposition Overview]({{site.baseurl}}/assets/final/eigen.svg)

The Lanczos algorithm [2] is a well-known algorithm for matrix tridiagonalization. The algorithm proceeds as follows.

 *  \\(\mathbf{v}\_0 \gets \mathbf{0}\\), \\(\mathbf{v}\_1 \gets \text{norm-1 random vector}\\), \\(\beta\_1 \gets 0\\)
 *  for \\(j = 1,\dots,m\\)
     *   \\(\mathbf{w}\_j \gets M\mathbf{v}\_j\\)
     *   \\(\alpha\_j \gets \mathbf{w}\_j^{\top}\mathbf{v}\_j\\)
     *   if \\(j < m\\)
         *   \\(\mathbf{w}\_j \gets \mathbf{w}\_j - \alpha\_j \mathbf{v}\_j - \beta\_j \mathbf{v}\_{j-1}\\)
         *   \\(\beta\_{j+1} \gets \\|\mathbf{w}\_j\\|\_2\\)
         *   \\(\mathbf{v}\_{j+1} \gets \mathbf{w}\_j / \beta\_{j+1}\\)

We get the result of tridiagonalization as
\\[
T\_m = \left(
\\begin{array}{c c c c}
\alpha\_1 & \beta\_2 & & \\\\
\beta\_2 & \alpha\_2 & \ddots & \\\\
 & \ddots & \ddots & \beta\_m \\\\
 & & \beta\_m & \alpha\_m \\\\
\\end{array}
\right).
\\]

## APPROACH

As we can see from the Lanczos algorithm, the iteration involves several matrix and vector operations. These operations includes vector dot-product, \\(\ell\_2\\) norm and SAXPY, whose parallelization on CUDA is simple and intuitive. However, it is not obvious how to gain high performance for matrix-vector multiplication, so this forms the first part of our project. Also we note that the characteristics of matrix \\(M\\) is closely tied to the structure of the corresponding graph. Most graphs (or networks) in real world are sparse and has a power law node degree distribution. So matrix \\(M\\) is highly sparse and the sparsity pattern for each row is very different. We must take this into account when we implement our parallel matrix operations.

### Sparse Matrix-Vector Multiplication (SPMV)

We use the compressed sparse row (CSR) format for sparse matrix storage in memory. As is shown in the following figure, in CSR format an array `column_index` stores consecutively all nonzero column indices for each row. We store the value of corresponding nonzero elements in an array `data` of the same length. Also, we use an array `row_pointer` to indicate the start position for each row in `column_index`. Using the CSR storage, we can carry out matrix-vector multiplication in a row-to-row basis.

![CSR Matrix-Vector Multiplication]({{site.baseurl}}/assets/final/csr-spmv.svg)

The sparse matrix-vector multiplication takes the largest portion of computation time in one iteration of the Lanczos algorithm, so we want to make it as efficient as possible through parallelization. The naive way to do this comes very naturally. We just use each CUDA thread to deal with one row of the matrix as is shown in the following figure.

![Work Assignment for Naive SPMV]({{site.baseurl}}/assets/final/naive-spmv.svg)

One potential problem of this method of parallelization is work imbalance. As we have mentioned above, the distribution of nonzero entries in the matrix rows can be very skewed. So some threads dealing with the long rows will take a long time and slow down the whole execution. Another potential problem of the naive method is cache locality. The 32 threads running in a warp are executing synchronously while accessing different rows in the matrix which may layout far apart from each other in memory. Thus the cache lines are likely to be read in and invalidated all the time.

To address the two problems in the naive method, we first adopted a warp-based assignment for rows [5]. In the warp-based method, each warp, which consists of 32 threads running synchronously in SIMD, is assigned to deal with one row. Each thread in the warp would deal with some columns and write the partial results to the shared memory. When all the threads in a warp are done, the partial results are summed into the final value and stored in the destination vector. This warp-based work assignment decreases the impact of work imbalance, because now long rows are handled by a warp of threads instead of only one thread, which makes it less different from short rows. The method also solves the cache locality problem, because now the threads in a warp would access adjacent areas of the memory (for the same row), which increases cache locality for data accessing.

![Work Assignment for Warp-based SPMV]({{site.baseurl}}/assets/final/warp-spmv.svg)

However, another significant problem arises due to wasting of computation units. For a long row, 32 threads in a warp would all be busy doing useful computation. For a short row, on the other hand, only a small number of threads in a warp would be actually working. The slowdown due to this wasting of computation can even diminish the speedup gained by less significant work imbalance impact and better cache locality. As a result, we finally decided to use a group-based work assignment. It is very similar to the warp-based work assignment except that instead of scheduling a fixed number (warp size of 32) of threads to deal with each row, we first estimate the number of nonzero elements per row by dividing the total number of nonzero elements in the matrix by the row count. Then we assign a group of threads which has the size of the smallest power of 2 greater than the estimated number of nonzero values per row. In this way, a warp can deal with multiple rows without wasting much of its computation while still getting some of the benefits of low workload imbalance and cache locality. This is our final solution for sparse matrix-vector multiplication.

![Work Assignment for Row-Group SPMV]({{site.baseurl}}/assets/final/group-spmv.svg)

On the way to this final solution, we have also tried two other methods of work assignments. The first is two-level kernel launch (dynamic parallelism). The first level kernel function would check the average number of nonzero values per row within a small range of rows and launch a second-level kernel function which would deal with this range of rows with a thread group size nearest to the average nonzero element size per row as we discussed above. We thought this method would give a better balanced work assignment because now the granularity of calculating the average row size is smaller. However, the overhead of two-level kernel launch is so high that we see the performance even drops. The other solution that we have tried is to parallelize over nonzero entries in the matrix instead of over rows. The problem of this method is that when multiple threads that deal with different entries for the same row want to update their results to the target vector, `atomic_add` must be used to ensure correctness. This greatly decrease the performance.

### The Lanczos Algorithm Implementation
 
Based on our CUDA SPMV routine, we have implemented the Lanczos tridiagonalization in CUDA. Each iteration consists of several kernel launches for a series of matrix/vector operations. The output of the CUDA Lanczos algorithms are diagonal elements \\(\alpha\_j\\) (\\(j=1,\dots,m\\)), and sub-diagonal elements \\(\beta\_j\\) (\\(j=1,\dots,m-1\\)) for \\(T\_m\\). Then we solve eigenvalues of \\(T\_m\\) to get an approximation of the eigenvalues of \\(M\\). Typically \\(m\\) is linear to the number \\(k\\) of eigenvalues we want to compute. Since \\(k\\) is often small even for a very large graphs, eigendecomposition for \\(T\_m\\) is trivial and can be done on CPU using some efficient algorithms such as [1].

We have also implemented matrix-vector procedures and the Lanczos algorithm on CPU to verify the correctness of our CUDA implementation as well as for performance comparison.

## EXPERIMENT SETUP

To better evaluate the performance of our SPMV routine, we first use generated random graphs so that we can control the size of the matrix while preserving the characteristics of adjacency matrices of scale-free networks (e.g. social networks) whose node degree distribution follow a power-law. We adopted the Barabási-Albert model [3] and set the average node degree to 3. We generated a set of graphs with node count ranging from 200,000 to 1,600,000 using the Python package [NetworkX](https://networkx.github.io).

For evaluation purpose we use two real-world datasets from the [SNAP](http://snap.stanford.edu) project. The as-Skitter dataset is an undirected Internet topology graph containing 1,696,415 nodes and 11,095,298 edges. The cit-Patents dataset is a directed citation network containing 3,774,768 nodes and 16,518,94 edges. We use the symmetrized adjacency matrix from both graphs in our experiments. Note that the number of nonzero entries in the matrix is twice the number of edges in the graph.

We perform all our experiments on Amazon Web Service EC2 `g2.2xlarge` instances. The instance has NVIDIA GK104 GPU with 1,536 CUDA cores and CUDA 7.0 Toolkit installed. The instance also has Intel Xeon E5-2670 CPU with 8 cores and with gcc/g++ 4.8.2 installed. We have also tested our programs in `ghc41` machine with NVIDIA GTX 780 GPU. The observation there is consistent with our experiment results on Amazon EC2.

We compare different versions of our SPMV routine to the implementation in [cuSparse](http://docs.nvidia.com/cuda/cusparse/) (`cusparseScsrmv` and `cusparseDcsrmv`). We record the kernel launch time for comparison. Both single-precision (`float`) and double-precision (`double`) data are tested.

We compare our GPU-accelerated Lanczos algorithm with our single-threaded C++ CPU implementation compiled with `-O3` optimization. We record the running time for the Lanczos algorithm at a fixed number of iterations on double-precision floating point data. We also compare our result with the running time of some CPU-based parallel eigensolver such as an MPI-based package [SLEPc](http://slepc.upv.es).

## RESULTS

We first show the sparse matrix-vector multiplication using our final solution (assigning work of row groups to warps), the cuSparse solution and our naive solution over generated graph data. The following graph shows the speedup for various GPU implementations. The data is single-precision. And the baseline is our sequential CPU implementation. Our final solution has similar performance compared with the cuSparse solution. Note that when graph node count reaches 3,200,000, the naive method performs surprisingly better than the other two methods. We suspect that it is because we are experimenting with generated data, and the generated data fail to remain a skew distribution of nonzero values per row when the row size is too large. Because of that, the naive method wins for its simplicity (less overhead).

![Single-precision SPMV Speedup over CPU]({{site.baseurl}}/assets/final/spmv-float-speedup.svg)

We also present the sparse matrix-vector multiplication for double-precision generated data on our final solution, the cuSparse solution and our naive solution. As we can see, our method performs better than cuSparse.

![Double-precision SPMV Speedup over CPU]({{site.baseurl}}/assets/final/spmv-double-speedup.svg)

The same comparison between the three methods is also conducted over real-world matrices. As we can see, our method still has better performance than cuSparse on large data.

![Read-world Graph SPMV Speedup over CPU]({{site.baseurl}}/assets/final/datasets-spmv.svg)

To sum up, our sparse matrix-vector multiplication is able to gain similar or even better performance on large generated or real-world graphs compared with the sparse matrix-vector multiplication provided by cuSparse.

Finally, we show the comparison of eigensolver running time for our CUDA Lanczos implementation and sequential CPU implementation. We calculate 10 eigenvalues for both graphs, fixing the number of iterations to 63.

![Eigensolver Running Time on GPU and CPU]({{site.baseurl}}/assets/final/datasets-lanczos.svg)

We can see that our GPU implementation achieved 10x speedup over our CPU eigensolver. However, the result may not be convincing as one may wonder how our GPU implementation perform if we compare it with some highly optimized parallel implementation on CPU. The difficulty for such comparison is that we are not able to find a software package that support the exact Lanczos procedure as ours. Existing scientific libraries often uses different eigendecomposition algorithms that have different time complexity and convergence rate, so it makes no sense to carry out direct comparison. The closest counterpart is the Lanczos solver in SLEPc, but it runs very slow. It took 84.9 sec to solve 10 largest eigenvalues for the cit-Patents graph, while we took only 31.8 sec even on CPU. We suspect that the Lanczos solver in SLEPc uses some post-processing steps to ensure numerical stability (see below), while our Lanczos solver is much simpler and has slightly worse accuracy (about 1e-3 to 1e-6 error rate for `double` matrices).

Although we are not able to conduct experiments on parallel CPU implementation, we argue that our GPU implementation has achieved high performance. This is because the bottleneck in Lanczos iteration is matrix-vector multiplication, which takes about 70% of the time in each iteration (after optimization) as we have measured, and our SPMV routine has already beat the highly optimized cuSparse implementation on large graphs.

## DISCUSSION

In this project we have implemented a simple form of the Lanczos algorithm. However, many research shows that special care must be taken to ensure the correctness of the algorithm. Due to limited machine precision for floating point numbers, the set of basis \\(\mathbf{v}\_j\\) many not be orthogonal as is expected. One many need to perform reorthogonalization or selective reorthogonalization on the basis every few steps of the iterations, but this procedure is computationally expensive and diminishes the speedup gained by parallelization on CUDA. A totally different approach ignores the numeric precision issues during the iteration, and performs the Cullum-Willoughby test [4]. This test compares the eigenvalues of \\(T\_m\\) and \\(\hat{T}\_m\\) (\\(T\_m\\) with the first row and first column removed), and identifies possible “spurious” eigenvalues arises due to loss of orthogonality. Our implementation has taken the latter approach, and can thus highly benefit from GPU parallelism.

As we have mentioned above, one possible reason that limits the speedup for SPMV is work imbalance. Even our final solution using row-group work assignment is not able to achieve ideal performance as we do not have complete knowledge for row sparsity pattern. Another reason is that the access to the vector being multiplied is nearly random, no matter how we schedule the calculation, so it is hard to exploit locality to speed up execution.

Also, we have noticed that machine precision has a great impact on the convergence of eigenvalue computation. Single-precision calculation often yields poor accuracy, so we have to use `double` data type for our Lanczos iteration. However, double-precision arithmetic operations are slower than single-precision ones due to GPU hardware design. So our implementation will run better on high-end GPUs which have better double-precision support.

Our Lanczos solver support efficient calculation of eigenvalues with acceptable accuracy on large graphs. Compared with eigensolvers in MPI, although our program is not able to handle graphs that do not fit into memory, the CUDA implementation has its advantage of high efficiency, which allows for applications such as real-time image segmentation.

## CONCLUSION

We have successfully achieved the goal mentioned in our proposal. Although we do not have time to develop interesting applications for our eigensolver (which is optional as we have planned), we have conducted detailed experiments to analyse the performance of our sparse matrix-vector routines and Lanczos eigensolver. Equal work was performed by both team members.

## REFERENCES

[1] Reinsch, C. H. (1973). Algorithm 464: eigenvalues of a real, symmetric, tridiagonal matrix. Communications of the ACM, 16(11), 689.

[2] Lanczos, C. (1950). An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. Journal of Research of the National Bureau of Standards, 45(4), 255-282.

[3] Barabási–Albert model. <http://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model>.

[4] Cullum, J. K., & Willoughby, R. A. (2002). Lanczos Algorithms for Large Symmetric Eigenvalue Computations: Vol. 1: Theory (Vol. 41). SIAM.

[5] Bell, N., & Garland, M. (2008). Efficient sparse matrix-vector multiplication on CUDA (Vol. 2, No. 5). Nvidia Technical Report NVR-2008-004, Nvidia Corporation.
