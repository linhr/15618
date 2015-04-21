---
layout: index
---

# Project Checkpoint Report

## PROGRESS OVERVIEW

So far we have roughly followed our schedule in our project proposal. Here we list what we have done in the past few weeks.

 *  We have read many related papers to understand the Lanczos algorithm and some other required steps for the eigendecomposition problem.
 *  We have generated some random graphs using the Python package [NetworkX](https://networkx.github.io/) for our experiments. We have also selected some real-world graphs to use from the [SNAP](http://snap.stanford.edu/) project.
 *  We have implemented the compressed sparse row (CSR) representation for sparse matrices.
 *  We have implemented several matrix and vector operations in both C++ and CUDA, including vector-vector dot product, vector-scalar multiplication, vector-scalar addition, SAXPY, vector L2 norm and matrix-vector multiplication. These operations are essential for the implementation of the Lanczos algorithm. All of them have been tested for correctness.
 *  We have implemented a primary version of the Lanczos algorithm in CUDA. The correctness and performance of this implementation remains to be tested.

In our proposal we originally plan to work out the Lanczos algorithm on dense graphs first. However, we found it is not necessary as the Lanczos algorithm is ignorant of the representation of matrices and it is not that hard to deal with sparse matrices from the beginning. Therefore, we skipped this first step and started working on sparse matrices directly.

Our work on the matrix-vector multiplication function in CUDA is worth detailed mentioning. This function, which can be the bottleneck of the Lanczos algorithm, has been implemented in two ways and tested for performance. The naive method deals with each row in the matrix using a CUDA thread. This method may result in work imbalance and discontinuous memory read which can slow down the execution. The other method, which is the "warp" method [1], deals with each row in the matrix using a warp of threads. When doing matrix-vector multiplication on the adjacency matrix of a 10,000-node complete graph and a 10,000-element random vector, the naive CUDA method and the "warp" CUDA method was able to achieve 1.18x and 24.08x speedup compared with the CPU-based method. When doing matrix-vector multiplication on the adjacency matrix of a generated 400,000-node social network graph and a 400,000-element random vector, the naive CUDA method and the "warp" CUDA method was able to achieve 2.80x and 3.04x speedup compared with the CPU-based method. Note that we only measure kernel execution time and ignore data transfer time between the device and the host. This is fair, because in the Lanczos algorithm we only need to copy the input data into the device once at the beginning. After that, many iterations of matrix-vector multiplication within the device memory are executed, so the data transfer time can be amortized.

## GOALS AND DELIVERABLES

As we have mentioned in our project proposal, for the parallelism competition we plan to show the speedup of our CUDA implementation of the Lanczos algorithm compared with some CPU implementation. Based on our experience so far, we find that it is also important to benchmark each matrix-vector operation involved, and we will present our findings in the competition. We will also give a detailed analysis of the performance of the Lanczos algorithm, based on our measured time break-down of each stage of the algorithm.

If we have time, we will apply our implementation to solve real world problems such as image segmentation and spectral clustering. We are still looking for interesting applications that can best demonstrate our work in the project.

## KNOWN ISSUES

After investigating the Lanczos algorithm, we found that this algorithm do not directly solve the eigendecomposition problem. Rather, it decompose a symmetric matrix \\(A\\) into the form \\(Q^{\top}AQ\\) where \\(Q\\) is an orthogonal matrix and \\(T\\) is a symmetric tridiagonal matrix. We need to solve the eigendecomposition problem for \\(T\\) and use the eigenvalues of \\(T\\) to approximate the eigenvalues of \\(A\\). One way is to use the QR algorithm, and fast QR decomposition algorithm exists for symmetric tridiagonal matrices [2]. However, this algorithm is not easy to parallelize as each iteration depends on the result of the previous one. We plan to evaluate the impact of this problem on the overall performance.

Also, as is mentioned in the literature, the Lanczos algorithm may not produce correct results due to limited precision of floating point arithmetic. Some forms of post-processing (reorthogonalization) is needed after certain numbers of iterations. We may need to implement this procedure in CUDA to ensure the numerical stability of the computation.

## REFINED SCHEDULE

Here we present an updated schedule of the second half of our project. Since eigendecomposition based on the Lanczos algorithm is more complicated than we have thought before, we plan to allot more time to create a robust and efficient implementation.

### April 21st - April 27th

Finish Lanczos algorithm on both CPU and GPU. Experiment on our implementation with large real-world graphs and do some performance fine-tuning. Also compare our solution to some well-known CPU and GPU based solutions.

### April 28th - May 4th

(Optional) Apply our solution to some real-world problems. Measure the performance gain of our CUDA implementation.

### May 5th - May 7th

Finalize our report. Create graphs or tables showing the result of our benchmark. Summarize our approach for the CUDA implementation and analyze performance.

### May 8th - May 11th

Prepare for the final presentation.

## REFERENCES

[1] Bell, N., & Garland, M. (2008). Efficient sparse matrix-vector multiplication on CUDA (Vol. 2, No. 5). Nvidia Technical Report NVR-2008-004, Nvidia Corporation.

[2] Ortega, J. M., & Kaiser, H. F. (1963). The LLT and QR methods for symmetric tridiagonal matrices. The Computer Journal, 6(1), 99-101.
