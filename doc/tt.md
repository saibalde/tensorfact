@page tt Tensor-Train Decomposition

### Computing Tensor-Train Decomposition from Full Tensor

### Computing Tensor-Train Decomposition from Tensor Entry Evaluations

### Rounding in Tensor-Train Format

### Fast Operations in Tensor-Train Format

One of the main advantages of the TT format is that it allows us to accelerate
common linear algebra operations. For instance, given the TT cores of two
tensors \f$\mathcal{X}\f$ and \f$\mathcal{Y}\f$ of the same size we can easily
construct the TT cores of their sum \f$\mathcal{Z} = \mathcal{X} +
\mathcal{Y}\f$ as
\f[
\mathbf{Z}_1(i_1) =
\begin{bmatrix}
    \mathbf{X}_1(i_1) & \mathbf{Y}_1(i_1)
\end{bmatrix},
\quad
\mathbf{Z}_k(i_k) =
\begin{bmatrix}
    \mathbf{X}_k(i_k) &                   \\
                      & \mathbf{Y}_k(i_k)
\end{bmatrix},
\quad
\mathbf{Z}_d(i_d) =
\begin{bmatrix}
    \mathbf{X}_d(i_d) \\
    \mathbf{Y}_d(i_d)
\end{bmatrix}
\f]
For a complete description of similar operations, see
[Oseledets (2011)](https://epubs.siam.org/doi/abs/10.1137/090752286).
