@mainpage Tensor Factorization Library

Computationally, tensors are simply multi-dimensional arrays. The entries of a
\f$n_1 \times \cdots \times n_d\f$ tensor \f$\mathcal{X}\f$ are indexed by
\f$\mathcal{X}(i_1, \ldots, i_d)\f$, where we have \f$0 \leq i_k \leq n_k - 1\f$
for \f$1 \leq k \leq d\f$. However, storing all the entries of such a tensor
requires \f$\mathcal{O}(n^d)\f$ memory where \f$n = \max\{n_1, \ldots, n_d\}\f$.
Clearly, as the dimensionality \f$d\f$ increases, the memory requirements
increases rapidly. Efforts to overcome this curse of dimensionality has led to
various tensor factorizations.

### Cannonical Polyadic (CP) Decomposition

The canoncial polyadic factorization decomposes a \f$d\f$-dimensional tensor as
\f[
\mathcal{X}(i_1, \ldots, i_d) = \sum_{\alpha = 0}^{r - 1} \mathbf{X}_1(i_1,
\alpha) \cdots \mathbf{X}_d(i_d, \alpha)
\f]
The components \f$\mathbf{X}_1, \ldots, \mathbf{X}_d\f$ are called the factor
matrices, and the summand limit \f$r\f$ is called the rank of the CP
decomposition. We note that the CP format requires \f$\mathcal{O}(d n r)\f$
memory for storage.

### Tucker Decomposition

### Tensor-train (TT) Decomposition

The tensor-train factorization decomposes a \f$d\f$-dimensional tensor as
\f[
\mathcal{X}(i_1, \ldots, i_d) = \sum_{\alpha_0 = 0}^{r_0 - 1} \cdots
\sum_{\alpha_d = 0}^{r_d - 1} \mathcal{X}_1(\alpha_0, i_1, \alpha_1) \cdots
\mathcal{X}_d(\alpha_{d - 1}, i_d, \alpha_d)
\f]
This factorization can also be expressed in matrix form as
\f[
\mathcal{X}(i_1, \ldots, i_d) = \mathbf{X}_1(i_1) \cdots \mathbf{X}_d(i_d),
\quad \mathbf{X}_k(i_k) = \mathcal{X}_k(:, i_k, :) \in \mathbb{R}^{r_{k - 1}
\times r_k}
\f]
The three-dimensional tensors \f$\mathcal{X}_1, \ldots, \mathcal{X}_d\f$ are
called the TT cores, and the summand limits \f$\{r_0, \ldots, r_d\}\f$ are
called TT-ranks of this decomposition. By convention, we choose \f$r_0 = r_d =
1\f$. Denoting the maximal TT-rank as \f$r = \max\{r_1, \ldots, r_{d - 1}\}\f$,
we note that the TT format requires \f$\mathcal{O}(d n r^2)\f$ memory for
storage.

### Hierarchical Tucker (H-Tucker) Decomposition
