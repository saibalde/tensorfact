#ifndef TT_VECTOR_HPP
#define TT_VECTOR_HPP

#include <armadillo>
#include <cmath>

/**
 * TT representation of a multidimensional vector
 *
 * A TT-vector is a memory-efficient representation of a multidimensional array
 * \f$v(i_0, \ldots, i_{d - 1})\f$ where each of the entries are computed as
 * \f[
 *     v(i_0, \ldots, i_{d - 1}) = v_0(i_0) \cdots v_{d - 1}(i_{d - 1})
 * \f]
 * Here \f$v_k(i_k)\f$ is the \f$i_k\f$-th slice of the 3D array \f$v_k\f$, also
 * referred to as the \f$k\f$-th TT core of \f$v\f$. Each of these slices are
 * \f$r_k \times r_{k + 1}\f$ dimensional matrices, with \f$r_0 = r_d = 1\f$.
 * Assuming \f$n_k \sim n\f$ and \f$r_k \sim r\f$, this reduces the storage
 * complexity \f$\mathcal{O}(n^d)\f$ of the full tensor to
 * \f$\mathcal{O}(d n r^2)\f$ in the TT format.
 */
class TtVector {
 public:
  /**
   * Default constructor.
   */
  TtVector() = default;

  /**
   * Construct a TT-vector from the cores.
   */
  TtVector(const arma::field<arma::Cube<double>> &cores);

  /**
   * Default destructor.
   */
  ~TtVector() = default;

  /**
   * Return the dimensionality of the TT-vector.
   */
  arma::uword NumDims() const { return ndim_; }

  /**
   * Return the size of the TT-vector.
   */
  const arma::Col<arma::uword> &Dims() const { return dims_; }

  /**
   * Return the TT-ranks of the TT-vector.
   */
  const arma::Col<arma::uword> &Ranks() const { return ranks_; }

  /**
   * Return the specified core of the TT-vector.
   */
  const arma::Cube<double> &Core(arma::uword i) const { return cores_(i); }

  /**
   * Compute and return the entry of the TT-vector at given index.
   */
  double operator()(const arma::Col<arma::uword> &index) const;

  /**
   * TT-vector addition.
   */
  friend TtVector operator+(const TtVector &vector1, const TtVector &vector2);

  /**
   * Left side scalar multiplication of TT-vector.
   */
  friend TtVector operator*(double constant, const TtVector &vector);

  /**
   * TT-vector dot product.
   */
  friend double Dot(const TtVector &vector1, const TtVector &vector2);

 private:
  arma::uword ndim_;
  arma::Col<arma::uword> dims_;
  arma::Col<arma::uword> ranks_;
  arma::field<arma::Cube<double>> cores_;
};

/**
 * Norm of a TT-vector
 */
inline double Norm(const TtVector &vector) {
  return std::sqrt(Dot(vector, vector));
}

#endif
