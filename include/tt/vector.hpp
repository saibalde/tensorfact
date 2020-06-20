#ifndef TT_VECTOR_HPP
#define TT_VECTOR_HPP

#include <armadillo>

namespace tt {

/**
 * TT representation of a multidimensional vector
 */
class Vector {
 public:
  /**
   * Default constructor.
   */
  Vector() = default;

  /**
   * Construct a TT-vector from the dimensions and TT ranks. The cores are all
   * initialized to zeros.
   */
  Vector(const arma::Col<arma::uword> &dims,
         const arma::Col<arma::uword> &ranks);

  /**
   * Construct a TT-vector from the cores.
   */
  Vector(const arma::field<arma::Cube<double>> &cores);

  /**
   * Default destructor.
   */
  ~Vector() = default;

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
   * Return the maximum TT-rank of the TT-vector.
   */
  arma::uword MaxRank() const { return maxrank_; }

  /**
   * Return the specified core of the TT-vector.
   */
  const arma::Cube<double> &Core(arma::uword i) const { return cores_(i); }

  /**
   * Compute and return the entry of the TT-vector at given index.
   */
  double operator()(const arma::Col<arma::uword> &index) const;

 private:
  arma::uword ndim_;
  arma::Col<arma::uword> dims_;
  arma::Col<arma::uword> ranks_;
  arma::uword maxrank_;
  arma::field<arma::Cube<double>> cores_;
};

}  // namespace tt

#endif
