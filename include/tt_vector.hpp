#ifndef TT_VECTOR_HPP
#define TT_VECTOR_HPP

#include <armadillo>
#include <cmath>

/**
 * TT representation of a multidimensional vector
 */
class TtVector {
 public:
  /**
   * Default constructor.
   */
  TtVector() = default;

  /**
   * Construct a TT-vector from the dimensions and TT ranks. The cores are all
   * initialized to zeros.
   */
  TtVector(const arma::Col<arma::uword> &dims,
           const arma::Col<arma::uword> &ranks);

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

  /**
   * Right side scalar multiplication for TT-vector.
   */
  TtVector operator*(double constant) const;

  /**
   * TT-vector addition.
   */
  TtVector operator+(const TtVector &other) const;

  /**
   * TT-vector unary minus operator.
   */
  TtVector operator-() const { return *this * (-1.0); }

  /**
   * TT-vector substraction.
   */
  TtVector operator-(const TtVector &other) const { return *this + (-other); }

  /**
   * TT-vector dot product.
   */
  double Dot(const TtVector &other) const;

  /**
   * TT-vector norm.
   */
  double Norm() const { return std::sqrt(this->Dot(*this)); }

 private:
  arma::uword ndim_;
  arma::Col<arma::uword> dims_;
  arma::Col<arma::uword> ranks_;
  arma::uword maxrank_;
  arma::field<arma::Cube<double>> cores_;
};

/**
 * Left side scalar multiplication for TT-vector.
 */
inline TtVector operator*(double constant, const TtVector &vector) {
  return vector * constant;
}

/**
 * Dot product betweent two TT-vectors.
 */
inline double Dot(const TtVector &vector1, const TtVector &vector2) {
  return vector1.Dot(vector2);
}

/**
 * Norm of a TT-vector
 */
inline double Norm(const TtVector &vector) { return vector.Norm(); }

#endif
