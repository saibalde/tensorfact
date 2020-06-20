#ifndef TT_VECTOR_HPP
#define TT_VECTOR_HPP

#include <armadillo>

namespace tt {

class Vector {
 public:
  Vector() = default;

  Vector(const arma::Col<arma::uword> &dims,
         const arma::Col<arma::uword> &ranks);

  Vector(const arma::field<arma::Cube<double>> &cores);

  ~Vector() = default;

  arma::uword NumDims() const { return ndim_; }

  const arma::Col<arma::uword> &Dims() const { return dims_; }

  const arma::Col<arma::uword> &Ranks() const { return ranks_; }

  arma::uword MaxRank() const { return maxrank_; }

  const arma::Cube<double> &Core(arma::uword i) const { return cores_(i); }

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
