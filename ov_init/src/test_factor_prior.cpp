/* Copyright (C) 2026 Joao Leonardo Silva Cotta
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#ifdef OV_TEST_CERES
#include "ceres/Factor_GenericPrior.h"
using Prior = ov_init::Factor_GenericPrior;
#else
#include "ceres_free/Factor_GenericPrior.h"
using Prior = ov_init::zbft_sfm::Factor_GenericPrior;
#endif
#include "utils/finite.h"

namespace {
int checks = 0, failures = 0;
void check(bool good, const char *message) {
  ++checks;
  if (!good) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
void dense_oracle(int n, double scale, bool zero_gradient) {
  std::vector<std::string> types;
  std::vector<int> sizes;
  if (n == 12) { types = {"vec3", "vec1", "vec8"}; sizes = {3, 1, 8}; }
  else { types = {"vec" + std::to_string(n)}; sizes = {n}; }
  Eigen::MatrixXd A(n,n), H;
  Eigen::VectorXd g(n), origin(n), point(n);
  for (int i=0; i<n; ++i) {
    origin(i) = .05 * (i+1); g(i) = zero_gradient ? 0. : .4 * std::cos(.7*(i+1));
    for (int j=0; j<n; ++j) A(i,j) = std::sin(.3*(i+1)*(j+2)) + (i == j ? 2. : 0.);
  }
  H = scale * (A.transpose()*A + .3 * Eigen::MatrixXd::Identity(n,n));
  g *= scale;
  Prior factor(origin, types, H, g);
  std::vector<const double *> params;
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> js;
  std::vector<double *> jac;
  int offset = 0; point = origin;
  for (int size : sizes) { params.push_back(point.data()+offset); js.emplace_back(n,size); offset += size; }
  for (auto &m : js) jac.push_back(m.data());
  Eigen::VectorXd res(n); Eigen::MatrixXd J(n,n);
  check(factor.Evaluate(params.data(), res.data(), jac.data()), "finite SPD prior evaluates");
  offset=0; for (auto &m : js) { J.middleCols(offset,m.cols())=m; offset += m.cols(); }
  check((J.transpose()*J-H).norm() < 2e-11*std::max(1.,H.norm()), "JtJ equals supplied information");
  check((J.transpose()*res-g).norm() < 2e-11*std::max(1.,g.norm()), "Jtr equals supplied nonzero gradient");
  check((factor.sqrtI.transpose()*factor.b-g).norm() < 2e-11*std::max(1.,g.norm()), "U-transpose b equals g");
  const double cost0 = res.squaredNorm();
  Eigen::VectorXd delta(n); for (int i=0;i<n;++i) delta(i)=.07*std::sin(i+.2);
  point = origin + delta;
  check(factor.Evaluate(params.data(),res.data(),nullptr), "offset prior evaluates");
  double expected = delta.dot(H*delta) + 2.*g.dot(delta);
  check(std::abs(res.squaredNorm()-cost0-expected) < 2e-10*std::max(1.,std::abs(expected)), "cost change matches dense quadratic");
  point = origin - H.ldlt().solve(g);
  check(factor.Evaluate(params.data(),res.data(),nullptr), "dense optimum evaluates");
  check((J.transpose()*res).norm() < 2e-10*std::max(1.,g.norm()), "dense optimum is stationary");
}
void reject(const Eigen::MatrixXd &H, const Eigen::MatrixXd &g, const Eigen::MatrixXd &origin) {
  Prior factor(origin, {"vec3"}, H, g);
  double p[3]={0,0,0}, residual[3]={17,19,23}, J[9];
  const double *params[]={p}; double *jac[]={J};
  for (double &v:J) v=31;
  check(!factor.Evaluate(params,residual,jac), "invalid prior fails recoverably");
  check(residual[0]==17 && residual[1]==19 && residual[2]==23, "invalid prior does not publish residuals");
  bool unchanged=true; for (double v:J) unchanged &= v==31;
  check(unchanged, "invalid prior does not publish Jacobians");
}
}
int main(int argc,char **argv) {
  for (int n : {1,3,8,12}) for (double scale:{1e-4,1.,1e4}) for (bool zero:{false,true}) dense_oracle(n,scale,zero);
  if (argc < 2 || std::string(argv[1]) != "--positive-only") {
    Eigen::MatrixXd H=Eigen::Matrix3d::Identity(), g=Eigen::Vector3d::Ones(), x=Eigen::Vector3d::Zero();
    reject(-H,g,x); reject(Eigen::Matrix3d::Zero(),g,x);
    Eigen::MatrixXd bad=H; bad(2,2)=0; reject(bad,g,x);
    bad=H; bad(0,1)=.2; reject(bad,g,x);
    for (double value : {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
      check(!ov_core::numeric::finite(value), "binary64 nonfinite guard survives fast math");
      bad=H; bad(0,0)=value; reject(bad,g,x);
      bad=g; bad(1)=value; reject(H,bad,x);
      bad=x; bad(2)=value; reject(H,g,bad);
    }
    reject(Eigen::MatrixXd::Identity(2,2),g,x);
    reject(H,Eigen::MatrixXd::Zero(2,1),x);
    reject(H,g,Eigen::MatrixXd::Zero(2,1));
  }
  std::printf("factor prior: %d/%d checks passed\n",checks-failures,checks);
  return failures ? 1 : 0;
}
