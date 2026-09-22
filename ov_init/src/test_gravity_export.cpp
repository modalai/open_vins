/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "dynamic/GravityAlignment.h"
#include "ceres_free/CostFunction.h"
#include "ceres_free/LocalParameterization.h"
#include "ceres_free/Problem.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace ov_core;
using namespace ov_init;
using namespace ov_init::zbft_sfm;
namespace {
using Mat=Eigen::MatrixXd;
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
int checks=0,failures=0;
double max_fd=0.,max_cov_fd=0.,max_export=0.,max_old_error=0.;
void check(bool pass,const char *why){++checks;if(!pass){++failures;std::printf("FAIL: %s\n",why);}}
M3 align(const V3&g) {
  const V3 u=g.normalized(),z=V3::UnitZ();
  const V3 axis=u.cross(z);
  const double angle=std::atan2(axis.norm(),u.dot(z));
  return axis.norm()>1e-12?M3(Eigen::AngleAxisd(angle,axis.normalized())):M3::Identity();
}
struct Means {
  std::array<M3,3> rotation;
  std::array<V3,3> position;
  V3 velocity{.7,-.2,.1};
  Means(){for(int i=0;i<3;++i){rotation[i]=exp_so3(V3(.11+.07*i,-.23+.03*i,.14-.05*i));position[i]=V3(.3+.2*i,-.4+.13*i,.2-.12*i);}}
};
Mat output_jacobian(const V3&g,const Means&m,bool warm) {
  GravityS2Parameterization sphere(g.norm());
  const Eigen::Matrix<double,3,2> basis=sphere.PlusJacobian(g.data());
  Eigen::Matrix<double,3,2> Jg;
  M3 R;
  check(gravity_export::alignment_rotation(g,R)&&(R-align(g)).norm()<1e-13,"smooth production rotation agrees with independent axis-angle at every tilt");
  check(gravity_export::alignment_left_jacobian(g,basis,R,Jg),"finite accepted gravity has an analytic alignment Jacobian");
  const int n=warm?27:15;Mat J=Mat::Zero(n,n+2);J.leftCols(n).setIdentity();
  gravity_export::pose_rows(J,0,m.rotation[0],m.position[0],R,Jg);
  gravity_export::velocity_rows(J,6,m.velocity,R,Jg);
  if(warm){gravity_export::pose_rows(J,15,m.rotation[1],m.position[1],R,Jg);gravity_export::pose_rows(J,21,m.rotation[2],m.position[2],R,Jg);}
  return J;
}
Eigen::VectorXd physical_output_error(const V3&g,const Means&m,bool warm,const Eigen::VectorXd&d) {
  const int n=warm?27:15;GravityS2Parameterization sphere(g.norm());
  V3 gp;const Eigen::Vector2d dg=d.tail<2>();sphere.Plus(g.data(),dg.data(),gp.data());
  const M3 R=align(g),Rp=align(gp);Eigen::VectorXd e=Eigen::VectorXd::Zero(n);
  const int count=warm?3:1;
  for(int i=0;i<count;++i){const int row=i==0?0:15+6*(i-1);
    const M3 qnom=m.rotation[i]*R.transpose(),qtrue=exp_so3(-d.segment<3>(row))*m.rotation[i]*Rp.transpose();
    e.segment<3>(row)=-log_so3(qtrue*qnom.transpose());
    e.segment<3>(row+3)=Rp*(m.position[i]+d.segment<3>(row+3))-R*m.position[i];
  }
  e.segment<3>(6)=Rp*(m.velocity+d.segment<3>(6))-R*m.velocity;
  e.segment<6>(9)=d.segment<6>(9);
  return e;
}
Mat fd_output(const V3&g,const Means&m,bool warm,double eps) {
  const int n=warm?27:15;Mat J(n,n+2);
  for(int col=0;col<n+2;++col){Eigen::VectorXd d=Eigen::VectorXd::Zero(n+2);d(col)=eps;
    J.col(col)=(physical_output_error(g,m,warm,d)-physical_output_error(g,m,warm,-d))/(2*eps);}
  return J;
}
Mat covariance(int n) {
  Mat L=Mat::Identity(n,n);
  for(int r=0;r<n;++r)for(int c=0;c<n;++c)L(r,c)+=.04*std::sin(.3+.47*r+.29*c);
  return .001*L*L.transpose();
}
void output_fd() {
  Means means;
  for(bool warm:{false,true})for(double degree:{0.,1e-6,.01,.099,.1,.101,.11,1.,5.,15.,29.})for(double azimuth:{.2,1.1,2.7}) {
    const double t=degree*M_PI/180.;const V3 g=9.81*V3(std::sin(t)*std::cos(azimuth),std::sin(t)*std::sin(azimuth),std::cos(t));
    const Mat J=output_jacobian(g,means,warm);Mat best;double error=1e100;
    for(double eps:{1e-4,1e-5,1e-6}){const Mat numeric=fd_output(g,means,warm,eps);const double e=(J-numeric).cwiseAbs().maxCoeff();if(e<error){error=e;best=numeric;}}
    max_fd=std::max(max_fd,error);check(error<3e-8,"actual mean re-alignment FD matches full current/warm output Jacobian");
    const Mat P=covariance(J.cols()),actual=J*P*J.transpose(),numeric=best*P*best.transpose();
    const double cov_error=(actual-numeric).cwiseAbs().maxCoeff();max_cov_fd=std::max(max_cov_fd,cov_error);
    check(cov_error<3e-10,"full covariance including gravity cross terms matches independently differenced map");
    Mat old=J.leftCols(J.rows())*P.topLeftCorner(J.rows(),J.rows())*J.leftCols(J.rows()).transpose();
    const double old_error=(actual-old).cwiseAbs().maxCoeff();max_old_error=std::max(max_old_error,old_error);
    check(old_error>1e-6,"deterministic-rotation-only negative control misses gravity uncertainty");
    Eigen::SelfAdjointEigenSolver<Mat> eigen(actual);check(eigen.info()==Eigen::Success&&eigen.eigenvalues().minCoeff()>0.,"mapped full-rank covariance remains SPD");
    // Ordinary deterministic rotations are still the correct zero-gravity-uncertainty limit.
    Mat exact=P;exact.bottomRows(2).setZero();exact.rightCols(2).setZero();
    check((J*exact*J.transpose()-old).norm()<1e-15,"zero gravity covariance/cross terms recover the deterministic frame transform");
  }
  const Mat Jpole=output_jacobian(V3(0,0,9.81),means,false);
  check(Jpole.rightCols(2).norm()>.1,"exact nominal vertical gravity retains its nonzero stochastic output derivative");
  Eigen::Matrix2d scalar;scalar<<1e-6,2e-6,2e-6,1e-4;Eigen::Vector2d slope(1.,-1.);
  check(std::abs((slope.transpose()*scalar*slope)(0)-97e-6)<1e-18,
        "exact commuting tilt example needs gravity variance and attitude/gravity covariance");
}

// A true solver covariance export with the actual S2 parameterization, full
// cross blocks and duplicate newest pose pointers (singular warm output).
class JointFactor:public CostFunction {
public:
  JointFactor(const Mat&P,const V3&g):g_(g){
    set_num_residuals(23);*mutable_parameter_block_sizes()={3,3,3,3,3,3,3,3};
    Eigen::LLT<Mat> llt(P);W_=llt.matrixL().solve(Mat::Identity(23,23));
    GravityS2Parameterization sphere(g.norm());B_=sphere.PlusJacobian(g.data());
  }
  bool Evaluate(double const *const *p,double *res,double **jac)const override {
    Eigen::Matrix<double,23,1> x;
    for(int i=0;i<7;++i)x.segment<3>(3*i)=Eigen::Map<const V3>(p[i]);
    x.tail<2>()=B_.transpose()*(Eigen::Map<const V3>(p[7])-g_);
    Eigen::Map<Eigen::VectorXd>(res,23)=W_*x;
    if(jac)for(int i=0;i<8;++i)if(jac[i]) {
      Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>> J(jac[i],23,3);
      if(i<7)J=W_.middleCols(3*i,3);else J=W_.rightCols(2)*B_.transpose();
    }
    return true;
  }
private:Mat W_;V3 g_;Eigen::Matrix<double,3,2>B_;
};
void solver_export() {
  V3 g=9.81*V3(.1,-.2,1.).normalized();GravityS2Parameterization sphere(9.81);
  std::array<V3,7> variables;for(auto&v:variables)v.setZero();
  const Mat P=covariance(23);JointFactor factor(P,g);Problem problem;
  std::vector<double*>parameters;
  for(auto &v:variables){problem.AddParameterBlock(v.data(),3);parameters.push_back(v.data());}
  problem.AddParameterBlock(g.data(),3,&sphere);parameters.push_back(g.data());problem.AddResidualBlock(&factor,nullptr,parameters);
  SolverOptions options;options.num_threads=1;
  std::vector<double*>request=parameters;
  // Insert an extra newest-clone alias before gravity: the same exact pose as IMU.
  request.insert(request.end()-1,{variables[0].data(),variables[1].data()});
  Mat actual;check(problem.ComputeCovariance(request,actual,options)&&actual.rows()==29,"actual solver exports navigation plus 2D gravity tangent, including repeated pose pointers");
  if(actual.rows()!=29)return;
  Mat A=Mat::Zero(29,23);A.topLeftCorner(21,21).setIdentity();A.block<6,6>(21,0).setIdentity();A.bottomRightCorner<2,2>().setIdentity();
  const Mat expected=A*P*A.transpose();const double e=(actual-expected).cwiseAbs().maxCoeff();max_export=std::max(max_export,e);
  check(e<2e-13,"solver export retains all gravity cross blocks in requested order");
  Means means;means.rotation[2]=means.rotation[0];means.position[2]=means.position[0];
  const Mat J=output_jacobian(g,means,true),out=J*actual*J.transpose();
  Eigen::SelfAdjointEigenSolver<Mat> eig(out);
  check(eig.info()==Eigen::Success&&eig.eigenvalues().minCoeff()>-1e-12&&eig.eigenvalues().head(6).cwiseAbs().maxCoeff()<1e-12,
        "same-time newest clone remains exactly correlated, with PSD singular covariance and no jitter");
  Mat difference=Mat::Zero(6,27);difference.leftCols(6).setIdentity();difference.middleCols(21,6)=-Eigen::Matrix<double,6,6>::Identity();
  check((difference*out*difference.transpose()).norm()<1e-14,"gravity mapping preserves newest IMU/clone equality and shared noise ownership");

  // Fixed-ba insertion must leave appended gravity and every existing cross block aligned.
  Mat omit=Mat::Zero(26,29);omit.topLeftCorner(12,12).setIdentity();omit.bottomRightCorner(14,14).setIdentity();
  const Mat reduced=omit*actual*omit.transpose();Mat restored=Mat::Zero(29,29);
  restored.topLeftCorner(12,12)=reduced.topLeftCorner(12,12);
  restored.bottomRightCorner(14,14)=reduced.bottomRightCorner(14,14);
  restored.block(0,15,12,14)=reduced.block(0,12,12,14);restored.block(15,0,14,12)=reduced.block(12,0,14,12);
  restored.block<3,3>(12,12)=.02*Eigen::Matrix3d::Identity();
  check((restored.rightCols(2)-actual.rightCols(2)).topRows(12).norm()==0.&&
        (restored.bottomRows(14).rightCols(2)-actual.bottomRows(14).rightCols(2)).norm()==0.,
        "legacy frozen-ba insertion preserves gravity tangent position and all non-ba cross terms");
}
void invalid_inputs() {
  const V3 g(0.,0.,9.81);GravityS2Parameterization sphere(9.81);const Eigen::Matrix<double,3,2>B=sphere.PlusJacobian(g.data());
  Eigen::Matrix<double,3,2>J=Eigen::Matrix<double,3,2>::Constant(123.),before=J;
  check(!gravity_export::alignment_left_jacobian(V3::Zero(),B,M3::Identity(),J)&&(J-before).norm()==0.,"zero gravity rejects without changing output");
  V3 bad=g;bad.x()=std::numeric_limits<double>::quiet_NaN();
  check(!gravity_export::alignment_left_jacobian(bad,B,M3::Identity(),J)&&(J-before).norm()==0.,"nonfinite gravity guard survives fast-math");
  check(!gravity_export::alignment_left_jacobian(-g,B,M3::Identity(),J)&&(J-before).norm()==0.,"antipodal alignment rejects the undefined minimal-rotation derivative");
}
} // namespace
int main(){output_fd();solver_export();invalid_inputs();
  std::printf("GRAVITY_EXPORT %s checks=%d failures=%d maxFD=%.17g maxCovFD=%.17g maxSolverExport=%.17g oldOmission=%.17g\n",
      failures?"FAIL":"PASS",checks,failures,max_fd,max_cov_fd,max_export,max_old_error);return failures?1:0;}
