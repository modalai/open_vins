/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "Factor_ImageReprojPhysical.h"
#include "Factor_GenericPrior.h"
#include "State_JPLQuatLocal.h"
#include "Problem.h"
#include "dynamic/ConditionalPhysicalWarm.h"
#include <Eigen/QR>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

namespace {
using namespace ov_init;
using namespace ov_init::zbft_sfm;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
int checks=0, failures=0;
double max_factor_error=0., max_joint_error=0.,max_resolve_error=0.;
void check(bool ok,const char *label) { ++checks; if (!ok) { ++failures; std::printf("FAIL: %s\n",label); } }
bool near(const MatrixXd &a,const MatrixXd &b,double tolerance=1e-9) {
  return a.rows()==b.rows() && a.cols()==b.cols() && (a-b).norm() <= tolerance*std::max(1.,b.norm());
}

void clock_factor(bool fisheye,double shift) {
  std::array<VectorXd,10> values;
  values[0]=ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(.13,-.21,.17)));
  values[1]=Eigen::Vector3d(.2,-.1,.3); values[2]=Eigen::Vector3d(1.,.5,4.);
  values[3]=ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(-.04,.08,.11)));
  values[4]=Eigen::Vector3d(.11,-.03,.02);
  values[5].resize(8); values[5]<<450.,455.,320.,240.,.03,-.004,.001,-.0002;
  values[6]=VectorXd::Constant(1,.125+shift);
  values[7]=Eigen::Vector3d(.4,-.3,.2); values[8]=Eigen::Vector3d(.008,-.005,.003);
  values[9]=Eigen::Vector3d(.01,-.02,.03);
  Eigen::Matrix3d A,G,Tg;
  A<<1.01,.003,-.002, .001,.995,.004, -.003,.002,1.006;
  G<<.998,.002,.004, -.001,1.008,-.003, .002,.004,.991;
  Tg<<.002,-.001,.003, .004,.002,-.002, -.003,.001,.002;
  const Eigen::Vector3d wm(.3,-.2,.15),am(.6,-.4,9.7);
  Factor_ImageReprojPhysical factor({422.,288.},1.3,fisheye,.125,wm,am,A,G,Tg);
  std::array<const double *,10> parameters;
  std::array<RowMatrix,10> jacobians;
  std::array<double *,10> pointers;
  for (int i=0;i<10;++i) {
    parameters[i]=values[i].data(); jacobians[i].resize(2,values[i].size()); pointers[i]=jacobians[i].data();
  }
  Eigen::Vector2d residual;
  check(factor.Evaluate(parameters.data(),residual.data(),pointers.data()),"clock factor evaluates the fixed nominal graph chart");
  State_JPLQuatLocal quaternion;
  for (int block=0;block<10;++block) {
    const int local=(block==0 || block==3) ? 3 : values[block].size();
    const VectorXd nominal=values[block];
    MatrixXd numeric(2,local);
    for (int j=0;j<local;++j) {
      const double h=block==5 ? 1e-5 : 1e-7;
      Eigen::Vector2d plus,minus;
      for (int sign : {1,-1}) {
        if (block==0 || block==3) {
          Eigen::Vector3d delta=Eigen::Vector3d::Zero(); delta(j)=sign*h;
          quaternion.Plus(nominal.data(),delta.data(),values[block].data());
        } else { values[block]=nominal; values[block](j)+=sign*h; }
        check(factor.Evaluate(parameters.data(),sign>0?plus.data():minus.data(),nullptr),"perturbed exposure evaluates without moving graph membership");
      }
      numeric.col(j)=(plus-minus)/(2.*h);
    }
    values[block]=nominal;
    max_factor_error=std::max(max_factor_error,(numeric-jacobians[block].leftCols(local)).cwiseAbs().maxCoeff());
    check(near(jacobians[block].leftCols(local),numeric,3e-6),"analytic clock/exposure derivatives agree with independent manifold differences");
  }
  check(jacobians[6].norm()>1. && (jacobians[6]*Eigen::VectorXd::Ones(1)).norm()>1.,
        "an absolute/common camera clock shift has a nonzero visual column");
  if (shift==0.) {
    Factor_ImageReprojCalib fixed({422.,288.},1.3,fisheye);
    std::array<RowMatrix,6> reference; std::array<double *,6> refs;
    for (int i=0;i<6;++i) { reference[i].resize(2,values[i].size()); refs[i]=reference[i].data(); }
    Eigen::Vector2d r;
    check(fixed.Evaluate(parameters.data(),r.data(),refs.data()) && near(r,residual,1e-14),"nominal clock factor preserves the old reprojection residual");
    for (int i=0;i<6;++i) check(near(reference[i],jacobians[i],1e-14),"nominal navigation/calibration image derivatives remain unchanged");
    check(jacobians[7].isZero(0.) && jacobians[8].isZero(0.) && jacobians[9].isZero(0.),
          "velocity and raw biases do not add fictitious image information at zero clock delta");
  }
  values[6](0)=std::numeric_limits<double>::quiet_NaN();
  check(!factor.Evaluate(parameters.data(),residual.data(),pointers.data()),"nonfinite clock rejects under fast math");
}

void nonlinear_resolves() {
  // Exact observations make the nominal residual zero, so the frozen GN
  // sensitivity is also the derivative of this nonlinear optimum. Perturb
  // absolute clocks while keeping graph nodes, IMU chart and feature rows fixed.
  Eigen::Vector4d q=ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(.12,-.08,.05)));
  Eigen::Vector3d p(.1,-.2,.3),v(.4,-.3,.2),bg(.008,-.005,.003),ba(.01,-.02,.03);
  const Eigen::Vector4d q0=q;
  const Eigen::Vector3d p0=p;
  const Eigen::Matrix3d R0=ov_core::quat_2_Rot(q0);
  std::array<double,2> clocks{{.125,-.25}};
  const auto c0=clocks;
  std::array<Eigen::Vector4d,2> extr_q{{ov_core::rot_2_quat(Eigen::Matrix3d::Identity()),
      ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(.02,-.03,.01)))}};
  std::array<Eigen::Vector3d,2> extr_p{{Eigen::Vector3d::Zero(),Eigen::Vector3d(.15,.02,-.01)}};
  Eigen::Matrix<double,8,1> intr; intr<<450.,455.,320.,240.,.03,-.004,.001,-.0002;
  const Eigen::Vector3d wm(.3,-.2,.15),am(.6,-.4,9.7);
  const Eigen::Matrix3d identity=Eigen::Matrix3d::Identity(),zero=Eigen::Matrix3d::Zero();
  std::array<Eigen::Vector3d,20> points;
  State_JPLQuatLocal quaternion;
  Problem problem;
  problem.AddParameterBlock(q.data(),4,&quaternion); problem.AddParameterBlock(p.data(),3);
  for(double *fixed:{v.data(),bg.data(),ba.data()}) { problem.AddParameterBlock(fixed,3);problem.SetParameterBlockConstant(fixed); }
  problem.AddParameterBlock(intr.data(),8);problem.SetParameterBlockConstant(intr.data());
  for(int camera=0;camera<2;++camera) {
    problem.AddParameterBlock(extr_q[camera].data(),4,&quaternion);problem.SetParameterBlockConstant(extr_q[camera].data());
    problem.AddParameterBlock(extr_p[camera].data(),3);problem.SetParameterBlockConstant(extr_p[camera].data());
    problem.AddParameterBlock(&clocks[camera],1);problem.SetParameterBlockConstant(&clocks[camera]);
  }
  std::vector<std::unique_ptr<Factor_ImageReprojPhysical>> factors;
  for(size_t i=0;i<points.size();++i) {
    points[i]<<-.8+.4*(i%5),-.5+.35*(i/5),3.5+.1*i;
    problem.AddParameterBlock(points[i].data(),3);problem.SetParameterBlockConstant(points[i].data());
    for(int camera=0;camera<2;++camera) {
      std::vector<double *> blocks{q.data(),p.data(),points[i].data(),extr_q[camera].data(),extr_p[camera].data(),
                                  intr.data(),&clocks[camera],v.data(),bg.data(),ba.data()};
      const double *reference[]{blocks[0],blocks[1],blocks[2],blocks[3],blocks[4],blocks[5]};
      Factor_ImageReprojCalib prediction(Eigen::Vector2d::Zero(),1.,false);
      Eigen::Vector2d uv;
      check(prediction.Evaluate(reference,uv.data(),nullptr),"independent base factor creates nominal visible observations");
      factors.emplace_back(new Factor_ImageReprojPhysical(uv,1.,false,c0[camera],wm,am,identity,identity,zero));
      problem.AddResidualBlock(factors.back().get(),nullptr,blocks);
    }
  }
  VectorXd anchor(7);anchor<<q0,p0;
  Factor_GenericPrior prior(anchor,{"quat","vec3"},20.*MatrixXd::Identity(6,6),VectorXd::Zero(6));
  problem.AddResidualBlock(&prior,nullptr,{q.data(),p.data()});
  SolverOptions options;options.num_threads=1;options.max_num_iterations=60;options.max_solver_time_seconds=10.;
  options.gradient_tolerance=1e-12;options.parameter_tolerance=1e-13;options.function_tolerance=1e-13;
  const auto baseline=problem.Solve(options);
  MatrixXd Q,S;
  check(!baseline.time_stopped && baseline.final_cost<1e-18 &&
        problem.ComputeConditionalCovariance({q.data(),p.data()},{&clocks[0],&clocks[1]},Q,S,options),
        "actual fixed-graph nonlinear problem exports absolute-clock conditional sensitivity");
  if(S.rows()!=6 || S.cols()!=2) return;
  MatrixXd differences(6,2);
  constexpr double h=2e-6;
  for(int camera=0;camera<2;++camera) {
    std::array<VectorXd,2> delta;
    for(int side=0;side<2;++side) {
      q=q0;p=p0;clocks=c0;clocks[camera]+=(side==0?1.:-1.)*h;
      const auto solved=problem.Solve(options);
      check(!solved.time_stopped && solved.final_cost<solved.initial_cost && solved.converged,
            "perturbed absolute clock re-solves the unchanged nonlinear observation graph");
      delta[side].resize(6);
      delta[side]<<-ov_core::log_so3(ov_core::quat_2_Rot(q)*R0.transpose()),p-p0;
    }
    differences.col(camera)=(delta[0]-delta[1])/(2.*h);
  }
  max_resolve_error=(differences-S).cwiseAbs().maxCoeff();
  check(near(differences,S,5e-5),"fixed-graph nonlinear re-solves agree with exported conditional-fit sensitivity");
  check((S.col(0)+S.col(1)).norm()>.1 && (differences.col(0)+differences.col(1)).norm()>.1,
        "actual nonlinear re-solves preserve the common absolute-clock direction relative to fixed IMU time");
}

class LinearFactor final : public CostFunction {
public:
  std::vector<MatrixXd> A;
  explicit LinearFactor(std::vector<MatrixXd> matrices) : A(std::move(matrices)) {
    set_num_residuals(A.front().rows());
    for (const auto &a : A) mutable_parameter_block_sizes()->push_back(a.cols());
  }
  bool Evaluate(double const *const *p,double *r,double **jac) const override {
    Eigen::Map<VectorXd> out(r,num_residuals()); out.setZero();
    for (size_t i=0;i<A.size();++i) {
      out.noalias()+=A[i]*Eigen::Map<const VectorXd>(p[i],A[i].cols());
      if (jac && jac[i]) Eigen::Map<RowMatrix>(jac[i],A[i].rows(),A[i].cols())=A[i];
    }
    return true;
  }
};

void dense_conditional(bool singular) {
  constexpr int n=23,k=5,m=89;
  std::mt19937 random(7251);
  std::normal_distribution<double> normal;
  auto matrix=[&](int rows,int cols) { MatrixXd x(rows,cols); for(int i=0;i<x.size();++i)x.data()[i]=normal(random); return x; };
  MatrixXd A=MatrixXd::Zero(m,29),B=matrix(m,k);
  A.block(0,0,31,n)=matrix(31,n); A.block(0,n,31,3)=matrix(31,3);
  A.block(31,0,35,n)=matrix(35,n); A.block(31,n+3,35,3)=matrix(35,3);
  A.bottomRows(n).leftCols(n).setIdentity();
  // Independent correlated state/calibration prior in conditional form. This
  // exercises the algebra; it does not enable the unsupported ResetBiasPrior API.
  const MatrixXd T0=.2*matrix(n,k);
  B.bottomRows(n)=-T0;
  std::array<double,15> imu{}; std::array<double,6> old{}; std::array<double,2> gravity{},clocks{};
  std::array<double,3> f{},g{},other{};
  Problem problem;
  for (const auto &block : std::vector<std::pair<double *,int>>{{f.data(),3},{clocks.data(),2},{old.data(),6},
       {gravity.data(),2},{other.data(),3},{g.data(),3},{imu.data(),15}}) problem.AddParameterBlock(block.first,block.second);
  problem.SetSchurLandmark(f.data()); problem.SetSchurLandmark(g.data());
  problem.SetParameterBlockConstant(clocks.data()); problem.SetParameterBlockConstant(other.data());
  LinearFactor first({A.block(0,0,31,15),A.block(0,15,31,6),A.block(0,21,31,2),A.block(0,23,31,3),B.block(0,0,31,2),B.block(0,2,31,3)});
  LinearFactor second({A.block(31,0,35,15),A.block(31,15,35,6),A.block(31,21,35,2),A.block(31,26,35,3),B.block(31,0,35,2),B.block(31,2,35,3)});
  LinearFactor prior({A.block(66,0,n,15),A.block(66,15,n,6),A.block(66,21,n,2),B.block(66,0,n,2),B.block(66,2,n,3)});
  problem.AddResidualBlock(&first,nullptr,{imu.data(),old.data(),gravity.data(),f.data(),clocks.data(),other.data()});
  problem.AddResidualBlock(&second,nullptr,{imu.data(),old.data(),gravity.data(),g.data(),clocks.data(),other.data()});
  problem.AddResidualBlock(&prior,nullptr,{imu.data(),old.data(),gravity.data(),clocks.data(),other.data()});
  MatrixXd Q,S; SolverOptions options;
  check(problem.ComputeConditionalCovariance({old.data(),gravity.data(),imu.data()},{other.data(),clocks.data()},Q,S,options),
        "unique solved blocks export with permuted calibration columns");
  const MatrixXd gain=A.completeOrthogonalDecomposition().solve(MatrixXd::Identity(m,m));
  MatrixXd reorderedB(m,k); reorderedB<<B.rightCols(3),B.leftCols(2);
  const MatrixXd denseS=-gain*reorderedB;
  // Canonical output: IMU15, two different-camera owners of old pose6, and a
  // newest owner that shares the current IMU graph node. Gravity remains in the
  // unique solved block order until this output map, including nonzero crosses.
  MatrixXd C=MatrixXd::Zero(33,n),C_dense=MatrixXd::Zero(33,29);
  C.block(0,8,15,15).setIdentity(); C.block(15,0,6,6).setIdentity();
  C.block(21,0,6,6).setIdentity(); C.block(27,8,6,6).setIdentity();
  MatrixXd Jg=.1*matrix(15,2),Jold=.1*matrix(6,2);
  C.block(0,6,15,2)=Jg; C.block(15,6,6,2)=Jold; C.block(21,6,6,2)=Jold; C.block(27,6,6,2)=Jg.topRows(6);
  C_dense.leftCols(15)=C.rightCols(15); C_dense.middleCols(15,6)=C.leftCols(6); C_dense.middleCols(21,2)=C.middleCols(6,2);
  MatrixXd D=MatrixXd::Zero(33,k);
  Eigen::Matrix<double,6,1> rate; rate<<.4,-.2,.3,.7,-.3,.2;
  D.block<6,1>(15,3)=rate; D.block<6,1>(21,4)=rate; D.block<6,1>(27,3)=1.2*rate;
  MatrixXd root=.025*matrix(k,k);
  if (singular) root.row(4)=root.row(3);
  const MatrixXd Pc=root*root.transpose();
  VectorXd inflation=VectorXd::Ones(33); inflation.head(15).setConstant(1.7);
  inflation.segment<3>(0).setConstant(1.2); inflation.segment<3>(3).setConstant(1.4);
  for(int i=15;i<33;i+=6) { inflation.segment<3>(i).setConstant(1.2); inflation.segment<3>(i+3).setConstant(1.4); }
  MatrixXd actual;
  check(conditional_warm::assemble(Q,S,C,D,Pc,inflation,actual),"conditional covariance assembles with gravity, repeated owners, and retained prior");
  // Dense Joseph-style estimator error map from independent measurement noise
  // and the original calibration random vector. No Schur blocks or assembled
  // Q/S are reused in this oracle.
  MatrixXd noise=MatrixXd::Zero(m+k,m+k); noise.topLeftCorner(m,m).setIdentity(); noise.bottomRightCorner(k,k)=Pc;
  MatrixXd error_map=MatrixXd::Zero(33+k,m+k);
  error_map.topLeftCorner(33,m)=inflation.asDiagonal()*C_dense*gain;
  error_map.topRightCorner(33,k)=inflation.asDiagonal()*(C_dense*denseS+D);
  error_map.bottomRightCorner(k,k).setIdentity();
  const MatrixXd expected=error_map*noise*error_map.transpose();
  max_joint_error=std::max(max_joint_error,(actual-expected).cwiseAbs().maxCoeff());
  check(near(actual,expected,5e-10),"full joint equals independent dense conditional Gaussian/Joseph covariance");
  check(near(actual.bottomRightCorner(k,k),Pc,0.),"complete calibration prior is preserved, including off-diagonal and singular support");
  MatrixXd differ=actual.block(15,15,6,6)+actual.block(21,21,6,6)-actual.block(15,21,6,6)-actual.block(21,15,6,6);
  const MatrixXd direct=inflation.segment<6>(15).asDiagonal()*(D.middleRows(15,6)-D.middleRows(21,6));
  check(near(differ,direct*Pc*direct.transpose(),2e-10),"same-node owner differential covariance has exactly the separate-clock prior support");
  check(singular ? differ.norm()<1e-10 : differ.norm()>1e-5,"perfectly correlated and independent clocks have different coincidence constraints");
  auto negative=[&](const MatrixXd &badS,const MatrixXd &badD,const MatrixXd &badPc,const char *label) {
    MatrixXd wrong; check(conditional_warm::assemble(Q,badS,C,badD,badPc,inflation,wrong) && !near(wrong,expected,1e-6),label);
  };
  negative(MatrixXd::Zero(n,k),D,Pc,"omitting the conditional-fit sensitivity fails the oracle");
  negative(S,MatrixXd::Zero(33,k),Pc,"omitting the direct exposure clock derivative fails the oracle");
  negative(S,2.*D,Pc,"adding owner clock uncertainty twice fails the oracle");
  negative(S,D,Pc.diagonal().asDiagonal(),"discarding prior off-diagonal covariance fails the oracle");
  MatrixXd no_gravity=C; no_gravity.middleCols(6,2).setZero(); MatrixXd wrong;
  check(conditional_warm::assemble(Q,S,no_gravity,D,Pc,inflation,wrong) && !near(wrong,expected,1e-6),"dropping gravity/output cross terms fails the oracle");
  const MatrixXd H=A.transpose()*A;
  const MatrixXd Hred=H.topLeftCorner(n,n)-H.topRightCorner(n,6)*H.bottomRightCorner(6,6).ldlt().solve(H.bottomLeftCorner(6,n));
  const MatrixXd wrong_s=-Hred.ldlt().solve(A.leftCols(n).transpose()*reorderedB); // missing landmark/calibration Schur term
  MatrixXd requested_wrong(n,k); requested_wrong<<wrong_s.middleRows(15,6),wrong_s.bottomRows(2),wrong_s.topRows(15);
  negative(requested_wrong,D,Pc,"omitting landmark-calibration Schur terms fails the oracle");
  MatrixXd relative=S,relative_direct=D;
  relative.col(3)=-relative.col(4);relative_direct.col(3)=-relative_direct.col(4);
  negative(relative,relative_direct,Pc,"replacing absolute clock columns by relative-only clocks fails the oracle");
  MatrixXd without_prior=reorderedB; without_prior.bottomRows(n).setZero();
  MatrixXd ignored=-gain*without_prior;
  requested_wrong<<ignored.middleRows(15,6),ignored.middleRows(21,2),ignored.topRows(15);
  negative(requested_wrong,D,Pc,"discarding state-calibration conditional prior columns fails the oracle");
  MatrixXd sentinel=MatrixXd::Constant(2,2,17.),out=sentinel,bad=Pc;
  bad(0,0)=-1.;
  check(!conditional_warm::assemble(Q,S,C,D,bad,inflation,out) && near(out,sentinel,0.),"invalid retained prior rejects without a partial output or jitter");
  bad=S; bad(0,0)=std::numeric_limits<double>::quiet_NaN();
  check(!conditional_warm::assemble(Q,bad,C,D,Pc,inflation,out) && near(out,sentinel,0.),"nonfinite sensitivity rejects atomically under fast math");
}
} // namespace

int main() {
  for(bool fisheye : {false,true}) for(double shift : {0.,.003,-.002}) clock_factor(fisheye,shift);
  nonlinear_resolves();
  for(bool singular : {false,true}) dense_conditional(singular);
  std::printf("PHYSICAL_CONSIDER %d/%d passed; factor FD %.3e dense joint %.3e nonlinear sensitivity %.3e\n",
              checks-failures,checks,max_factor_error,max_joint_error,max_resolve_error);
  return failures?1:0;
}
