/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "Factor_ConditionalBiasPrior.h"
#include "Factor_GenericPrior.h"
#include "Problem.h"
#include "State_JPLQuatLocal.h"
#include <array>
#include <cstdio>
#include <random>

namespace {
using namespace ov_init;
using namespace ov_init::zbft_sfm;
using namespace ov_init::conditional_bias;
using namespace ov_core;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
int checks=0,failures=0;double max_condition_error=0.,max_fd_error=0.,max_joint_error=0.;
void check(bool pass,const char *label) { ++checks;if(!pass){++failures;std::printf("FAIL: %s\n",label);} }
bool near(const MatrixXd &a,const MatrixXd &b,double tolerance=1e-10) {
  return a.rows()==b.rows() && a.cols()==b.cols() && (a-b).norm()<=tolerance*std::max(1.,b.norm());
}
MatrixXd random_matrix(int rows,int cols) {
  static std::mt19937 generator(8372);static std::normal_distribution<double> sample;
  MatrixXd result(rows,cols);for(int i=0;i<result.size();++i)result.data()[i]=sample(generator);return result;
}

void conditioning(bool singular,double inflation) {
  constexpr int k=7;const int rank=singular?4:k;
  VectorXd camera_scale(k);camera_scale<<.002,.03,.02,.04,.3,1.5,.8;
  MatrixXd root=camera_scale.asDiagonal()*random_matrix(k,rank);
  MatrixXd Pc=root*root.transpose();
  MatrixXd T=.03*random_matrix(6,k)*camera_scale.cwiseInverse().asDiagonal();
  Matrix6 independent=random_matrix(6,6);independent=.0003*(independent*independent.transpose()+Matrix6::Identity());
  Matrix6 Pbb=independent+T*Pc*T.transpose();MatrixXd Pbc=T*Pc;
  Vector6 rw;rw<<1e-5,2e-5,3e-5,2e-4,3e-4,4e-4;
  Vector6 floor;floor<<.02,.03,.02,.15,.18,.12;
  constexpr double gap=.37;
  Conditioned output;
  check(condition(Pbb,Pbc,Pc,gap,rw,inflation,floor,output),"complete correlated bias/camera prior conditions");
  Matrix6 expected_bias=inflation*inflation*Pbb;
  expected_bias.diagonal()+=inflation*inflation*gap*rw;
  Matrix6 expected_conditional=inflation*inflation*independent;
  expected_conditional.diagonal()+=inflation*inflation*gap*rw;
  for(int i=0;i<6;++i) {
    const double added=std::max(0.,floor(i)*floor(i)-expected_bias(i,i));
    expected_bias(i,i)+=added;expected_conditional(i,i)+=added;
  }
  max_condition_error=std::max(max_condition_error,(output.covariance-expected_conditional).cwiseAbs().maxCoeff());
  check(output.calibration_rank==rank && near(output.bias_covariance,expected_bias) &&
        near(output.bias_calibration_covariance,inflation*Pbc),"gap and declared inflation preserve complete Pbb/Pbc/Pcc relationships");
  check(near(output.covariance,expected_conditional) && near(output.regression*root,inflation*T*root),
        "conditional covariance and supported mean sensitivity match independent generative Gaussian model");
  check(near(output.sqrt_information*output.covariance*output.sqrt_information.transpose(),Matrix6::Identity()),
        "conditional whitening is inverse covariance without jitter");
  for(int variant=0;variant<4;++variant) {
    VectorXd units(k);for(int i=0;i<k;++i) units(i)=std::pow(10.,((i+variant)%7)-3.);
    Vector6 bias_units;for(int i=0;i<6;++i)bias_units(i)=variant<2?1.:std::pow(10.,i%3-1.);
    const Matrix6 Bb=bias_units.asDiagonal();const MatrixXd Cc=units.asDiagonal();
    Conditioned scaled;
    check(condition(Bb*Pbb*Bb,Bb*Pbc*Cc,Cc*Pc*Cc,gap,(bias_units.array().square()*rw.array()).matrix(),
                    inflation,(bias_units.array()*floor.array()).matrix(),scaled),"mixed clock/extrinsic/intrinsic and bias unit changes retain valid conditioning");
    check(scaled.calibration_rank==rank && near(scaled.covariance,Bb*output.covariance*Bb) &&
          near(scaled.regression*Cc,Bb*output.regression),"unit rescaling cannot create or discard stochastic calibration support");
  }
  // A prior aged to the newest graph node and then followed by CPI would add
  // this extra independent covariance twice. It must disagree with the oracle.
  Conditioned wrong;
  check(condition(Pbb,Pbc,Pc,gap+2.,rw,inflation,floor,wrong) && !near(wrong.covariance,output.covariance,1e-7),
        "aging to the latest node rather than first future node fails the gap covariance oracle");
  check(condition(Pbb,MatrixXd::Zero(6,k),Pc,gap,rw,inflation,floor,wrong) &&
        !near(wrong.covariance,output.covariance,1e-7),"discarding bias/calibration cross covariance fails the conditioning oracle");
}

void invalid_conditioning() {
  Matrix6 Pbb=Matrix6::Identity();MatrixXd Pc=MatrixXd::Ones(2,2),Pbc=MatrixXd::Zero(6,2);
  Conditioned sentinel;sentinel.regression=MatrixXd::Constant(3,3,17.);sentinel.calibration_rank=77;
  for(int variant=0;variant<8;++variant) {
    auto bb=Pbb;auto bc=Pbc;auto cc=Pc;double gap=.1;
    if(variant==0)bc(0,0)=1e-8; // full joint's negative eigenvalue is below roundoff, but its range error is not
    if(variant==1) { bc(0,0)=1e-8;VectorXd scale(2);scale<<1e-6,1e6;bc=bc*scale.asDiagonal();cc=scale.asDiagonal()*cc*scale.asDiagonal(); }
    if(variant==2)bb(0,0)=-1.;
    if(variant==3)cc(0,1)+=.1;
    if(variant==4)bb(0,0)=std::numeric_limits<double>::quiet_NaN();
    if(variant==5)bc.resize(5,2);
    if(variant==6)gap=-.1;
    if(variant==7)cc(0,0)=0.;
    if(variant<2)check(!valid_joint(bb,bc,cc),"capture rejects range defects before joint reset selection, invariant to calibration units");
    auto output=sentinel;
    check(!condition(bb,bc,cc,gap,Vector6::Constant(.01),1.,Vector6::Zero(),output) &&
          output.calibration_rank==77 && near(output.regression,sentinel.regression,0.),"invalid joint/range/timing rejects atomically under fast math");
  }
  auto output=sentinel;
  check(!condition(Pbb,Matrix6::Identity(),Matrix6::Identity(),0.,Vector6::Zero(),1.,Vector6::Zero(),output),
        "deterministic conditional biases are not replaced by fabricated jitter");
  check(condition(Pbb,Matrix6::Identity(),Matrix6::Identity(),.1,Vector6::Constant(.01),1.,Vector6::Zero(),output) &&
        near(output.covariance,.001*Matrix6::Identity()),"genuine future bias random walk supplies its actual conditional covariance");
}

void factor_and_solver(double perturbation) {
  std::vector<InitCameraCalibrationBlock> calibration(3);
  calibration[0].kind=InitCameraCalibrationKind::Clock;calibration[0].mean=VectorXd::Constant(1,.125);
  calibration[1].kind=InitCameraCalibrationKind::Extrinsics;calibration[1].mean.resize(7);
  calibration[1].mean<<rot_2_quat(exp_so3(Eigen::Vector3d(.13,-.21,.17))),Eigen::Vector3d(.1,-.2,.3);
  calibration[2].kind=InitCameraCalibrationKind::Intrinsics;calibration[2].mean.resize(8);
  calibration[2].mean<<450.,455.,320.,240.,.03,-.004,.001,-.0002;
  for(auto &block:calibration)block.fej=block.mean;
  constexpr int k=15;
  const Vector6 mean=Vector6::LinSpaced(.005,.03);
  Conditioned prior;prior.regression=.01*random_matrix(6,k);
  Matrix6 root=.04*random_matrix(6,6);prior.covariance=root*root.transpose()+.003*Matrix6::Identity();
  prior.sqrt_information=prior.covariance.llt().matrixL().solve(Matrix6::Identity());
  Factor_ConditionalBiasPrior factor(mean,prior,calibration);
  std::array<VectorXd,6> values;
  values[0]=mean.head<3>()+Eigen::Vector3d(.01,-.02,.03);values[1]=mean.tail<3>()-Eigen::Vector3d(.01,-.02,.03);
  values[2]=calibration[0].mean;values[3]=calibration[1].mean.head<4>();values[4]=calibration[1].mean.tail<3>();values[5]=calibration[2].mean;
  State_JPLQuatLocal quat;
  VectorXd dc=perturbation*VectorXd::LinSpaced(k,-.2,.3);
  values[2](0)+=dc(0);values[4]+=dc.segment<3>(4);values[5]+=dc.tail<8>();
  const Eigen::Vector4d q0=values[3];quat.Plus(q0.data(),dc.segment<3>(1).data(),values[3].data());
  std::array<const double *,6> p;std::array<double *,6> jac;std::array<RowMatrix,6> storage;
  for(int i=0;i<6;++i){p[i]=values[i].data();storage[i].resize(6,values[i].size());jac[i]=storage[i].data();}
  Vector6 r,b;b<<values[0],values[1];
  check(factor.Evaluate(p.data(),r.data(),jac.data()) && near(r,prior.sqrt_information*(b-mean-prior.regression*dc)),
        "real conditional bias factor uses the retained calibration error chart and mean sensitivity");
  for(int block=0;block<6;++block) {
    const auto nominal=values[block];const int local=block==3?3:nominal.size();MatrixXd differences(6,local);
    for(int column=0;column<local;++column) {
      const double h=block==5?1e-5:1e-7;Vector6 plus,minus;
      for(int sign:{1,-1}) {
        values[block]=nominal;
        if(block==3){Eigen::Vector3d step=Eigen::Vector3d::Zero();step(column)=sign*h;quat.Plus(nominal.data(),step.data(),values[block].data());}
        else values[block](column)+=sign*h;
        check(factor.Evaluate(p.data(),sign>0?plus.data():minus.data(),nullptr),"conditional prior manifold perturbation evaluates");
      }
      differences.col(column)=(plus-minus)/(2.*h);
    }
    values[block]=nominal;
    max_fd_error=std::max(max_fd_error,(differences-storage[block].leftCols(local)).cwiseAbs().maxCoeff());
    check(near(differences,storage[block].leftCols(local),1e-7),"conditional bias prior calibration/bias derivatives agree with independent differences");
  }
  if(perturbation!=0.)return;
  Problem problem;
  for(int i=0;i<6;++i) {problem.AddParameterBlock(values[i].data(),values[i].size(),i==3?&quat:nullptr);if(i>=2)problem.SetParameterBlockConstant(values[i].data());}
  std::vector<double *> blocks;for(auto &value:values)blocks.push_back(value.data());
  problem.AddResidualBlock(&factor,nullptr,blocks);
  Matrix6 L=random_matrix(6,6);const Matrix6 R=.01*(L*L.transpose()+Matrix6::Identity());
  const Matrix6 information=R.llt().solve(Matrix6::Identity());
  const Vector6 observation=mean+Vector6::LinSpaced(-.01,.02);
  Factor_GenericPrior measurement(observation,{"vec3","vec3"},information,Vector6::Zero());
  problem.AddResidualBlock(&measurement,nullptr,{values[0].data(),values[1].data()});
  SolverOptions options;options.num_threads=1;options.max_solver_time_seconds=5.;
  options.gradient_tolerance=1e-12;options.parameter_tolerance=1e-12;options.function_tolerance=1e-12;
  const auto solve=problem.Solve(options);MatrixXd Q,S;
  check(!solve.time_stopped && solve.converged && problem.ComputeConditionalCovariance({values[0].data(),values[1].data()},
        {values[2].data(),values[3].data(),values[4].data(),values[5].data()},Q,S,options),"real solver includes conditional bias prior camera columns");
  const Matrix6 K=(prior.covariance+R).ldlt().solve(prior.covariance).transpose();
  const Matrix6 M=Matrix6::Identity()-K;
  const Matrix6 expected_Q=M*prior.covariance*M.transpose()+K*R*K.transpose();
  Vector6 estimate;estimate<<values[0],values[1];
  max_joint_error=std::max(max_joint_error,(Q-expected_Q).cwiseAbs().maxCoeff());
  check(near(estimate,mean+K*(observation-mean)) && near(Q,expected_Q) && near(S,M*prior.regression),
        "actual conditional factor solve/export matches independent Gaussian update mean, Joseph Q and S");
  check(!near(S,MatrixXd::Zero(6,k),1e-7),"omitting minus-WT calibration columns fails the actual solver oracle");
}
} // namespace

int main() {
  for(bool singular:{false,true})for(double inflation:{1.,2.})conditioning(singular,inflation);
  invalid_conditioning();factor_and_solver(0.);factor_and_solver(.1);
  std::printf("CONDITIONAL_BIAS_PRIOR %d/%d passed; conditioning %.3e factor FD %.3e solver %.3e\n",
              checks-failures,checks,max_condition_error,max_fd_error,max_joint_error);
  return failures?1:0;
}
