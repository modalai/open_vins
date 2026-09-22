/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_INIT_FACTOR_CONDITIONAL_BIAS_PRIOR_H
#define OV_INIT_FACTOR_CONDITIONAL_BIAS_PRIOR_H

#include "CostFunction.h"
#include "init/ConditionalBiasPrior.h"
#include "utils/InitializerPhysicalWarmResult.h"
#include "utils/quat_ops.h"

namespace ov_init {
namespace zbft_sfm {

// r=W*(db-T*dc). Calibration is held constant during the conditional fit;
// -W*T is required during export, even though dc=0 at every optimizer iterate.
// Parameters: bg3, ba3, then semantic calibration blocks (extrinsics split
// into quaternion4 and position3), in the same order used by the Pc snapshot.
class Factor_ConditionalBiasPrior final : public CostFunction {
public:
  Factor_ConditionalBiasPrior(const conditional_bias::Vector6 &mean,const conditional_bias::Conditioned &prior,
                              const std::vector<ov_core::InitCameraCalibrationBlock> &calibration)
      : mean_(mean),W_(prior.sqrt_information),calibration_(calibration) {
    set_num_residuals(6);*mutable_parameter_block_sizes()={3,3};
    int dimension=0;
    for(const auto &block:calibration_) {
      dimension+=block.local_size();
      if(block.kind==ov_core::InitCameraCalibrationKind::Extrinsics) {
        mutable_parameter_block_sizes()->push_back(4);mutable_parameter_block_sizes()->push_back(3);
      } else mutable_parameter_block_sizes()->push_back(block.value_size());
    }
    valid_=prior.regression.rows()==6 && prior.regression.cols()==dimension &&
        ov_core::numeric::finite_matrix(mean_) && ov_core::numeric::finite_matrix(W_) &&
        ov_core::numeric::finite_matrix(prior.regression);
    if(valid_) H_=-W_*prior.regression;
  }

  bool Evaluate(double const *const *p,double *residual,double **jac) const override {
    if(!valid_) return false;
    using ov_core::numeric::finite_matrix;
    for(size_t i=0;i<parameter_block_sizes().size();++i)
      if(!finite_matrix(Eigen::Map<const Eigen::VectorXd>(p[i],parameter_block_sizes()[i]))) return false;
    conditional_bias::Vector6 delta;
    delta<<Eigen::Map<const Eigen::Vector3d>(p[0]),Eigen::Map<const Eigen::Vector3d>(p[1]);
    Eigen::Map<conditional_bias::Vector6> r(residual);r=W_*(delta-mean_);
    if(jac && jac[0]) Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>>{jac[0]}=W_.leftCols<3>();
    if(jac && jac[1]) Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>>{jac[1]}=W_.rightCols<3>();
    int parameter=2,column=0;
    for(const auto &block:calibration_) {
      const int n=block.local_size();
      if(!n || block.mean.size()!=block.value_size() || !finite_matrix(block.mean)) return false;
      if(block.kind==ov_core::InitCameraCalibrationKind::Extrinsics) {
        const Eigen::Matrix3d R=ov_core::quat_2_Rot(Eigen::Map<const Eigen::Vector4d>(p[parameter]));
        const Eigen::Matrix3d R0=ov_core::quat_2_Rot(block.mean.head<4>());
        const Eigen::Vector3d angle=-ov_core::log_so3(R*R0.transpose());
        r.noalias()+=H_.middleCols(column,3)*angle;
        r.noalias()+=H_.middleCols(column+3,3)*(Eigen::Map<const Eigen::Vector3d>(p[parameter+1])-block.mean.tail<3>());
        if(jac && jac[parameter]) {
          Eigen::Map<Eigen::Matrix<double,6,4,Eigen::RowMajor>> out(jac[parameter]);out.setZero();
          out.leftCols<3>()=H_.middleCols(column,3)*ov_core::Jl_so3(-angle).inverse();
          if(!finite_matrix(out)) return false;
        }
        if(jac && jac[parameter+1]) Eigen::Map<Eigen::Matrix<double,6,3,Eigen::RowMajor>>{jac[parameter+1]}=H_.middleCols(column+3,3);
        parameter+=2;
      } else {
        r.noalias()+=H_.middleCols(column,n)*(Eigen::Map<const Eigen::VectorXd>(p[parameter],n)-block.mean);
        if(jac && jac[parameter]) Eigen::Map<Eigen::Matrix<double,6,Eigen::Dynamic,Eigen::RowMajor>>{jac[parameter],6,n}=H_.middleCols(column,n);
        ++parameter;
      }
      column+=n;
    }
    return finite_matrix(r);
  }

private:
  conditional_bias::Vector6 mean_;
  conditional_bias::Matrix6 W_;
  Eigen::MatrixXd H_;
  std::vector<ov_core::InitCameraCalibrationBlock> calibration_;
  bool valid_=false;
};

} // namespace zbft_sfm
} // namespace ov_init
#endif
