#ifndef OV_ZCALIB_BIAS_LINKED_CALIB_H
#define OV_ZCALIB_BIAS_LINKED_CALIB_H

#include "JointCalib.h"

namespace ov_zcalib {

/// Diagnostics for the experimental connected-bias objective. These are not
/// acceptance decisions. A caller must run independent recovery/holdout gates.
struct BiasLinkedReport {
  int nodes = 0;
  int gap_links = 0;
  std::vector<WindowBoundaryBias> boundary; // original window order
  std::vector<double> accepted_merit;
  Eigen::MatrixXd joint_information; // [calibration, raw-frame bias nodes]
  Eigen::VectorXd variable_scales;
};

class BiasLinkedCalib {
public:
  /// Experimental standalone solve; not selected by any production profile.
  /// Windows must belong to one continuous physical IMU/clock segment, with
  /// strictly disjoint clone spans and unique visual measurements. Clock resets
  /// or sensor changes require separate calls. Touching spans are refused too:
  /// aliasing a bias alone would not remove duplicate boundary image factors.
  ///
  /// The initial physical bias prior is applied ONCE at the earliest endpoint;
  /// it must not be learned from the other half in a consistency experiment.
  /// Raw-frame Brownian bias transitions bridge the positive gaps. Interior
  /// dynamics already contain their own bias evolution, so those intervals are
  /// never added a second time. Poses/gravity remain independent nuisances;
  /// this is a connected-bias model, not a complete trajectory smoother.
  /// IMU covariance propagation and whitening are fixed at noise_lin for the
  /// entire solve, independently of the explicit physical bias-prior mean.
  /// The mean factors retain WindowBA's first-order CPI bias/calibration
  /// corrections; covariance is a local Gauss-Newton approximation.
  ///
  /// rep.Lambda/sigma marginalize ALL boundary biases. Local conditional
  /// curvature with the fitted biases held fixed is never reported as Tg
  /// precision. Failure restores the input calibration. No window is silently
  /// dropped and no partial evaluation can become an accepted posterior.
  /// A successful return certifies a finite positive-definite posterior at a
  /// complete accepted iterate, not convergence of the nonlinear problem.
  /// Iteration/backtrack caps can leave nonstationary nuisances; inspect
  /// rep.qn_max_final and require independent convergence/recovery gates.
  /// These numerical checks do not certify production calibration accuracy.
  static bool solve(const std::vector<WindowData> &windows, SharedCalib &calib,
                    const JointConfig &cfg, JointReport &rep,
                    const WindowBiasPrior &initial_prior,
                    BiasLinkedReport *detail = nullptr, PreintStore *store = nullptr,
                    std::vector<WindowWarmState> *warm_out = nullptr);
};
}
#endif
