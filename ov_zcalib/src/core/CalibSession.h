/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Joao Leonardo Silva Cotta
 *
 * ov_zcalib: live-session state machine + guided-excitation progress.
 * The streaming one-pass Lambda-sum drives the DISPLAY ONLY; the committed answer
 * always comes from the end-of-session VarPro refinement (JointCalib) over the
 * retained windows — committing the one-pass estimate is a correctness bug by
 * contract. Prompts are driven by the weakest eigenpairs of the prior-whitened
 * information sum (marginal sigmas lie under correlation).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_ZCALIB_CALIB_SESSION_H
#define OV_ZCALIB_CALIB_SESSION_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ov_zcalib {

enum class SessionState { SETTLE, COLLECT, SOLVE_REFINE, VERIFY, COMMIT, THERMAL_HOLD, ABORT };

/**
 * @brief Streaming display-side fusion: accumulates whitened window information and
 *        maps the weakest mode to an operator prompt (variation, not constancy:
 *        Dw wants per-axis oscillation, Da wants gravity re-orientation sweeps,
 *        td wants angular rate, tr wants row coverage).
 */
class CalibSession {
public:
  CalibSession(int np, const Eigen::VectorXd &prior_sigma, const std::vector<std::string> &labels)
      : L_(Eigen::MatrixXd::Zero(np, np)), prior_(prior_sigma), labels_(labels) {}

  void add_window_information(const Eigen::MatrixXd &Lambda_w) { L_ += Lambda_w; }

  /// Posterior/prior improvement per dof and the weakest whitened mode
  void progress(Eigen::VectorXd &improve, std::string &prompt) const {
    const int np = (int)L_.rows();
    Eigen::MatrixXd Lw = prior_.asDiagonal() * L_ * prior_.asDiagonal();
    Lw.diagonal().array() += 1.0; // + whitened prior
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Lw);
    const Eigen::MatrixXd Sw = ldlt.solve(Eigen::MatrixXd::Identity(np, np));
    improve = Sw.diagonal().cwiseMax(1e-300).cwiseSqrt().cwiseInverse(); // >1 means beating prior
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Lw);
    int imax;
    eig.eigenvectors().col(0).cwiseAbs().maxCoeff(&imax);
    const std::string &weak = labels_[imax];
    if (weak.rfind("dw", 0) == 0)
      prompt = "oscillate rotation about the weak gyro axis (" + weak + ")";
    else if (weak.rfind("da", 0) == 0)
      prompt = "slow tilt sweeps re-orienting gravity (" + weak + ")";
    else if (weak.rfind("td", 0) == 0)
      prompt = "increase angular rate (time offset weak)";
    else if (weak.rfind("tr", 0) == 0)
      prompt = "sweep the scene vertically for row coverage (readout weak)";
    else if (weak.rfind("p_IinC", 0) == 0)
      prompt = "sharp rotation bursts about two axes (lever arm weak)";
    else
      prompt = "keep varied 6-axis motion (" + weak + " weak)";
  }

private:
  Eigen::MatrixXd L_;
  Eigen::VectorXd prior_;
  std::vector<std::string> labels_;
};

} // namespace ov_zcalib

#endif // OV_ZCALIB_CALIB_SESSION_H
