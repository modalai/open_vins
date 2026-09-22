#include "BiasLinkedCalib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>

#include "PosteriorChecks.h"
#include "ceres_free/Parallel.h"
#include "utils/quat_ops.h"

namespace ov_zcalib {
namespace {
using Bias = Eigen::Matrix<double, 6, 1>;
struct Evaluation {
  bool ok = false;
  double cost = 0.0;
  Eigen::MatrixXd H;
  Eigen::VectorXd g;
  std::vector<WindowWarmState> warm;
  std::vector<WindowSolveReport> reports;
};
bool positive(double x) { return finite_scalar(x) && x > 0.0; }
}

bool BiasLinkedCalib::solve(const std::vector<WindowData> &windows, SharedCalib &calib,
                           const JointConfig &cfg, JointReport &rep,
                           const WindowBiasPrior &initial_prior, BiasLinkedReport *detail,
                           PreintStore *store, std::vector<WindowWarmState> *warm_out) {
  rep = JointReport{};
  if (detail) *detail = BiasLinkedReport{};
  const auto begin = std::chrono::steady_clock::now();
  const auto wall = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(); };
  const auto elapsed = [&]() { return cfg.budget_clock ? cfg.budget_clock() : wall(); };
  const auto expired = [&]() {
    if (cfg.max_wall_s < 0) return true;
    if (cfg.max_wall_s == 0) return false;
    const double t=elapsed();
    return !finite_scalar(t) || t < 0 || t >= cfg.max_wall_s;
  };
  const int nw = static_cast<int>(windows.size()), np = calib.local_dim();
  if (nw == 0 || np == 0 || cfg.window_max_iters <= 0 || cfg.outer_iterations <= 0 || cfg.max_backtracks < 0 ||
      !finite_scalar(cfg.max_wall_s) || expired() ||
      !finite_matrix(initial_prior.bg) || !finite_matrix(initial_prior.ba) ||
      !positive(calib.bg_prior_sigma) || !positive(calib.ba_prior_sigma) ||
      !positive(calib.noise.sigma_w) || !positive(calib.noise.sigma_a) ||
      !positive(calib.noise.sigma_wb) || !positive(calib.noise.sigma_ab)) return false;
  std::vector<int> order(nw);
  std::iota(order.begin(), order.end(), 0);
  for (const auto &w : windows) {
    if (w.clone_times.size() < 3 || !w.has_seeds) return false;
    for (size_t k=0; k<w.clone_times.size(); ++k)
      if (!finite_scalar(w.clone_times[k]) || (k && w.clone_times[k] <= w.clone_times[k-1])) return false;
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) { return windows[a].clone_times.front() < windows[b].clone_times.front(); });
  for (int i=1; i<nw; ++i)
    if (!(windows[order[i]].clone_times.front() > windows[order[i-1]].clone_times.back())) return false;

  const SharedCalib entry = calib;
  const auto fail = [&]() { calib = entry; rep.ok=false; rep.wall_s=wall(); return false; };
  // Choose the covariance reference once. Freezing noise mixing alone does
  // not freeze propagation; full interval weights are built below too.
  if (!calib.noise_frozen) {
    calib.noise_lin = calib.imu;
    calib.noise_frozen = true;
  }
  auto layout = calib.free_blocks();
  const int nn = 2*nw, dim = np+6*nn;
  std::vector<int> rank(nw);
  for (int i=0; i<nw; ++i) rank[order[i]]=i;
  std::vector<Bias> nodes(nn);
  for (auto &b : nodes) b << initial_prior.bg, initial_prior.ba;
  std::vector<WindowBiasPrior> priors(nw, initial_prior);
  std::vector<std::vector<double>> centers;
  rep.prior_sigma_vec.resize(np);
  int off=0;
  for (const auto &b : layout) {
    const auto cap=cfg.step_cap.find(b.name);
    if (cap!=cfg.step_cap.end() && !positive(cap->second)) return fail();
    centers.emplace_back(b.ptr,b.ptr+b.gsize);
    const bool cp = b.name=="cam" && cfg.use_cam_prior_vec && b.cam>=0 && static_cast<size_t>(b.cam)<cfg.cam_prior_vec.size();
    if (cp && cfg.use_cam_prior_center && static_cast<size_t>(b.cam)<cfg.cam_prior_center.size())
      for (int k=0;k<b.lsize;++k)
        if (cfg.cam_prior_vec[b.cam](k)>1e-8) centers.back()[k]=cfg.cam_prior_center[b.cam](k);
    for (int k=0;k<b.gsize;++k)
      if (!finite_scalar(b.ptr[k]) || !finite_scalar(centers.back()[k])) return fail();
    for (int k=0;k<b.lsize;++k) {
      const double s=cp ? cfg.cam_prior_vec[b.cam](k) :
        ((b.name=="da" && cfg.use_da_prior_vec) ? cfg.da_prior_vec(k) :
         (cfg.prior_sigma.count(b.name) ? cfg.prior_sigma.at(b.name) : 1.0));
      if (!positive(s)) return fail();
      rep.prior_sigma_vec(off+k)=s;
      rep.labels.push_back(b.label()+"["+std::to_string(k)+"]");
    }
    off+=b.lsize;
  }
  Eigen::VectorXd scales(dim);
  scales.head(np)=rep.prior_sigma_vec;
  for (int i=0;i<nn;++i) {
    scales.segment<3>(np+6*i).setConstant(calib.bg_prior_sigma);
    scales.segment<3>(np+6*i+3).setConstant(calib.ba_prior_sigma);
  }
  SharedCalib weight_reference=calib;
  weight_reference.imu=calib.noise_lin;
  std::vector<WindowEvaluationContext> weights(nw);
  for (int wi=0;wi<nw;++wi) {
    if (expired()) { rep.hit_wall_budget=true; return fail(); }
    if (!WindowBA::make_evaluation_context(windows[wi],weight_reference,weights[wi])) return fail();
  }
  const auto snapshot = [&]() {
    std::vector<std::vector<double>> s;
    for (const auto &b:layout) s.emplace_back(b.ptr,b.ptr+b.gsize);
    return s;
  };
  const auto restore = [&](const std::vector<std::vector<double>> &s) {
    for (size_t i=0;i<layout.size();++i) std::copy(s[i].begin(),s[i].end(),layout[i].ptr);
  };
  const auto boundary = [&](int wi) {
    WindowBoundaryBias b;
    const int i=2*rank[wi];
    b.bg_first=nodes[i].head<3>(); b.ba_first=nodes[i].tail<3>();
    b.bg_last=nodes[i+1].head<3>(); b.ba_last=nodes[i+1].tail<3>();
    b.include_first_prior=rank[wi]==0;
    // Only the immutable IMU weights are consumed. Context prior/gauge means
    // must not replace the explicit physical prior or each window's gauge.
    b.imu_weights=&weights[wi];
    return b;
  };
  PreintStore local_store;
  if (!store) store=&local_store;
  for (const auto &w:windows) store->ensure(w.uid);
  std::vector<WindowPreint*> slots(nw,nullptr);
  // Duplicate UIDs would race in the cache; use uncached slots in that case.
  for (int i=0;i<nw;++i) {
    bool unique=windows[i].uid!=0;
    for (int j=0;j<nw;++j) if (i!=j && windows[i].uid==windows[j].uid) unique=false;
    if (unique) slots[i]=store->ensure(windows[i].uid);
  }
  ov_init::zbft_sfm::ParallelExecutor pool(cfg.num_threads);
  std::vector<WindowWarmState> accepted_warm(nw);
  auto evaluate = [&]() {
    Evaluation e;
    e.H=Eigen::MatrixXd::Zero(dim,dim); e.g=Eigen::VectorXd::Zero(dim);
    e.warm=accepted_warm; e.reports.resize(nw);
    std::vector<char> ok(nw,0);
    const double pass_begin=wall();
    pool.parallel_dynamic(nw,[&](int,int wi) {
      if (expired()) return;
      const auto b=boundary(wi);
      ok[wi]=WindowBA::solve_and_export(windows[wi],calib,true,e.reports[wi],cfg.window_max_iters,
          false,&e.warm[wi],slots[wi],nullptr,nullptr,&priors[wi],&b);
    });
    rep.max_pass_s=std::max(rep.max_pass_s,wall()-pass_begin);
    ++rep.evaluation_passes;
    for (int wi=0;wi<nw;++wi) {
      const auto &r=e.reports[wi];
      rep.t_preint_sum+=r.t_preint; rep.t_inner_sum+=r.t_inner; rep.t_export_sum+=r.t_export; rep.t_factor_sum+=r.t_factor;
      rep.inner_iters_sum+=r.iterations; rep.time_stops+=r.time_stopped;
      if (r.preint_hit) ++rep.preint_hits; else ++rep.preint_misses;
      if (accepted_warm[wi].valid) ++rep.warm_evals; else ++rep.cold_evals;
      if (!ok[wi] || r.time_stopped || r.free_dim!=np+12 || r.Lambda.rows()!=np+12 ||
          r.Lambda.cols()!=np+12 || r.gred.size()!=np+12 || !finite_scalar(r.qn) ||
          !finite_scalar(r.cost_final) || !finite_matrix(r.Lambda) || !finite_matrix(r.gred)) return e;
      std::vector<int> map(np+12);
      for (int k=0;k<np;++k) map[k]=k;
      for (int k=0;k<12;++k) map[np+k]=np+12*rank[wi]+k;
      for (int a=0;a<np+12;++a) {
        e.g(map[a])+=r.gred(a);
        for (int b=0;b<np+12;++b) e.H(map[a],map[b])+=r.Lambda(a,b);
      }
      e.cost+=r.cost_final;
    }
    if (expired()) return e;
    // Only gaps receive a new RW factor. Each window's ACI factors already
    // account for bias evolution throughout its observed time span.
    for (int i=1;i<nw;++i) {
      const double dt=windows[order[i]].clone_times.front()-windows[order[i-1]].clone_times.back();
      for (int k=0;k<6;++k) {
        const double rw=k<3 ? calib.noise.sigma_wb : calib.noise.sigma_ab;
        const double variance=rw*rw*dt;
        if (!positive(variance)) return e;
        const double info=1.0/variance, r=nodes[2*i](k)-nodes[2*i-1](k);
        const int left=np+6*(2*i-1)+k, right=np+12*i+k;
        e.cost+=0.5*info*r*r;
        e.g(left)-=info*r; e.g(right)+=info*r;
        e.H(left,left)+=info; e.H(right,right)+=info;
        e.H(left,right)-=info; e.H(right,left)-=info;
      }
    }
    // Proper global calibration priors enter once. Use their actual local
    // Jacobian, including the JPL quaternion residual away from its center.
    int o=0;
    for (size_t bi=0;bi<layout.size();++bi) {
      const auto &b=layout[bi];
      Eigen::VectorXd r(b.lsize);
      Eigen::MatrixXd J=Eigen::MatrixXd::Identity(b.lsize,b.lsize);
      if (b.is_quat) {
        const Eigen::Vector4d q=ov_core::quat_multiply(Eigen::Map<const Eigen::Vector4d>(b.ptr),
            ov_core::Inv(Eigen::Vector4d(Eigen::Map<const Eigen::Vector4d>(centers[bi].data()))));
        r=2.0*q.head<3>();
        J=q(3)*Eigen::Matrix3d::Identity()+ov_core::skew_x(q.head<3>());
      } else for (int k=0;k<b.lsize;++k) r(k)=b.ptr[k]-centers[bi][k];
      const Eigen::VectorXd inv=rep.prior_sigma_vec.segment(o,b.lsize).cwiseInverse();
      const Eigen::MatrixXd A=inv.asDiagonal()*J;
      const Eigen::VectorXd z=inv.asDiagonal()*r;
      e.cost+=0.5*z.squaredNorm();
      e.H.block(o,o,b.lsize,b.lsize).noalias()+=A.transpose()*A;
      e.g.segment(o,b.lsize).noalias()+=A.transpose()*z;
      o+=b.lsize;
    }
    e.H=0.5*(e.H+e.H.transpose()).eval();
    e.ok=finite_scalar(e.cost) && finite_matrix(e.H) && finite_matrix(e.g);
    return e;
  };

  Evaluation accepted=evaluate();
  if (!accepted.ok) { rep.hit_wall_budget=expired(); return fail(); }
  accepted_warm=accepted.warm;
  auto accepted_p=snapshot(); auto accepted_nodes=nodes;
  std::vector<double> merits{accepted.cost};
  double lambda=1e-3;
  int rejected=0;
  for (int step=0;step<cfg.outer_iterations;) {
    if (expired() || (cfg.max_wall_s>0 && elapsed()+1.1*rep.max_pass_s>=cfg.max_wall_s)) {
      rep.hit_wall_budget=true; break;
    }
    Eigen::MatrixXd H=scales.asDiagonal()*accepted.H*scales.asDiagonal();
    const Eigen::VectorXd g=scales.asDiagonal()*accepted.g;
    H.diagonal()+=lambda*H.diagonal().cwiseMax(1.0);
    if (!finite_matrix(H) || !finite_matrix(g)) return fail();
    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info()!=Eigen::Success || !finite_matrix(ldlt.vectorD()) || (ldlt.vectorD().array()<=0).any()) return fail();
    Eigen::VectorXd dp=scales.asDiagonal()*ldlt.solve(-g);
    if (!finite_matrix(dp)) return fail();
    double scale=1.0;
    const double trust=dp.cwiseQuotient(scales).lpNorm<Eigen::Infinity>();
    if (trust>3.0) scale=3.0/trust;
    int o=0;
    for (const auto &b:layout) {
      const auto cap=cfg.step_cap.find(b.name);
      if (cap!=cfg.step_cap.end()) {
        const double magnitude=dp.segment(o,b.lsize).cwiseAbs().maxCoeff();
        if (magnitude>cap->second) scale=std::min(scale,cap->second/magnitude);
      }
      o+=b.lsize;
    }
    dp*=scale; rep.last_step_norm=dp.norm();
    if (rep.last_step_norm<1e-10) { rep.stopped_early=true; break; }
    o=0;
    for (auto &b:layout) {
      if (b.is_quat) {
        Eigen::Vector4d dq; dq<<0.5*dp.segment<3>(o),1.0; dq.normalize();
        Eigen::Map<Eigen::Vector4d> q(b.ptr);
        q=ov_core::quat_multiply(dq,Eigen::Vector4d(q));
      } else for (int k=0;k<b.lsize;++k) b.ptr[k]+=dp(o+k);
      o+=b.lsize;
    }
    for (int i=0;i<nn;++i) nodes[i]+=dp.segment<6>(np+6*i);
    Evaluation candidate=evaluate();
    if (candidate.ok && candidate.cost<accepted.cost) {
      accepted=std::move(candidate); accepted_warm=accepted.warm;
      accepted_p=snapshot(); accepted_nodes=nodes; merits.push_back(accepted.cost);
      lambda=std::max(1e-9,lambda/3.0); rejected=0; ++step; ++rep.accepted_passes;
      if (cfg.verbose) std::printf("[bias-linked] step %d merit %.9e\n",step,accepted.cost);
    } else {
      restore(accepted_p); nodes=accepted_nodes;
      if (expired()) { rep.hit_wall_budget=true; break; }
      lambda*=10.0;
      if (++rejected>cfg.max_backtracks) break;
    }
  }
  restore(accepted_p); nodes=accepted_nodes;
  // Marginalize the fitted boundary biases; never report conditional Hpp.
  const Eigen::MatrixXd H=scales.asDiagonal()*accepted.H*scales.asDiagonal();
  if (!finite_matrix(H)) return fail();
  Eigen::LDLT<Eigen::MatrixXd> bb(H.bottomRightCorner(dim-np,dim-np));
  if (bb.info()!=Eigen::Success || !finite_matrix(bb.vectorD()) || (bb.vectorD().array()<=0).any()) return fail();
  Eigen::MatrixXd marginal=H.topLeftCorner(np,np)-H.topRightCorner(np,dim-np)*bb.solve(H.bottomLeftCorner(dim-np,np));
  marginal=0.5*(marginal+marginal.transpose()).eval();
  Eigen::VectorXd scaled_sigma;
  if (!posterior_sigmas(marginal,scaled_sigma)) return fail();
  const auto inv=scales.head(np).cwiseInverse().eval();
  rep.Lambda=inv.asDiagonal()*marginal*inv.asDiagonal();
  rep.sigma=scales.head(np).cwiseProduct(scaled_sigma);
  if (!finite_matrix(rep.Lambda) || !finite_matrix(rep.sigma)) return fail();
  rep.ok=true; rep.dim_p=np; rep.windows_used=nw; rep.final_merit=accepted.cost; rep.wall_s=wall();
  for (const auto &r:accepted.reports) rep.qn_max_final=std::max(rep.qn_max_final,r.qn);
  if (warm_out) *warm_out=accepted.warm;
  if (detail) {
    detail->nodes=nn; detail->gap_links=nw-1; detail->accepted_merit=std::move(merits);
    detail->joint_information=accepted.H; detail->variable_scales=scales;
    for (int wi=0;wi<nw;++wi) {
      auto b=boundary(wi);
      b.imu_weights=nullptr; // solve-local weight storage must never escape
      detail->boundary.push_back(b);
    }
  }
  return true;
}
}
