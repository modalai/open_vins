/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
namespace {
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
using Mat=Eigen::MatrixXd;
using HF=UpdaterHelper::UpdaterHelperFeature;
using Rep=LandmarkRepresentation::Representation;
int checks=0,failures=0;
double max_state_fd=0.,max_feature_fd=0.,max_gauge=0.,max_public_cov=0.;
void check(bool pass,const char *why) {
  ++checks;if(!pass){++failures;std::printf("FAIL: %s\n",why);}
}
bool finite_matrix(const Mat &matrix) {
  for(Eigen::Index i=0;i<matrix.size();++i){std::uint64_t bits;const double value=matrix.data()[i];std::memcpy(&bits,&value,sizeof(bits));
    if((bits&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))return false;}return true;
}
const std::vector<Rep> representations{LandmarkRepresentation::GLOBAL_3D,LandmarkRepresentation::ANCHORED_3D,
  LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH,LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH,
  LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE};
bool anchored(Rep r){return LandmarkRepresentation::is_relative_representation(r);}
int dimension(Rep r){return r==LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE?1:3;}
V3 to_local(const std::shared_ptr<State>&s,size_t c,double t,const V3&p,bool fej=false) {
  auto view=s->pose_for_camera(c,t),extr=s->_calib_IMUtoCAM.at(c);
  return extr->Rot()*(fej?view->Rot_fej():view->Rot())*(p-(fej?view->pos_fej():view->pos()))+extr->pos();
}
V3 to_world(const std::shared_ptr<State>&s,size_t c,double t,const V3&p,bool fej=false) {
  auto view=s->pose_for_camera(c,t),extr=s->_calib_IMUtoCAM.at(c);
  return (fej?view->Rot_fej():view->Rot()).transpose()*extr->Rot().transpose()*(p-extr->pos())+
         (fej?view->pos_fej():view->pos());
}
Eigen::Vector2d project(const V3&p) {return {400.*p.x()/p.z()+320.,405.*p.y()/p.z()+240.};}
std::vector<std::shared_ptr<Type>> variables(const std::shared_ptr<State>&s) {
  std::vector<std::shared_ptr<Type>> out{s->_imu};
  for(int c=0;c<s->_options.num_cameras;++c) {
    for(const auto &v:std::vector<std::shared_ptr<Type>>{s->cam_imu_dt_var(c),s->_calib_IMUtoCAM.at(c),
          s->_cam_intrinsics.at(c),s->_calib_camera_readout.at(c)}) if(v->id()>=0)out.push_back(v);
  }
  for(const auto &v:s->_exposure_poses)out.push_back(v.pose);
  for(const auto &l:s->_features_SLAM)out.push_back(l.second);
  std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a->id()<b->id();});return out;
}
void perturb(const std::shared_ptr<State>&s,int id,const Eigen::VectorXd&dx) {
  for(const auto &v:variables(s))if(v->id()==id){v->update(dx);return;}
  check(false,"finite-difference variable resolves to owned covariance Type");
}
struct Fixture {
  std::shared_ptr<State>s;
  V3 point{.9,.25,4.8};
  std::vector<double>raw;
  Fixture(bool fej=false,bool extrinsic=true) {
    StateOptions o;o.num_cameras=2;o.max_clone_size=6;o.max_aruco_features=0;o.do_fej=fej;
    o.do_calib_camera_timeoffset=true;o.do_calib_camera_pose=extrinsic;o.imu_model=StateOptions::RPNG;
    o.physical_camera_clones=true;o.integration_method=StateOptions::DISCRETE;
    check(o.configure_clone_policy(false,false),"physical visual fixture uses bounded camera pose policy");
    s=std::make_shared<State>(o);s->_timestamp=9.98;
    for(int c=0;c<2;++c) {
      Eigen::VectorXd intr(8);intr<<400,405,320,240,0,0,0,0;auto camera=std::make_shared<CamRadtan>(640,480);camera->set_value(intr);
      s->_cam_intrinsics_cameras[c]=camera;s->_cam_intrinsics[c]->set_value(intr);s->_cam_intrinsics[c]->set_fej(intr);
      Eigen::Matrix<double,7,1>extr;extr<<rot_2_quat(exp_so3(c?V3(.035,-.018,.012):V3(-.01,.015,.005))),.08*c,.015*c,0.;
      s->_calib_IMUtoCAM[c]->set_value(extr);s->_calib_IMUtoCAM[c]->set_fej(extr);
      Eigen::VectorXd td(1);td<< (c?-.006:.003);s->cam_imu_dt_var(c)->set_value(td);s->cam_imu_dt_var(c)->set_fej(td);
    }
    Eigen::Matrix<double,16,1>x=s->_imu->value();x.segment<3>(7)<<.65,.08,.04;s->_imu->set_value(x);s->_imu->set_fej(x);
    NoiseManager noise;Propagator prop(noise,9.81);
    for(int k=0;k<=180;++k){ImuData z;z.timestamp=9.95+.005*k;z.wm=V3(.2,-.12,.16);z.am=V3(.1,-.08,9.81);prop.feed_imu(z);}
    for(int k=0;k<6;++k) {
      double t=10.+.1*k;raw.push_back(t);
      for(int c:{1,0}) {
        const double endpoint=t+s->cam_imu_dt(c);Propagator::EndpointKinematics rates;
        check(prop.propagate_to_imu(s,endpoint,endpoint-s->cam_imu_dt_ref(),rates),"fixture propagates to each true camera endpoint");
        State::ExposurePose v;v.camera_id=c;v.raw_time=t;v.imu_time=endpoint;v.pose=StateHelper::augment_pose_view(s,c,rates.omega);
        v.kinematics.omega=rates.omega;v.kinematics.omega_fej=rates.omega_fej;v.kinematics.vel=s->_imu->vel();v.kinematics.vel_fej=s->_imu->vel_fej();
        s->_exposure_poses.push_back(v);
      }
    }
    if(fej)for(auto &v:s->_exposure_poses) {
      Eigen::Matrix<double,7,1>f=v.pose->value();f.head<4>()=rot_2_quat(exp_so3(V3(.004,-.003,.002))*v.pose->Rot());
      f.tail<3>()+=V3(.007,-.004,.003);v.pose->set_fej(f);
    }
    check(s->_clones_IMU.empty()&&s->clone_count()==12,"physical fixture has no legacy raw-time clone aliases");
  }
  std::shared_ptr<Feature>track(bool noise=false,bool both=true)const {
    auto f=std::make_shared<Feature>();f->featid=123;f->quality=1.;
    for(int c:both?std::vector<int>{0,1}:std::vector<int>{1})for(size_t k=0;k<raw.size();++k) {
      Eigen::Vector2f uv=project(to_local(s,c,raw[k],point)).cast<float>();
      if(noise){uv.x()+=.025*std::sin(.7*k+c);uv.y()+=.02*std::cos(.6*k-c);}
      Eigen::Vector2f un;un<<(uv.x()-320)/400.,(uv.y()-240)/405.;
      f->timestamps[c].push_back(raw[k]);f->uvs[c].push_back(uv);f->uvs_norm[c].push_back(un);
    }
    return f;
  }
  HF feature(Rep r,bool distinct_fej=false)const {
    auto track_value=track();HF f;f.featid=123;f.feat_representation=r;f.timestamps=track_value->timestamps;
    f.uvs=track_value->uvs;f.uvs_norm=track_value->uvs_norm;f.anchor_cam_id=1;f.anchor_clone_timestamp=raw.front();
    f.p_FinG=point;f.p_FinG_fej=point+(distinct_fej?V3(.12,-.08,.16):V3::Zero());
    f.p_FinA=to_local(s,1,raw.front(),point);f.p_FinA_fej=f.p_FinA+V3(-.13,.09,.2);return f;
  }
};
struct Linear{Mat Hf,Hx;Eigen::VectorXd r;std::vector<std::shared_ptr<Type>>order;};
Linear linear(const std::shared_ptr<State>&s,HF f){Linear a;UpdaterHelper::get_feature_jacobian_full(s,f,a.Hf,a.Hx,a.r,a.order);return a;}
Mat aligned(const std::shared_ptr<State>&s,const Linear&a){Mat out=Mat::Zero(a.Hx.rows(),s->max_covariance_size());int col=0;
  for(auto &v:a.order){out.middleCols(v->id(),v->size())=a.Hx.middleCols(col,v->size());col+=v->size();}return out;}
V3 global_point(const std::shared_ptr<State>&s,const HF&f){return anchored(f.feat_representation)?
  to_world(s,f.anchor_cam_id,f.anchor_clone_timestamp,f.p_FinA):f.p_FinG;}
Eigen::VectorXd prediction(const std::shared_ptr<State>&s,const HF&f){
  size_t n=0;for(const auto &c:f.timestamps)n+=c.second.size();Eigen::VectorXd out(2*n);int row=0;const V3 p=global_point(s,f);
  for(const auto &c:f.timestamps)for(double t:c.second){out.segment<2>(row)=project(to_local(s,c.first,t,p));row+=2;}return out;
}
void projection_and_fd() {
  for(bool fej:{false,true})for(Rep rep:representations) {
    Fixture a(fej);auto f=a.feature(rep,true);const auto L=linear(a.s,f);const Mat H=aligned(a.s,L);
    check(L.r.norm()<8e-5,"physical exposure mean matches independent double projection to float-observation rounding");
    for(const auto &v:L.order)check(v!=a.s->cam_imu_dt_var(0)&&v!=a.s->cam_imu_dt_var(1)&&
       v!=a.s->_imu->bg()&&v!=a.s->_imu->ba(),"physical measurement has no duplicate temporal or bridge-bias columns");
    auto lin=StateHelper::clone_state(a.s);HF fl=f;
    // Production FEJ has two deliberate point conventions: global uses its
    // stored frozen world point; anchored lifts the best current world point
    // into the frozen anchor pose. Never substitute pFinA_fej for that lift.
    const V3 point=fej&&!anchored(rep)?f.p_FinG_fej:global_point(a.s,f);
    if(fej)for(auto &v:lin->_exposure_poses)v.pose->set_value(v.pose->fej());
    lin->_options.do_fej=false;
    if(anchored(rep))fl.p_FinA=to_local(lin,fl.anchor_cam_id,fl.anchor_clone_timestamp,point);else fl.p_FinG=point;
    int column=0;double best_state=1e100,best_feature=1e100;
    for(double eps:{1e-5,1e-6}) {
      double state_error=0.,feature_error=0.;column=0;
      for(const auto &v:L.order) {
        for(int c=0;c<v->size();++c) {
          auto plus=StateHelper::clone_state(lin),minus=StateHelper::clone_state(lin);Eigen::VectorXd dx=Eigen::VectorXd::Zero(v->size());dx(c)=eps;
          perturb(plus,v->id(),dx);perturb(minus,v->id(),-dx);
          Eigen::VectorXd fd=(prediction(plus,fl)-prediction(minus,fl))/(2*eps);
          state_error=std::max(state_error,(fd-L.Hx.col(column+c)).cwiseAbs().maxCoeff());
        }column+=v->size();
      }
      for(int c=0;c<dimension(rep);++c) {
        HF plus=fl,minus=fl;Eigen::VectorXd dx=Eigen::VectorXd::Zero(dimension(rep));dx(c)=eps;
        if(anchored(rep)) {
          Landmark p(dimension(rep)),m(dimension(rep));p._feat_representation=m._feat_representation=rep;
          p.set_from_xyz(fl.p_FinA,false);m.set_from_xyz(fl.p_FinA,false);p.update(dx);m.update(-dx);
          plus.p_FinA=p.get_xyz(false);minus.p_FinA=m.get_xyz(false);
        }else{plus.p_FinG+=dx;minus.p_FinG-=dx;}
        Eigen::VectorXd fd=(prediction(lin,plus)-prediction(lin,minus))/(2*eps);
        feature_error=std::max(feature_error,(fd-L.Hf.col(c)).cwiseAbs().maxCoeff());
      }
      best_state=std::min(best_state,state_error);best_feature=std::min(best_feature,feature_error);
    }
    max_state_fd=std::max(max_state_fd,best_state);max_feature_fd=std::max(max_feature_fd,best_feature);
    check(best_state<2e-5&&best_feature<2e-5,"physical observer/anchor/extrinsic/feature derivatives match double geometry at stated current or FEJ point");
    const auto P=StateHelper::get_full_covariance(a.s);
    check((P.row(a.s->cam_imu_dt_var(0)->id())*H.transpose()).norm()>1e-8,
          "independent active camera clock still couples to visual innovation through pose covariance");
    // Current clock estimates and stale legacy caches may not warp owned poses a second time.
    for(int c=0;c<2;++c){Eigen::VectorXd td(1);td<<.04-.08*c;a.s->cam_imu_dt_var(c)->set_value(td);
      for(double t:a.raw){a.s->_epoch_residuals[t][c]=.1;PreintBridgeData b;b.valid=true;b.dt=.1;b.DR=exp_so3(V3(.1,.2,-.1));a.s->_epoch_bridges[t][c]=b;}}
    auto unchanged=linear(a.s,f);
    check((L.r-unchanged.r).norm()==0.&&(L.Hx-unchanged.Hx).norm()==0.&&(L.Hf-unchanged.Hf).norm()==0.,
          "owned exposure mean and Jacobian ignore later td estimates and legacy bridge payloads");
  }
}
void nullspaces_and_owners() {
  for(bool fej:{false,true}) {
    Fixture a(fej);const auto first=a.raw.front();
    check(a.s->pose_for_camera(0,first)!=a.s->pose_for_camera(1,first)&&
          (a.s->pose_for_camera(0,first)->pos()-a.s->pose_for_camera(1,first)->pos()).norm()>1e-4,
          "same raw timestamp resolves different camera exposure poses");
    for(Rep rep:representations) {
      auto f=a.feature(rep,true);auto L=linear(a.s,f);Mat H=aligned(a.s,L);
      Mat N=Mat::Zero(a.s->max_covariance_size(),4);const V3 z(0,0,1);
      for(const auto &v:a.s->_exposure_poses) {
        const M3 R=fej?v.pose->Rot_fej():v.pose->Rot();const V3 p=fej?v.pose->pos_fej():v.pose->pos();
        N.block<3,3>(v.pose->id()+3,0).setIdentity();N.block<3,1>(v.pose->id(),3)=R*z;
        N.block<3,1>(v.pose->id()+3,3)=-skew_x(p)*z;
      }
      Mat gauge=H*N;
      if(!anchored(rep)){Mat nf=Mat::Zero(3,4);nf.leftCols(3).setIdentity();nf.col(3)=-skew_x(fej?f.p_FinG_fej:f.p_FinG)*z;gauge+=L.Hf*nf;}
      max_gauge=std::max(max_gauge,gauge.cwiseAbs().maxCoeff());
      check(gauge.cwiseAbs().maxCoeff()<2e-10,"physical full system preserves global translation and yaw nullspaces at its actual point convention");
      if(anchored(rep)) {
        f.timestamps={{1,{first}}};f.uvs={{1,{a.track()->uvs.at(1).front()}}};f.uvs_norm={{1,{Eigen::Vector2f::Zero()}}};
        auto own=linear(a.s,f);
        check(own.r.norm()<4e-5&&own.Hx.norm()<2e-10,"same-owner physical anchor and observer state/extrinsic columns cancel exactly");
      }
    }
    // Compare coordinate choices only at matched world linearization points.
    auto gf=a.feature(LandmarkRepresentation::GLOBAL_3D,false);auto g=linear(a.s,gf);Mat gx=aligned(a.s,g);auto gr=g.r;
    UpdaterHelper::nullspace_project_inplace(g.Hf,gx,gr);
    for(Rep rep:{LandmarkRepresentation::ANCHORED_3D,LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH}) {
      auto l=linear(a.s,a.feature(rep,false));Mat hx=aligned(a.s,l);auto r=l.r;UpdaterHelper::nullspace_project_inplace(l.Hf,hx,r);
      check((gx.transpose()*gx-hx.transpose()*hx).norm()/std::max(1.,(gx.transpose()*gx).norm())<2e-10&&
            (gx.transpose()*gr-hx.transpose()*r).norm()<2e-6,"global/anchored physical systems preserve information and score after feature elimination");
    }
    auto track=a.track();a.s->_exposure_poses.erase(a.s->_exposure_poses.begin());
    UpdaterHelper::clean_feature_measurements(a.s,*track,a.raw);
    check(track->timestamps.at(0).size()==6&&track->timestamps.at(1).size()==5&&track->uvs.at(1).size()==5&&
          track->uvs_norm.at(1).size()==5,"physical measurement cleaning uses camera/raw ownership, not union of raw times");
  }
}
struct SlamProbe:UpdaterSLAM{using UpdaterSLAM::UpdaterSLAM;using UpdaterSLAM::perform_anchor_change;};
void public_updates() {
  UpdaterOptions options,aruco;FeatureInitializerOptions init;
  for(bool fej:{false,true}) {
    Fixture a(fej,false);auto global=StateHelper::clone_state(a.s),anchor=StateHelper::clone_state(a.s);
    global->_options.feat_rep_msckf=LandmarkRepresentation::GLOBAL_3D;anchor->_options.feat_rep_msckf=LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    std::vector<std::shared_ptr<Feature>>g{a.track(true)},b{a.track(true)};UpdaterMSCKF ug(options,init),ub(options,init);
    ug.update(global,g);ub.update(anchor,b);
    check(g.size()==1&&b.size()==1,"actual physical MSCKF accepts coordinate-matched tracks with unchanged gates");
    const double cov=(StateHelper::get_full_covariance(global)-StateHelper::get_full_covariance(anchor)).cwiseAbs().maxCoeff();
    max_public_cov=std::max(max_public_cov,cov);
    check(cov<2e-10&&(global->_imu->value()-anchor->_imu->value()).norm()<2e-9,
          "actual physical MSCKF triangulation/handoff preserves mean and covariance across 3D coordinate charts");
    Fixture slam(fej,false);slam.s->_options.feat_rep_slam=LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    std::vector<std::shared_ptr<Feature>>tracks{slam.track(false,false)};SlamProbe updater(options,aruco,init);updater.delayed_init(slam.s,tracks);
    check(slam.s->_features_SLAM.count(123)==1,"actual physical anchored SLAM delayed initialization succeeds");
    if(!slam.s->_features_SLAM.count(123))continue;
    auto landmark=slam.s->_features_SLAM.at(123);
    const V3 world=to_world(slam.s,landmark->_anchor_cam_id,landmark->_anchor_clone_timestamp,landmark->get_xyz(false));
    const V3 frozen=to_world(slam.s,landmark->_anchor_cam_id,landmark->_anchor_clone_timestamp,landmark->get_xyz(true),true);
    check((world-slam.point).norm()<4e-5,"physical SLAM anchor coordinates reconstruct true triangulated world point");
    for(const auto &key:std::vector<std::pair<size_t,double>>{{0,slam.raw.front()},{1,slam.raw.back()},{0,slam.raw[2]}}) {
      updater.perform_anchor_change(slam.s,landmark,key.second,key.first);
      check((to_world(slam.s,key.first,key.second,landmark->get_xyz(false))-world).norm()<2e-11&&
            (to_world(slam.s,key.first,key.second,landmark->get_xyz(true),true)-frozen).norm()<2e-11,
            "physical SLAM reanchor retains current and FEJ world means across raw-time aliases and cameras");
    }
    std::vector<std::shared_ptr<Feature>>observations{slam.track(true)};updater.update(slam.s,observations);
    check(observations.size()==1&&finite_matrix(StateHelper::get_full_covariance(slam.s)),
          "actual persistent physical SLAM update uses owned observations after repeated reanchor");
  }
}
} // namespace
int main(){Printer::setPrintLevel("ERROR");projection_and_fd();nullspaces_and_owners();public_updates();
  std::printf("PHYSICAL_VISUAL %s checks=%d failures=%d maxStateFD=%.17g maxFeatureFD=%.17g maxGauge=%.17g maxPublicCov=%.17g\n",
      failures?"FAIL":"PASS",checks,failures,max_state_fd,max_feature_fd,max_gauge,max_public_cov);return failures?1:0;}
