/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "cam/CamRadtan.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "update/LegacyExposure.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <array>
#include <cstdio>

using namespace ov_core;
using namespace ov_msckf;
using namespace ov_type;
using V2 = Eigen::Vector2d;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using HF = UpdaterHelper::UpdaterHelperFeature;
using Intr = Eigen::Matrix<double, 8, 1>;

// Independent Rodrigues and Brown-Conrady defining equations, without the
// production SO(3), camera projection, or analytic Jacobian helpers.
M3 cross(const V3 &v) {
  M3 a; a << 0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.; return a;
}
M3 rotation(const V3 &v) {
  const double a=v.norm(); const M3 K=cross(v);
  const double s=a<1e-7 ? 1.-a*a/6.+a*a*a*a/120. : std::sin(a)/a;
  const double c=a<1e-7 ? .5-a*a/24.+a*a*a*a/720. : (1.-std::cos(a))/(a*a);
  return M3::Identity()+s*K+c*K*K;
}
V2 project(const V3 &p,const Intr &k) {
  const double x=p.x()/p.z(),y=p.y()/p.z(),r=x*x+y*y;
  const double radial=1.+k(4)*r+k(5)*r*r;
  return {k(0)*(x*radial+2.*k(6)*x*y+k(7)*(r+2.*x*x))+k(2),
          k(1)*(y*radial+k(6)*(r+2.*y*y)+2.*k(7)*x*y)+k(3)};
}
struct Map {
  M3 R,C;
  V3 p,f,c,w,u,v;
  Intr k;
  double tau;
  int representation=0;
  V3 lambda=V3::Zero();
  V3 point() const {
    if(representation==0)return f;
    const V3 anchor=representation==1 ? lambda : V3(lambda.x()/lambda.z(),lambda.y()/lambda.z(),1./lambda.z());
    return R.transpose()*C.transpose()*(anchor-c)+p;
  }
  V2 normalized(bool body) const {
    const M3 E=rotation(-w*tau);
    const V3 y=body ? V3(E*(R*(point()-p)-u*tau)) : V3(E*R*(point()-p-v*tau));
    const V3 point=C*y+c; return point.head<2>()/point.z();
  }
  V2 predict(bool body) const { const V2 uv=normalized(body); return project(V3(uv.x(),uv.y(),1.),k); }
};

V2 camera_jet(const Map &value,const Map &origin,const Map &current,bool body,bool fej) {
  if(!fej)return value.predict(body);
  const V2 uv=current.normalized(body);
  Eigen::Matrix2d J;const double h=2e-5;
  for(int axis=0;axis<2;++axis) {
    V3 a(uv.x(),uv.y(),1.),b=a,c=a,d=a;
    a(axis)+=2*h;b(axis)+=h;c(axis)-=h;d(axis)-=2*h;
    J.col(axis)=(-project(a,current.k)+8*project(b,current.k)-8*project(c,current.k)+project(d,current.k))/(12*h);
  }
  return project(V3(uv.x(),uv.y(),1.),value.k)+J*(value.normalized(body)-origin.normalized(body));
}

struct Fixture {
  std::shared_ptr<State> s;
  std::shared_ptr<PoseJPL> pose;
  HF feature;
  Fixture(bool fej,bool shifted,double tau,int representation,bool unequal_creation) {
    StateOptions o; o.num_cameras=2; o.do_fej=fej;
    o.do_calib_camera_pose=true; o.do_calib_camera_intrinsics=true;
    o.do_calib_camera_timeoffset=true; o.max_aruco_features=0;
    s=std::make_shared<State>(o);
    Intr intr; intr << 400.,410.,320.,240.,-.03,.006,.002,-.003;
    for(int camera=0;camera<2;++camera) {
      auto model=std::make_shared<CamRadtan>(640,480); model->set_value(intr);
      s->_cam_intrinsics_cameras[camera]=model;
      s->_cam_intrinsics[camera]->set_value(intr); s->_cam_intrinsics[camera]->set_fej(intr);
      Eigen::Matrix<double,7,1> e;
      e << rot_2_quat(rotation(V3(.03,-.04,.015))),.08,.01,-.02;
      s->_calib_IMUtoCAM[camera]->set_value(e);s->_calib_IMUtoCAM[camera]->set_fej(e);
      Eigen::VectorXd t(1);t << (camera ? tau : 0.);
      s->cam_imu_dt_var(camera)->set_value(t);s->cam_imu_dt_var(camera)->set_fej(t);
    }
    Eigen::Matrix<double,16,1> imu=s->_imu->value();
    imu.head<4>()=rot_2_quat(rotation(V3(.11,-.06,.17)));
    imu.segment<3>(4)=V3(.2,-.15,.12);
    imu.segment<3>(7)=V3(1.3,-.7,.2);
    s->_imu->set_value(imu);s->_imu->set_fej(imu);s->_timestamp=10.;
    if(unequal_creation) {
      Eigen::Matrix<double,16,1> first=imu;
      first.head<4>()=rot_2_quat(rotation(V3(-.02,.014,-.011))*s->_imu->Rot());
      first.segment<3>(7)+=V3(.08,-.04,.03);s->_imu->set_fej(first);
    }
    StateHelper::augment_clone(s,V3(.7,-.4,.9),V3(.7,-.4,.9));
    pose=s->_clones_IMU.at(10.);
    feature.featid=5;
    feature.feat_representation=representation==0 ? LandmarkRepresentation::GLOBAL_3D :
      representation==1 ? LandmarkRepresentation::ANCHORED_3D : LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH;
    feature.p_FinG=feature.p_FinG_fej=V3(1.3,.7,5.4);
    if(shifted) {
      Eigen::VectorXd d(6);d << .024,-.015,.019,.03,-.02,.01;pose->update(d);
      feature.p_FinG+=V3(.02,.01,-.025);
      Eigen::VectorXd t(1);t << tau+(tau==0. ? 0. : .0012);s->cam_imu_dt_var(1)->set_value(t);
    }
    feature.anchor_cam_id=1;feature.anchor_clone_timestamp=10.;
    feature.p_FinA=s->_calib_IMUtoCAM.at(1)->Rot()*pose->Rot()*(feature.p_FinG-pose->pos())+s->_calib_IMUtoCAM.at(1)->pos();
    feature.p_FinA_fej=feature.p_FinA;
    feature.timestamps[1]={10.};
    feature.uvs[1].push_back(map(false).predict(true).cast<float>());
    feature.uvs_norm[1].push_back(Eigen::Vector2f(.2,.1));
  }
  Map map(bool linearization) const {
    Map a;const bool fej=linearization&&s->_options.do_fej;
    a.R=fej?pose->Rot_fej():pose->Rot();a.p=fej?pose->pos_fej():pose->pos();
    a.f=fej?feature.p_FinG_fej:feature.p_FinG;
    a.C=s->_calib_IMUtoCAM.at(1)->Rot();a.c=s->_calib_IMUtoCAM.at(1)->pos();
    a.k=s->_cam_intrinsics.at(1)->value();
    const auto &kin=s->_clones_kinematics.at(10.);
    a.w=fej?kin.omega_fej:kin.omega;a.v=fej?kin.vel_fej:kin.vel;
    a.u=pose->Rot_fej()*a.v;
    a.tau=fej?s->cam_imu_dt_var(1)->fej()(0)-s->cam_imu_dt_var(0)->fej()(0)
             :s->cam_imu_dt_var(1)->value()(0)-s->cam_imu_dt_var(0)->value()(0);
    if(feature.feat_representation!=LandmarkRepresentation::GLOBAL_3D) {
      a.representation=feature.feat_representation==LandmarkRepresentation::ANCHORED_3D ? 1 : 2;
      const V3 best=pose->Rot().transpose()*a.C.transpose()*(feature.p_FinA-a.c)+pose->pos();
      const V3 anchor=a.C*a.R*(best-a.p)+a.c;
      a.lambda=a.representation==1 ? anchor : V3(anchor.x()/anchor.z(),anchor.y()/anchor.z(),1./anchor.z());
      a.f=best;
    }
    return a;
  }
  Map perturb(Map a,const std::shared_ptr<Type> &type,int axis,double amount) const {
    if(type==pose) {
      if(axis<3)a.R=rotation(-amount*V3::Unit(axis))*a.R;
      else a.p(axis-3)+=amount;
    } else if(type==s->_calib_IMUtoCAM.at(1)) {
      if(axis<3)a.C=rotation(-amount*V3::Unit(axis))*a.C;
      else a.c(axis-3)+=amount;
    } else if(type==s->_cam_intrinsics.at(1))a.k(axis)+=amount;
    else if(type==s->cam_imu_dt_var(1))a.tau+=amount;
    else if(type==s->cam_imu_dt_var(0))a.tau-=amount;
    else std::abort();
    return a;
  }
};

int main(int argc,char **argv) {
  const bool expect_predecessor=argc==2&&std::string(argv[1])=="--expect-predecessor";
  int failures=0;
  int scenarios=0;double max_legacy_gauge=0.,max_body_gauge=0.,max_legacy_fd=0.,max_body_fd=0.;
  for(int representation:{0,1,2})for(bool unequal:{false,true})for(bool fej:{false,true})for(bool shifted:{false,true})for(double tau:{0.,.006,-.012}) {
    Fixture f(fej,shifted,tau,representation,unequal);
    Eigen::MatrixXd Hf,Hx;Eigen::VectorXd residual;std::vector<std::shared_ptr<Type>> order;
    UpdaterHelper::get_feature_jacobian_full(f.s,f.feature,Hf,Hx,residual,order);
    Eigen::MatrixXd body(2,Hx.cols()+3),world(2,Hx.cols()+3),actual(2,Hx.cols()+3);
    actual << Hx,Hf;
    const Map lin=f.map(true),current=f.map(false);
    const auto evaluate=[&](const Map &a,bool body){return camera_jet(a,lin,current,body,fej);};
    constexpr double h=2e-5;
    int col=0,pose_col=-1;
    for(const auto &type:order) {
      if(type==f.pose)pose_col=col;
      for(int k=0;k<type->size();++k,++col) {
        for(bool attached:{false,true}) {
          const V2 d=(-evaluate(f.perturb(lin,type,k,2*h),attached)+8*evaluate(f.perturb(lin,type,k,h),attached)
                      -8*evaluate(f.perturb(lin,type,k,-h),attached)+evaluate(f.perturb(lin,type,k,-2*h),attached))/(12*h);
          (attached?body:world).col(col)=d;
        }
      }
    }
    for(int k=0;k<3;++k)for(bool attached:{false,true}) {
      Map a=lin,b=lin,c=lin,d=lin;
      if(representation) {a.lambda(k)+=2*h;b.lambda(k)+=h;c.lambda(k)-=h;d.lambda(k)-=2*h;}
      else {a.f(k)+=2*h;b.f(k)+=h;c.f(k)-=h;d.f(k)-=2*h;}
      (attached?body:world).col(col+k)=(-evaluate(a,attached)+8*evaluate(b,attached)-8*evaluate(c,attached)+evaluate(d,attached))/(12*h);
    }
    Eigen::MatrixXd N=Eigen::MatrixXd::Zero(Hx.cols()+3,4);
    N.block<3,1>(pose_col,0)=lin.R*V3::UnitZ();
    N.block<3,1>(pose_col+3,0)=-cross(lin.p)*V3::UnitZ();
    if(!representation)N.block<3,1>(Hx.cols(),0)=-cross(lin.f)*V3::UnitZ();
    N.block<3,3>(pose_col+3,1).setIdentity();if(!representation)N.block<3,3>(Hx.cols(),1).setIdentity();
    const double old_gauge=(actual*N).cwiseAbs().maxCoeff(),new_gauge=(body*N).cwiseAbs().maxCoeff();
    const double old_fd=(actual-world).cwiseAbs().maxCoeff(),new_fd=(actual-body).cwiseAbs().maxCoeff();
    max_legacy_gauge=std::max(max_legacy_gauge,old_gauge);max_body_gauge=std::max(max_body_gauge,new_gauge);
    max_legacy_fd=std::max(max_legacy_fd,old_fd);max_body_fd=std::max(max_body_fd,new_fd);
    double equivariance=0.;
    for(double yaw:{-.7,.4,1.1}) {
      const M3 G=rotation(V3(0,0,yaw));const V3 t(.8,-.3,.2);
      Map moved=lin;moved.R=lin.R*G.transpose();moved.p=G*lin.p+t;moved.f=G*lin.f+t;
      equivariance=std::max(equivariance,(moved.predict(true)-lin.predict(true)).norm());
    }
    std::printf("WARP rep=%d unequal=%d fej=%d shifted=%d tau=%+.4f legacy_gauge=%.12g body_gauge=%.12g legacy_world_FD=%.12g legacy_body_FD=%.12g body_equivariance=%.12g\n",
                representation,unequal,fej,shifted,tau,old_gauge,new_gauge,old_fd,new_fd,equivariance);
    if(new_gauge>2e-7||equivariance>1e-10)return 2;
    const double mean_error=(residual-(f.feature.uvs.at(1).front().cast<double>()-current.predict(true))).cwiseAbs().maxCoeff();
    if(!expect_predecessor&&(new_fd>2e-6||old_gauge>2e-9||mean_error>2e-10)) {
      ++failures;std::printf("FAIL actual helper body map/jet derivative gauge or mean: mean_error=%.12g\n",mean_error);
    }
    ++scenarios;
  }
  std::printf("WARP_CONTRACT scenarios=%d max_legacy_gauge=%.12g max_body_gauge=%.12g max_legacy_world_FD=%.12g max_legacy_body_FD=%.12g\n",scenarios,max_legacy_gauge,max_body_gauge,max_legacy_fd,max_body_fd);
  if(expect_predecessor)return max_legacy_gauge>1e-2&&max_body_fd>1e-2 ? 0 : 3;
  std::printf("LEGACY_EXPOSURE scenarios=%d failures=%d\n",scenarios,failures);
  return failures ? 1 : 0;
}
