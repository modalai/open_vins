/* Copyright (C) 2026 OpenVINS Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Vec.h"
#include "utils/print.h"

#ifndef TEST_OLD_VOID_API
#define TEST_OLD_VOID_API 0
#endif
using namespace ov_msckf;
using namespace ov_type;
namespace {
using Mat=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
int checks=0,failures=0;
void check(bool pass,const char *label){++checks;if(!pass){++failures;std::printf("FAIL: %s\n",label);}}
bool same(const Mat&a,const Mat&b){return a.rows()==b.rows()&&a.cols()==b.cols()&&std::memcmp(a.data(),b.data(),sizeof(double)*a.size())==0;}
double stored_value(std::uint64_t bits){volatile std::uint64_t source=bits;bits=source;double out;std::memcpy(&out,&bits,sizeof(out));return out;}
bool augment(const std::shared_ptr<State>&s,const std::shared_ptr<Type>&v,const std::vector<std::shared_ptr<Type>>&order,
             const Mat&Hx,const Mat&Hf,const Mat&R,const Vector&r){
#if TEST_OLD_VOID_API
  StateHelper::initialize_invertible(s,v,order,Hx,Hf,R,r);
  return v->id()>=0; // Compile the predecessor against its real void declaration.
#else
  return StateHelper::initialize_invertible(s,v,order,Hx,Hf,R,r);
#endif
}
struct Fixture {
  std::shared_ptr<State>s;
  std::shared_ptr<Vec>v=std::make_shared<Vec>(3);
  Mat P,Hx,Hf,R;Vector r;
  std::vector<std::shared_ptr<Type>>order;
  Fixture(){
    StateOptions o;s=std::make_shared<State>(o);order={s->_imu->bg(),s->_imu->p()};
    Mat L=Mat::Identity(15,15);
    for(int i=0;i<15;++i)for(int j=0;j<i;++j)L(i,j)=.03*std::sin(.6*i+.9*j);
    P=.2*L*L.transpose();StateHelper::set_initial_covariance(s,P,{s->_imu});
    Hx.resize(3,6);for(int i=0;i<3;++i)for(int j=0;j<6;++j)Hx(i,j)=.1*std::sin(.4+i+.6*j);
    Hf.resize(3,3);Hf<<1.,.2,-.1,-.3,.8,.15,.1,-.2,1.2;
    R=.1*Mat::Identity(3,3);r.resize(3);r<<.02,-.01,.03;
    Vector mean(3);mean<<.2,-.3,.4;v->set_value(mean);v->set_fej(mean+.1*Vector::Ones(3));
  }
};
void valid_case(double scale){
  Fixture f;f.Hf*=scale;
  Mat J=Mat::Zero(3,15);J.middleCols(9,3)=f.Hx.leftCols(3);J.middleCols(3,3)=f.Hx.rightCols(3);
  const auto factor=f.Hf.colPivHouseholderQr();
  Mat map=Mat::Zero(18,18);map.topLeftCorner(15,15).setIdentity();
  map.bottomLeftCorner(3,15)=-factor.solve(J);map.bottomRightCorner(3,3)=factor.solve(Mat::Identity(3,3));
  Mat prior=Mat::Zero(18,18);prior.topLeftCorner(15,15)=f.P;prior.bottomRightCorner(3,3)=f.R;
  const Mat expected=map*prior*map.transpose();const Mat old_imu=f.s->_imu->value(),old_fej=f.v->fej();
  const Vector expected_mean=f.v->value()+factor.solve(f.r);
  check(augment(f.s,f.v,f.order,f.Hx,f.Hf,f.R,f.r),"well-conditioned augmentation succeeds");
  const Mat actual=StateHelper::get_full_covariance(f.s);
  check(actual.rows()==18&&f.v->id()==15,"valid augmentation adds exactly three owned coordinates");
  if(actual.rows()==18)check((actual-expected).norm()<2e-12*(1+expected.norm()),"full covariance agrees with independent dense latent-variable map");
  check((f.v->value()-expected_mean).norm()<1e-12*(1+expected_mean.norm()),"mean agrees with independent QR solve");
  check(same(f.s->_imu->value(),old_imu)&&same(f.v->fej(),old_fej),"augmentation retains existing means and the proposed variable FEJ");
}
void invalid_case(int mode){
  Fixture f;
  if(mode==0)f.Hf.setZero();
  if(mode==1)f.Hf.col(2)=f.Hf.col(0)+2.*f.Hf.col(1);
  if(mode==2){f.Hf.setIdentity();f.Hf(2,2)=1e-18;}
  if(mode==3)f.Hf(0,0)=stored_value(UINT64_C(0x7ff8000000000001));
  if(mode==4)f.Hx(0,0)=stored_value(UINT64_C(0x7ff8000000000001));
  if(mode==5)f.r(0)=stored_value(UINT64_C(0x7ff0000000000000));
  if(mode==6)f.R=-f.R;
  if(mode==7)f.R.setZero();
  if(mode==8)f.R(0,0)=.2;
  if(mode==9)f.Hf=1e-160*Mat::Identity(3,3);
  if(mode==10){f.Hf.setIdentity();f.r(0)=1e308;auto value=f.v->value();value(0)=1e308;f.v->set_value(value);}
  if(mode==11)f.R(0,0)=stored_value(UINT64_C(0x7ff8000000000001));
  const Mat prior=StateHelper::get_full_covariance(f.s),imu=f.s->_imu->value(),value=f.v->value(),fej=f.v->fej();
  check(!augment(f.s,f.v,f.order,f.Hx,f.Hf,f.R,f.r),"invalid direct augmentation is rejected");
  check(same(prior,StateHelper::get_full_covariance(f.s))&&same(imu,f.s->_imu->value()),"failed augmentation leaves complete existing mean/covariance unchanged");
  check(f.v->id()==-1&&same(value,f.v->value())&&same(fej,f.v->fej()),"failed augmentation preserves proposed variable ownership, mean and FEJ");
}
void ownership_case(int mode, int rows) {
  Fixture f; StateOptions options; auto other=std::make_shared<State>(options);
  std::shared_ptr<State> supplied_state=f.s;
  std::shared_ptr<Type> proposed=f.v;
  if(mode==0)f.order[0]=other->_imu->bg(); // actual foreign subvariable, identical live ID
  if(mode==1){auto stale=std::make_shared<Vec>(3);stale->set_local_id(f.order[0]->id());f.order[0]=stale;}
  if(mode==2)proposed=other->_imu->p(); // must not steal ownership or mutate another state
  if(mode==3)proposed=f.s->_imu->p(); // a live subvariable is not a new top-level variable
  if(mode==4)supplied_state.reset();
  if(mode==5)proposed.reset();
  if(mode==6)f.order[0].reset();
  if(mode==7)f.Hx.conservativeResize(3,5);
  if(mode==8){auto v=f.v->fej();v(0)=stored_value(UINT64_C(0x7ff8000000000001));f.v->set_fej(v);}
  if(mode==9){check(augment(f.s,f.v,f.order,f.Hx,f.Hf,f.R,f.r),"duplicate-proposal fixture has a live prior variable");}
  if(rows>3){f.Hx.conservativeResize(rows,f.Hx.cols());f.Hx.bottomRows(rows-3).setZero();f.Hf.conservativeResize(rows,3);f.Hf.bottomRows(rows-3).setZero();f.r.conservativeResize(rows);f.r.tail(rows-3).setZero();f.R=.1*Mat::Identity(rows,rows);}
  const Mat prior=StateHelper::get_full_covariance(f.s),imu=f.s->_imu->value(),other_imu=other->_imu->value();
  const Mat value=f.v->value(),fej=f.v->fej(),Hx=f.Hx,Hf=f.Hf,R=f.R,res=f.r;
  const int id=f.v->id(), proposal_id=proposed ? proposed->id() : -1;
  const Mat proposal_value=proposed ? proposed->value() : Mat(), proposal_fej=proposed ? proposed->fej() : Mat();
  const bool accepted=rows==0 ? augment(supplied_state,proposed,f.order,f.Hx,f.Hf,f.R,f.r) :
    StateHelper::initialize(supplied_state,proposed,f.order,f.Hx,f.Hf,f.R,f.r,1.);
  check(!accepted,"invalid ownership refuses before direct or QR augmentation");
  check(same(prior,StateHelper::get_full_covariance(f.s))&&same(imu,f.s->_imu->value())&&same(other_imu,other->_imu->value()),"failure preserves both actual state owners");
  check(f.v->id()==id&&same(value,f.v->value())&&same(fej,f.v->fej())&&
        (!proposed||(proposed->id()==proposal_id&&same(proposal_value,proposed->value())&&same(proposal_fej,proposed->fej()))),"failure preserves proposed and foreign variable ID/value/FEJ");
  check(same(Hx,f.Hx)&&same(Hf,f.Hf)&&same(R,f.R)&&same(res,f.r),"invalid preflight preserves caller-owned factor before QR");
}
void qr_rank_case(int rows){
  Fixture f;Mat Hx=Mat::Zero(rows,6),Hf=Mat::Zero(rows,3),R=.1*Mat::Identity(rows,rows);Vector r=Vector::Zero(rows);
  Hf.topLeftCorner(2,2).setIdentity();
  const Mat prior=StateHelper::get_full_covariance(f.s),mean=f.v->value();
  check(!StateHelper::initialize(f.s,f.v,f.order,Hx,Hf,R,r,1.),"actual QR initializer rejects an unobservable direction even with zero residual");
  check(same(prior,StateHelper::get_full_covariance(f.s))&&f.v->id()==-1&&same(mean,f.v->value()),"rank failure before or after residual gate is atomic");
}
}
int main(int argc,char**argv){
  ov_core::Printer::setPrintLevel("ERROR");const std::string selected=argc>1?argv[1]:"all";
  if(selected=="all"||selected=="valid")for(double s:{.1,1.,10.})valid_case(s);
  for(int i=0;i<12;++i)if(selected=="all"||selected=="invalid"+std::to_string(i))invalid_case(i);
  for(int n:{3,6})if(selected=="all"||selected=="qr"+std::to_string(n))qr_rank_case(n);
  for(int mode=0;mode<10;++mode)for(int rows:{0,3,6})if(selected=="all"||selected=="owner"+std::to_string(mode)+"_"+std::to_string(rows))ownership_case(mode,rows);
  check(checks>0,"requested case exists");std::printf("DELAYED_INIT_ATOMIC %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);
  return failures?1:0;
}
