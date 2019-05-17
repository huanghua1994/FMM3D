#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "lfmm3d_c.h"
#include "complex.h"
#include "cprini.h"

int main(int argc, char **argv)
{
  cprin_init("stdout", "fort.13");
  int ns=2000;
  int nt=1999;
  double *source = (double *)malloc(3*ns*sizeof(double));
  double *targ = (double *)malloc(3*nt*sizeof(double));

  double *charge = (double *)malloc(ns*sizeof(double));
  double *dipvec = (double *)malloc(3*ns*sizeof(double));

  int ntest = 10;

  int ntests = 36;
  int ipass[36];

  for(int i=0;i<ntests;i++)
  {
    ipass[i] = 0;
  }


  double thresh = 1e-15;

  double err=0;

  double *pot = (double *)malloc(ns*sizeof(double));
  double *potex = (double *)malloc(ntest*sizeof(double));
  double *grad = (double *)malloc(3*ns*sizeof(double));
  double *gradex = (double *)malloc(3*ntest*sizeof(double));

  double *pottarg = (double *)malloc(nt*sizeof(double));
  double *pottargex = (double *)malloc(ntest*sizeof(double));
  double *gradtarg = (double *)malloc(3*nt*sizeof(double));
  double *gradtargex = (double *)malloc(3*ntest*sizeof(double));

  int nd = 1;

  int pg = 0;
  int pgt = 0;



  for(int i=0;i<ns;i++)
  {
    source[3*i] = pow(rand01(),2);
    source[3*i+1] = pow(rand01(),2);
    source[3*i+2] = pow(rand01(),2);

    charge[i] = rand01() + I*rand01();

    dipvec[3*i] = rand01() +I*rand01();
    dipvec[3*i+1] = rand01() + I*rand01();
    dipvec[3*i+2] = rand01() + I*rand01();

  }

  for(int i=0; i<nt; i++)
  {
    targ[3*i] = rand01();
    targ[3*i+1] = rand01();
    targ[3*i+2] = rand01();

  }


  double eps = 0.5e-6;


  int itest = 0;
  cprin_message("testing source to source");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 1;
  pgt = 0;

  lfmm3dpartstoscp_(&eps, &ns, source, charge, pot);

  dzero(ntest,potex);
  l3ddirectcp_(&nd, source, charge, &ns, source, &ntest, potex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 2;
  pgt = 0;
  lfmm3dpartstoscg_(&eps, &ns, source, charge, pot, grad);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  l3ddirectcg_(&nd, source, charge, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 1;
  pgt = 0;
  lfmm3dpartstosdp_(&eps, &ns, source, dipvec, pot);

  dzero(ntest,potex);
  l3ddirectdp_(&nd, source, dipvec, &ns, source, &ntest, potex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 0;
  lfmm3dpartstosdg_(&eps, &ns, source, dipvec, pot, grad);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  l3ddirectdg_(&nd, source, dipvec, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 1;
  pgt = 0;
  lfmm3dpartstoscdp_(&eps, &ns, source, charge, dipvec, pot);

  dzero(ntest,potex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, source, &ntest, potex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 0;
  lfmm3dpartstoscdg_(&eps, &ns, source, charge, dipvec, pot, grad);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 0;
  pgt = 1;

  lfmm3dpartstotcp_(&eps, &ns, source, charge, &nt, targ, pottarg);

  dzero(ntest,pottargex);
  l3ddirectcp_(&nd, source, charge, &ns, targ, &ntest, 
       pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 0;
  pgt = 2;
  lfmm3dpartstotcg_(&eps, &ns, source, charge, &nt, targ, 
      pottarg, gradtarg);

  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectcg_(&nd, source, charge, &ns, targ, &ntest, pottargex, 
     gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 0;
  pgt = 1;
  lfmm3dpartstotdp_(&eps, &ns, source, dipvec, &nt, targ, pottarg);

  dzero(ntest,pottargex);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 0;
  pgt = 2;
  lfmm3dpartstotdg_(&eps, &ns, source, dipvec, &nt, targ, 
     pottarg, gradtarg);

  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectdg_(&nd, source, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 0;
  pgt = 1;
  lfmm3dpartstotcdp_(&eps, &ns, source, charge, dipvec, &nt, targ,
     pottarg);

  dzero(ntest,pottargex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 0;
  pgt = 2;
  lfmm3dpartstotcdg_(&eps, &ns, source, charge, dipvec, &nt, 
     targ, pottarg, gradtarg);

  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 1;
  pgt = 1;

  lfmm3dpartstostcp_(&eps, &ns, source, charge, pot, 
    &nt, targ, pottarg);

  dzero(ntest,potex);
  dzero(ntest,pottargex);
  l3ddirectcp_(&nd, source, charge, &ns, source, &ntest, 
       potex, &thresh);
  l3ddirectcp_(&nd, source, charge, &ns, targ, &ntest, 
       pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 2;
  pgt = 2;
  lfmm3dpartstostcg_(&eps, &ns, source, charge, pot, grad, 
      &nt, targ, pottarg, gradtarg);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectcg_(&nd, source, charge, &ns, source, &ntest, potex, 
     gradex, &thresh);
  l3ddirectcg_(&nd, source, charge, &ns, targ, &ntest, pottargex, 
     gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 1;
  pgt = 1;
  lfmm3dpartstostdp_(&eps, &ns, source, dipvec, pot, 
     &nt, targ, pottarg);

  dzero(ntest,potex);
  dzero(ntest,pottargex);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     source, &ntest, potex, &thresh);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 2;
  lfmm3dpartstostdg_(&eps, &ns, source, dipvec, pot, grad, 
     &nt, targ, pottarg, gradtarg);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectdg_(&nd, source, dipvec, &ns, source, &ntest, 
     potex, gradex, &thresh);
  l3ddirectdg_(&nd, source, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 1;
  pgt = 1;
  lfmm3dpartstostcdp_(&eps, &ns, source, charge, dipvec, pot, 
     &nt, targ,  pottarg);

  dzero(ntest,potex);
  dzero(ntest,pottargex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     source, &ntest, potex, &thresh);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 2;
  lfmm3dpartstostcdg_(&eps, &ns, source, charge, dipvec, pot, grad,
     &nt,targ, pottarg, gradtarg);

  dzero(ntest,potex);
  dzero(3*ntest,gradex);
  dzero(ntest,pottargex);
  dzero(3*ntest,gradtargex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, source, &ntest, 
     potex, gradex, &thresh);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  free(charge);
  free(dipvec);
  free(pot);
  free(potex);
  free(grad);
  free(gradex);
  free(pottarg);
  free(pottargex);
  free(gradtarg);
  free(gradtargex);

  nd = 2;
  charge = (double *)malloc(ns*nd*sizeof(double));
  dipvec = (double *)malloc(3*ns*nd*sizeof(double));

  pot = (double *)malloc(ns*nd*sizeof(double));
  potex = (double *)malloc(ntest*nd*sizeof(double));
  grad = (double *)malloc(3*ns*nd*sizeof(double));
  gradex = (double *)malloc(3*ntest*nd*sizeof(double));

  pottarg = (double *)malloc(nt*nd*sizeof(double));
  pottargex = (double *)malloc(ntest*nd*sizeof(double));
  gradtarg = (double *)malloc(3*nt*nd*sizeof(double));
  gradtargex = (double *)malloc(3*ntest*nd*sizeof(double));



  for(int i=0;i<nd*ns;i++)
  {

    charge[i] = rand01() + I*rand01();

    dipvec[3*i] = rand01() +I*rand01();
    dipvec[3*i+1] = rand01() + I*rand01();
    dipvec[3*i+2] = rand01() + I*rand01();

  }


  itest += 1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 1;
  pgt = 0;

  lfmm3dpartstoscp_vec_(&nd, &eps, &ns, source, charge, pot);

  dzero(nd*ntest,potex);
  l3ddirectcp_(&nd, source, charge, &ns, source, &ntest, potex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 2;
  pgt = 0;
  lfmm3dpartstoscg_vec_(&nd, &eps, &ns, source, charge, pot, grad);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  l3ddirectcg_(&nd, source, charge, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 1;
  pgt = 0;
  lfmm3dpartstosdp_vec_(&nd, &eps, &ns, source, dipvec, pot);

  dzero(nd*ntest,potex);
  l3ddirectdp_(&nd, source, dipvec, &ns, source, &ntest, potex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 0;
  lfmm3dpartstosdg_vec_(&nd, &eps, &ns, source, dipvec, pot, grad);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  l3ddirectdg_(&nd, source, dipvec, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 1;
  pgt = 0;
  lfmm3dpartstoscdp_vec_(&nd, &eps, &ns, source, charge, dipvec, pot);

  dzero(nd*ntest,potex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, source, &ntest, potex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 0;
  lfmm3dpartstoscdg_vec_(&nd, &eps, &ns, source, charge, dipvec, pot, grad);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, source, &ntest, potex, gradex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 0;
  pgt = 1;

  lfmm3dpartstotcp_vec_(&nd, &eps, &ns, source, charge, &nt, targ, pottarg);

  dzero(nd*ntest,pottargex);
  l3ddirectcp_(&nd, source, charge, &ns, targ, &ntest, 
       pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 0;
  pgt = 2;
  lfmm3dpartstotcg_vec_(&nd, &eps, &ns, source, charge, &nt, targ, 
      pottarg, gradtarg);

  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectcg_(&nd, source, charge, &ns, targ, &ntest, pottargex, 
     gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 0;
  pgt = 1;
  lfmm3dpartstotdp_vec_(&nd, &eps, &ns, source, dipvec, &nt, targ, pottarg);

  dzero(nd*ntest,pottargex);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 0;
  pgt = 2;
  lfmm3dpartstotdg_vec_(&nd, &eps, &ns, source, dipvec, &nt, targ, 
     pottarg, gradtarg);

  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectdg_(&nd, source, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 0;
  pgt = 1;
  lfmm3dpartstotcdp_vec_(&nd, &eps, &ns, source, charge, dipvec, &nt, targ,
     pottarg);

  dzero(nd*ntest,pottargex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 0;
  pgt = 2;
  lfmm3dpartstotcdg_vec_(&nd, &eps, &ns, source, charge, dipvec, &nt, 
     targ, pottarg, gradtarg);

  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges");
  cprin_message("output: potentials");
  cprin_skipline(2);


  pg = 1;
  pgt = 1;

  lfmm3dpartstostcp_vec_(&nd, &eps, &ns, source, charge, pot, 
    &nt, targ, pottarg);

  dzero(nd*ntest,potex);
  dzero(nd*ntest,pottargex);
  l3ddirectcp_(&nd, source, charge, &ns, source, &ntest, 
       potex, &thresh);
  l3ddirectcp_(&nd, source, charge, &ns, targ, &ntest, 
       pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest += 1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges");
  cprin_message("output: gradients");
  cprin_skipline(2);
  


  pg = 2;
  pgt = 2;
  lfmm3dpartstostcg_vec_(&nd, &eps, &ns, source, charge, pot, grad, 
      &nt, targ, pottarg, gradtarg);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectcg_(&nd, source, charge, &ns, source, &ntest, potex, 
     gradex, &thresh);
  l3ddirectcg_(&nd, source, charge, &ns, targ, &ntest, pottargex, 
     gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);


  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");



  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);
  

  pg = 1;
  pgt = 1;
  lfmm3dpartstostdp_vec_(&nd, &eps, &ns, source, dipvec, pot, 
     &nt, targ, pottarg);

  dzero(nd*ntest,potex);
  dzero(nd*ntest,pottargex);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     source, &ntest, potex, &thresh);
  l3ddirectdp_(&nd, source, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 2;
  lfmm3dpartstostdg_vec_(&nd, &eps, &ns, source, dipvec, pot, grad, 
     &nt, targ, pottarg, gradtarg);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectdg_(&nd, source, dipvec, &ns, source, &ntest, 
     potex, gradex, &thresh);
  l3ddirectdg_(&nd, source, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  
  itest +=1;
  cprin_message("testing source to target");
  cprin_message("interaction: charges+dipoles");
  cprin_message("output: potentials");
  cprin_skipline(2);

  pg = 1;
  pgt = 1;
  lfmm3dpartstostcdp_vec_(&nd, &eps, &ns, source, charge, dipvec, pot, 
     &nt, targ,  pottarg);

  dzero(nd*ntest,potex);
  dzero(nd*ntest,pottargex);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     source, &ntest, potex, &thresh);
  l3ddirectcdp_(&nd, source, charge, dipvec, &ns, 
     targ, &ntest, pottargex, &thresh);

  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }


  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  itest +=1;
  cprin_message("testing source to source+target");
  cprin_message("interaction: charges + dipoles");
  cprin_message("output: gradients");
  cprin_skipline(2);


  pg = 2;
  pgt = 2;
  lfmm3dpartstostcdg_vec_(&nd, &eps, &ns, source, charge, dipvec, pot, grad,
     &nt,targ, pottarg, gradtarg);

  dzero(nd*ntest,potex);
  dzero(nd*3*ntest,gradex);
  dzero(nd*ntest,pottargex);
  dzero(nd*3*ntest,gradtargex);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, source, &ntest, 
     potex, gradex, &thresh);
  l3ddirectcdg_(&nd, source, charge, dipvec, &ns, targ, &ntest, 
     pottargex, gradtargex, &thresh);
  comp_err_lap(nd*ntest,pg,pgt,pot,potex,pottarg,pottargex,grad,gradex,
       gradtarg,gradtargex,&err);

  if(err<eps)
  {
    ipass[itest] = 1;
  }

  cprind("l2 rel error=",&err,1);

  cprin_skipline(2);
  cprin_message("========");


  free(charge);
  free(dipvec);
  free(pot);
  free(potex);
  free(grad);
  free(gradex);
  free(pottarg);
  free(pottargex);
  free(gradtarg);
  free(gradtargex);


  int isum = 0;
  for(int i=0;i<ntests;i++)
  {
    isum = isum + ipass[i];
  }

  cprinf("Number of tests out of 36 passed =",&isum,1);

  return 0;
}  
