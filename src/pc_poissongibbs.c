/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.

    Non-linear Gibbs sampler for posterior obtained by contitioning a Gaussian prior
    on a Poisson process.
*/

#include "parmgmc/pc/pc_poissongibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

typedef struct {
  PetscRandom prand; // Random numbers
  Vec random_workspace;
  PetscInt random_work_ptr;
  PetscInt  sample_index;
  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
}  *PC_PoissonGibbs;

#define RANDOM_BUFFER_SIZE 64

/* Coarsen user context
 * 
 *   pc_fine: preconditioner on fine level
 *   Ip: interpolation matrix
 *   pc_coarse: preconditioner on coarse level
 */
static PetscErrorCode PoissonGibbsCoarsenCtxImpl(PC pc_fine, Mat Ip, PC pc_coarse) {
  PetscFunctionBeginUser;
  PoissonGibbsCtx* ctx_fine;   // context associated with fine PC
  PoissonGibbsCtx* ctx_coarse; // context associated with coarse PC 
  PetscCall(PCGetApplicationContext(pc_fine, &ctx_fine));
  ctx_coarse = (PoissonGibbsCtx*)malloc(sizeof(PoissonGibbsCtx));
  // Create and copy event counts
  PetscCall(VecDuplicate(ctx_fine->event_counts,&ctx_coarse->event_counts));
  PetscCall(VecCopy(ctx_fine->event_counts,ctx_coarse->event_counts));
  // Create coarse level offset vector nu
  PetscCall(VecDuplicate(ctx_fine->nu,&ctx_coarse->nu));
  // Create coarse matrix B_c = P.B
  PetscCall(MatTransposeMatMult(Ip, ctx_fine->B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ctx_coarse->B));
  PetscCall(PCSetApplicationContext(pc_coarse, ctx_coarse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Destroy user context
 * 
 *   ctx: application context
 */
static PetscErrorCode PoissonGibbsDestroyCtxImpl(PetscCtx ctx) {
  PetscFunctionBeginUser;
  PoissonGibbsCtx* poissonctx = (PoissonGibbsCtx*)ctx;
  PetscCall(VecDestroy(&poissonctx->event_counts));
  PetscCall(VecDestroy(&poissonctx->nu));
  PetscCall(MatDestroy(&poissonctx->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Update coarse level nu after each solve
 *
 * Sets nu_c = nu + B^T.theta, this routine is added as a post-solve hook
 *
 *    ksp: solver object   
 *    x: current state theta
 *    res: current right hand side (ignored)
 *    ctx: coarse level user context
 */
static PetscErrorCode PoissonGibbsPostSolveImpl(KSP ksp, Vec x, Vec res, PetscCtx ctx) {
  PetscFunctionBeginUser;
  PoissonGibbsCtx* ctx_fine; // (fine level) context
  PC pc; // (fine level) PC object
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetApplicationContext(pc, &ctx_fine));
  // Compute nu_c = nu + B^T.theta
  PoissonGibbsCtx* ctx_coarse = (PoissonGibbsCtx*)ctx;
  PetscCall(MatMultTransposeAdd(ctx_fine->B, x, ctx_fine->nu, ctx_coarse->nu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPoissonGetMaxNnzPerRow(Mat mat, PetscInt *max_nnz_per_row) {
  PetscInt nnz, nnz_per_row;
  PetscInt* row_ptr;
  PetscBool done;

  PetscFunctionBeginUser;
  // Work out maximum number of entries per row for A and B
  *max_nnz_per_row = 0;
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i=1; i<nnz; ++i) {
    *max_nnz_per_row = PetscMax(*max_nnz_per_row,row_ptr[i]-row_ptr[i-1]);
  }
  PetscCall(MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPoissonGibbsStandardNormal(PC pc, PetscScalar *r) {
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;  
  if (poissongibbs->random_work_ptr >= RANDOM_BUFFER_SIZE) {
    PetscCall(VecSetRandomStandardNormal(poissongibbs->random_workspace, poissongibbs->prand));
    poissongibbs->random_work_ptr=0;
  }
  PetscCall(VecGetValues(poissongibbs->random_workspace, 1, &poissongibbs->random_work_ptr, r));
  poissongibbs->random_work_ptr++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPoissonGibbsUniform(PC pc, PetscScalar *r) {
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;  
  PetscCall(PetscRandomGetValueReal(poissongibbs->prand, r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient dphi/dtheta(theta) 

 * dphi/dtheta = sum_{k} B_{ik} exp(B_{ik} theta + nu_k ) +( theta - bar(mu))/sigma^2
 */
static PetscScalar grad_phi(PetscScalar theta,
                            PetscScalar mu_bar,
                            PetscScalar sigma,
                            PetscInt n_k,
                            PetscScalar* nu,
                            PetscScalar* b) {
  PetscScalar g = (theta - mu_bar)/(sigma*sigma);
  for (PetscInt k=0;k<n_k;++k) {
    g += b[k]*exp(b[k]*theta+nu[k]);
  }
  return g;
}

/* bar(F) */
static PetscScalar Fbar(PetscScalar theta,
                        PetscScalar theta_bar,
                        PetscScalar sigma,
                        PetscInt n_k,
                        PetscScalar* nu,
                        PetscScalar* b) {
  PetscScalar f = 0;
  for (PetscInt k=0;k<n_k;++k) 
    f += exp(b[k]*theta+nu[k]) + ((theta_bar - theta)*b[k]-1.0)*exp(b[k]*theta_bar+nu[k]);
  return f;
}

static PetscErrorCode PCPoissonGibbsFindMaximum(PetscScalar mu_bar,
                                                PetscScalar sigma,
                                                PetscInt n_k,
                                                PetscScalar* nu,
                                                PetscScalar* b,
                                                PetscScalar* theta_bar) {
  PetscScalar theta, theta_old, theta_left, theta_right, theta_mid;
  PetscScalar g, g_old, g_left, g_mid;
  PetscScalar bisection_tolerance = 1.E-12;
                                                  
  PetscFunctionBeginUser;
  g = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
  theta_old = theta;
  g_old = g;
  // Bracket minimum
  while (true) {
    if (g > 0) {
      theta -= sigma;
    } else {
      theta += sigma;
    }
    g = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
    // stop when derivative changes sign
    if ( ((g > 0) && (g_old < 0)) || ((g < 0) && (g_old > 0)) ) break;
    theta_old = theta;
    g_old = g;
  }          
  // Bisection of minimum
  if (theta_old < theta) {
    theta_left = theta_old;
    theta_right = theta;
    g_left = g_old;
  } else {
    theta_left = theta;
    theta_right = theta_old;
    g_left = g;
  }

  while (theta_right-theta_left > bisection_tolerance) {
    theta_mid = 0.5*(theta_left+theta_right);
    g_mid = grad_phi(theta_mid, mu_bar, sigma, n_k, nu, b);
    if ( ((g_left > 0) && (g_mid > 0)) || (g_left < 0) && (g_mid < 0) ) {
      theta_left = theta_mid;
      g_left = g_mid;
    } else {
      theta_right = theta_mid;
    }
  }
  *theta_bar = 0.5*(theta_right+theta_left);
  PetscFunctionReturn(PETSC_SUCCESS);                                         
}

/* Generate a new sample (computational routine) */
static PetscErrorCode PCPoissonGibbsSample(PC pc, Vec b, Vec y)
{
  PetscInt nrow, ncol, ncols_A, ncols_B, max_nnz_per_row;
  PetscInt *cols_A;
  PetscScalar *vals_A;
  PetscInt *cols_B;
  PetscScalar *vals_B;
  PetscScalar sigma;
  PetscScalar mu_bar;
  PetscScalar* y_local;
  PetscScalar* n_local;
  PetscScalar* nu_local;
  PetscScalar theta_bar;
  Vec w;
  Vec v_diag;
  PetscScalar* diag;
  PetscScalar r, y_new;
  PoissonGibbsCtx* ctx;
  PC_PoissonGibbs poissongibbs = pc->data;
  PetscInt random_idx;
  PetscScalar random_val;

  PetscFunctionBeginUser;  
  PetscCall(PCGetApplicationContext(pc, &ctx));
  Mat A = pc->pmat;
  Mat B = ctx->B;

  // Storage for local part of vectors
  PetscCall(PCPoissonGetMaxNnzPerRow(A, &max_nnz_per_row));
  PetscCall(PetscMalloc1(max_nnz_per_row, &y_local));
  PetscCall(PCPoissonGetMaxNnzPerRow(B, &max_nnz_per_row));
  PetscCall(PetscMalloc1(max_nnz_per_row, &n_local));
  PetscCall(PetscMalloc1(max_nnz_per_row, &nu_local));
  
  PetscCall(VecDuplicate(y, &v_diag));
  PetscCall(MatGetDiagonal(A, v_diag));
  PetscCall(VecGetArray(v_diag, &diag));
  PetscCall(MatGetSize(A, &nrow, &ncol));
  for (PetscInt i=0; i<nrow; ++i) {
    sigma = 1./sqrt(diag[i]);
    PetscCall(MatGetRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatGetRow(B, i, &ncols_B, &cols_B, &vals_B));
    PetscCall(VecGetValues(b, 1, &i, &mu_bar));
    PetscCall(VecGetValues(y, ncols_A, cols_A, y_local));
    for (PetscInt j=0; j<ncols_A; ++j) {
      mu_bar -= vals_A[j]*y_local[j];
    }
    PetscCall(VecGetValues(ctx->event_counts, ncols_B, cols_B, n_local));
    PetscCall(VecGetValues(ctx->nu, ncols_B, cols_B, nu_local));
    for (PetscInt j=0; j<ncols_B; ++j) {
      mu_bar += vals_B[j]*n_local[j];
    }
    mu_bar *= sigma*sigma;
    if (ncols_B > 0) {
      // sample with rejection sampling
      PetscCall(PCPoissonGibbsFindMaximum(mu_bar, sigma, ncols_B, nu_local, vals_B, &theta_bar));
      PetscBool accepted = PETSC_FALSE;
      while (!accepted) {
        PCPoissonGibbsStandardNormal(pc, &r);
        y_new = theta_bar + sigma*r;
        PCPoissonGibbsUniform(pc, &r);
        accepted = (log(r) <= -Fbar(y_new, theta_bar, sigma, ncols_B, nu_local, vals_B));
      }
    } else {
      // just draw a Gaussian random variable with mean mu_bar and width sigma
      PCPoissonGibbsStandardNormal(pc, &r);
      y_new = mu_bar + sigma*r;
    }
    PetscCall(VecSetValue(y, i, y_new, INSERT_VALUES));
    PetscCall(MatRestoreRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatRestoreRow(B, i, &ncols_B, &cols_B, &vals_B));
  }
  PetscCall(VecRestoreArray(v_diag, &diag));
  PetscCall(PetscFree(y_local));
  PetscCall(PetscFree(n_local));
  PetscCall(PetscFree(nu_local));
  PetscCall(VecDestroy(&v_diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PoissonGibbs(PC pc, Vec b, Vec y)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscCall(PCPoissonGibbsSample(pc, b, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_PoissonGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  poissongibbs->sample_index = 0;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(PCPoissonGibbsSample(pc, b, y));
  }

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_PoissonGibbs(PC pc)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&poissongibbs->prand));
  if (poissongibbs->del_scb) {
    PetscCall(poissongibbs->del_scb(poissongibbs->cbctx));
    poissongibbs->del_scb = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_PoissonGibbs(PC pc)
{
  PC_PoissonGibbs poissongibbs = pc->data;
  PoissonGibbsCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&poissongibbs->prand));
  if (poissongibbs->del_scb) {
    PetscCall(poissongibbs->del_scb(poissongibbs->cbctx));
    poissongibbs->del_scb = NULL;
  }
  PetscCall(VecDestroy(&poissongibbs->random_workspace));
  PetscCall(PetscFree(poissongibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_PoissonGibbs(PC pc)
{
  PC_PoissonGibbs poissongibbs = pc->data;
  
  PetscFunctionBeginUser;
  if (!poissongibbs->prand) PetscCall(ParMGMCGetPetscRandom(&poissongibbs->prand));
  PetscCall(VecCreate(MPI_COMM_WORLD, &poissongibbs->random_workspace));
  PetscCall(VecSetSizes(poissongibbs->random_workspace, RANDOM_BUFFER_SIZE, PETSC_DETERMINE));
  PetscCall(VecSetType(poissongibbs->random_workspace, VECSEQ));
  poissongibbs->random_work_ptr = RANDOM_BUFFER_SIZE;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPostSolve_C", (void (*)())PoissonGibbsPostSolveImpl));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCDestroyContext_C", (void (*)())PoissonGibbsDestroyCtxImpl));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCCoarsenContext_C", (void (*)())PoissonGibbsCoarsenCtxImpl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_PoissonGibbs(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_PoissonGibbs poissongibbs = pc->data;
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson Gibbs options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetSampleCallback_PoissonGibbs(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  if (poissongibbs->del_scb) {
    PetscCall(poissongibbs->del_scb(poissongibbs->cbctx));
    poissongibbs->del_scb = NULL;
  }
  poissongibbs->scb     = cb;
  poissongibbs->cbctx   = ctx;
  poissongibbs->del_scb = deleter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_PoissonGibbs(PC pc, PetscViewer viewer)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_PoissonGibbs(PC pc)
{
  PC_PoissonGibbs poissongibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissongibbs));
  pc->data       = poissongibbs;
  
  pc->ops->apply           = PCApply_PoissonGibbs;
  pc->ops->applyrichardson = PCApplyRichardson_PoissonGibbs;
  pc->ops->destroy         = PCDestroy_PoissonGibbs;
  pc->ops->reset           = PCReset_PoissonGibbs;
  pc->ops->setup           = PCSetUp_PoissonGibbs;
  pc->ops->setfromoptions  = PCSetFromOptions_PoissonGibbs;
  pc->ops->view            = PCView_PoissonGibbs;
  PetscCall(PCRegisterSetSampleCallback(pc, PCSetSampleCallback_PoissonGibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
