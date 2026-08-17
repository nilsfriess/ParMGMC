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
  PoissonGibbsCtx* ctx_fine;   // context associated with fine PC
  PoissonGibbsCtx* ctx_coarse; // context associated with coarse PC 
  PetscScalar drop_tolerance = 1.E-12;

  PetscFunctionBeginUser;
  PetscCall(PCGetApplicationContext(pc_fine, &ctx_fine));
  PetscCall(PetscNew(&ctx_coarse));
  // Create and copy event counts
  PetscCall(VecDuplicate(ctx_fine->event_counts, &ctx_coarse->event_counts));
  PetscCall(VecCopy(ctx_fine->event_counts, ctx_coarse->event_counts));
  // Create coarse level offset vector nu
  PetscCall(VecDuplicate(ctx_fine->nu, &ctx_coarse->nu));
  // Create coarse matrix B_c = P.B
  PetscCall(MatTransposeMatMult(Ip, ctx_fine->B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ctx_coarse->B));
  PetscCall(MatFilter(ctx_coarse->B, drop_tolerance, PETSC_TRUE, PETSC_FALSE));
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
  PetscCall(PetscFree(ctx));
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
  (void) res; 
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
  PetscInt nnz;
  const PetscInt* row_ptr;
  PetscBool done;

  PetscFunctionBeginUser;
  // Work out maximum number of entries per row for A and B
  *max_nnz_per_row = 0;
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i=1; i<=nnz; ++i) {
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
static PetscScalar grad_phi(const PetscScalar theta,
                            const PetscScalar mu_bar,
                            const PetscScalar sigma,
                            const PetscInt n_k,
                            const PetscScalar* nu,
                            const PetscScalar* b) {
  PetscScalar g = (theta - mu_bar)/(sigma*sigma);
  for (PetscInt k=0;k<n_k;++k) {
    g += b[k]*exp(b[k]*theta+nu[k]);
  }
  return g;
}

static PetscErrorCode PCPoissonGibbsFindMaximum(const PetscScalar mu_bar,
                                                const PetscScalar sigma,
                                                const PetscInt n_k,
                                                const PetscScalar* nu,
                                                const PetscScalar* b,
                                                PetscScalar* theta_bar) {
  PetscScalar theta, theta_old, theta_left, theta_right, theta_mid;
  PetscScalar g, g_old, g_left, g_mid;
  PetscScalar bracketing_tolerance = 1.E-12;
  PetscScalar bisection_tolerance = 1.E-10;
  PetscScalar delta;
         
  PetscFunctionBeginUser;
  theta = mu_bar;
  g = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
  theta_old = theta;
  g_old = g;
  // Bracket minimum
  delta = sigma;
  while (true) {
    if (g > 0) {
      theta -= delta;
    } else {
      theta += delta;
    }
    delta*=2;
    g = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
    // stop when derivative changes sign
    if ( ((g > 0) && (g_old < 0)) || ((g < 0) && (g_old > 0)) ) break;
    if (fabs(g) < bracketing_tolerance) break;
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

  while ((theta_right-theta_left)/fmax(fabs(theta_right),fabs(theta_left)) > bisection_tolerance) {
    theta_mid = 0.5*(theta_left+theta_right);
    g_mid = grad_phi(theta_mid, mu_bar, sigma, n_k, nu, b);
    if ( ((g_left > 0) && (g_mid > 0)) || ((g_left < 0) && (g_mid < 0)) ) {
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
  PetscInt rstart, rend, ncols_A, ncols_B, max_nnz_per_row;
  const PetscInt *cols_A;
  const PetscScalar *vals_A;
  const PetscInt *cols_B;
  const PetscScalar *vals_B;
  PetscScalar sigma;
  PetscScalar mu_bar;
  PetscScalar* theta;
  PetscScalar* n_local;
  PetscScalar* nu_local;
  PetscScalar theta_bar;  
  Vec v_diag, nu_tilde;
  const PetscScalar* diag;
  const PetscScalar* f_rhs;
  PetscScalar r, theta_prime;
  PoissonGibbsCtx* ctx;  

  PetscFunctionBeginUser;  
  PetscCall(PCGetApplicationContext(pc, &ctx));
  Mat A = pc->pmat;
  Mat B = ctx->B;

  PetscCall(VecDuplicate(ctx->nu, &nu_tilde));
  PetscCall(MatMultTransposeAdd(ctx->B, y, ctx->nu, nu_tilde));
  
  // Storage for local part of vectors
  PetscCall(PCPoissonGetMaxNnzPerRow(A, &max_nnz_per_row));
  PetscCall(PCPoissonGetMaxNnzPerRow(B, &max_nnz_per_row));
  PetscCall(PetscMalloc1(max_nnz_per_row, &n_local));
  PetscCall(PetscMalloc1(max_nnz_per_row, &nu_local));
  
  PetscCall(VecDuplicate(y, &v_diag));
  PetscCall(MatGetDiagonal(A, v_diag));
  PetscCall(VecGetArrayRead(v_diag, &diag));
  PetscCall(VecGetArrayRead(b, &f_rhs));
  PetscCall(VecGetArray(y,&theta));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i=rstart; i<rend; ++i) {
    PetscInt iloc = i - rstart;
    sigma = 1./sqrt(diag[iloc]);
    PetscCall(MatGetRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatGetRow(B, i, &ncols_B, &cols_B, &vals_B));
    PetscCall(VecGetValues(ctx->event_counts, ncols_B, cols_B, n_local));
    PetscCall(VecGetValues(nu_tilde, ncols_B, cols_B, nu_local));
    for (PetscInt k=0; k<ncols_B; ++k) {
      nu_local[k] -= theta[iloc]*vals_B[k];
    }
    mu_bar = f_rhs[iloc];
    for (PetscInt j=0; j<ncols_A; ++j) {
      if (cols_A[j] != i)
        mu_bar -= vals_A[j]*theta[cols_A[j]-rstart];
    }
    for (PetscInt j=0; j<ncols_B; ++j) {
      mu_bar += vals_B[j]*n_local[j];
    }
    mu_bar *= sigma*sigma;
    if (ncols_B > 0) {
      // sample with rejection sampling
      PetscCall(PCPoissonGibbsFindMaximum(mu_bar, sigma, ncols_B, nu_local, vals_B, &theta_bar));
      PetscBool accepted = PETSC_FALSE;
      while (!accepted) {
        PetscCall(PCPoissonGibbsStandardNormal(pc, &r));
        theta_prime = theta_bar + sigma*r;
        PetscCall(PCPoissonGibbsUniform(pc, &r));
        PetscScalar Fbar = 0;
        for (PetscInt k=0; k<ncols_B; ++k) {
          Fbar += exp(vals_B[k]*theta_prime+nu_local[k]) + ((theta_bar - theta_prime)*vals_B[k]-1.0)*exp(vals_B[k]*theta_bar+nu_local[k]);
        }
        if (isnan(Fbar) || isinf(Fbar)) {
          return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_FP, PETSC_ERROR_INITIAL, "Encountered invalid Fbar value (NaN or Inf) in Poisson-Gibbs rejection step");
        }
        //if (Fbar < 0) printf("Fbar < 0: %e\n",Fbar);
        accepted = (log(r) <= -Fbar);
      }
    } else {
      // just draw a Gaussian random variable with mean mu_bar and width sigma
      PetscCall(PCPoissonGibbsStandardNormal(pc, &r));
      theta_prime = mu_bar + sigma*r;
    }
    theta[iloc] = theta_prime;
    for (PetscInt k=0; k<ncols_B; ++k) {
      nu_local[k] += theta[iloc]*vals_B[k];
    }
    PetscCall(VecSetValues(nu_tilde, ncols_B, cols_B, nu_local, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(nu_tilde));
    PetscCall(VecAssemblyEnd(nu_tilde));
    PetscCall(MatRestoreRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatRestoreRow(B, i, &ncols_B, &cols_B, &vals_B));
  }
  PetscCall(VecRestoreArrayRead(v_diag, &diag));
  PetscCall(VecRestoreArrayRead(b, &f_rhs));
  PetscCall(VecRestoreArray(y, &theta));
  PetscCall(PetscFree(n_local));
  PetscCall(PetscFree(nu_local));
  PetscCall(VecDestroy(&v_diag));
  PetscCall(VecDestroy(&nu_tilde));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PoissonGibbs(PC pc, Vec b, Vec y)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscCall(PCPoissonGibbsSample(pc, b, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_PoissonGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  (void)w;
  (void)rtol;
  (void)abstol;
  (void)dtol;
  poissongibbs->sample_index = 0;
  if (guesszero) PetscCall(VecZeroEntries(y));
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
  (void)poissongibbs;
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
  (void)poissongibbs;
  (void)viewer;
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
