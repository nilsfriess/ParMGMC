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
  PetscInt  sample_index;
  void *cbctx;
  PetscErrorCode (*scb)(PetscInt, Vec, void *);
  PetscErrorCode (*del_scb)(void *);
}  *PC_PoissonGibbs;

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
  PetscCall(VecCopy(ctx_coarse->event_counts,ctx_coarse->event_counts));
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

/* Generate a new sample (computational routine) */
static PetscErrorCode PCPoissonGibbsSample(PC pc, Vec b, Vec y, Vec w)
{
  PetscInt nrow, ncol, ncols_A, ncols_B, nnz, nnz_per_row_A, nnz_per_row_B;
  PetscInt *cols_A;
  PetscScalar *vals_A;
  PetscInt *cols_B;
  PetscScalar *vals_B;
  PetscScalar A_ii;
  PetscScalar sigma;
  PetscScalar mu_bar;
  PetscScalar* y_local;
  PetscScalar* n_local;
  PetscInt* row_ptr;
  PetscBool done;
  Vec diag;
  PoissonGibbsCtx* ctx;

  PetscFunctionBeginUser;
  PC_PoissonGibbs poissongibbs = pc->data;
  PetscCall(PCGetApplicationContext(pc, &ctx));
  Mat A = pc->pmat;
  Mat B = ctx->B;

  // Work out maximum number of entries per row for A and B
  nnz_per_row_A = 0;
  PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i=1; i<nnz; ++i) {
    nnz_per_row_A = PetscMax(nnz_per_row_A,row_ptr[i]-row_ptr[i-1]);
  }
  PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  nnz_per_row_B = 0;
  PetscCall(MatGetRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i=1; i<nnz; ++i) {
    nnz_per_row_B = PetscMax(nnz_per_row_B,row_ptr[i]-row_ptr[i-1]);
  }
  PetscCall(MatRestoreRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  PetscCall(PetscMalloc1(nnz_per_row_A, &y_local));
  PetscCall(PetscMalloc1(nnz_per_row_B, &n_local));

  Vec event_counts = ctx->event_counts;
  PetscCall(VecDuplicate(y, &diag));
  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(MatGetSize(A, &nrow, &ncol));
  for (PetscInt i=0; i<nrow; ++i) {
    PetscCall(VecGetValues(diag, 1, &i, &A_ii));
    sigma = 1./sqrt(A_ii);
    PetscCall(MatGetRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatGetRow(B, i, &ncols_B, &cols_B, &vals_B));
    PetscCall(VecGetValues(b, 1, &i, &mu_bar));
    PetscCall(VecGetValues(y, ncols_A, cols_A, y_local));
    for (PetscInt j=0; j<ncols_A; ++j) {
      mu_bar -= vals_A[j]*y_local[j];
    }
    PetscCall(VecGetValues(event_counts, ncols_B, cols_B, n_local));
    for (PetscInt j=0; j<ncols_B; ++j) {
      mu_bar += vals_B[j]*n_local[j];
    }
    if (ncols_B > 0) {
      // sample with rejection sampling
    } else {
      // just draw a Gaussian random variable with mean mu_bar and width sigma
    }
    mu_bar /= A_ii;
    PetscCall(MatRestoreRow(A, i, &ncols_A, &cols_A, &vals_A));
    PetscCall(MatRestoreRow(B, i, &ncols_B, &cols_B, &vals_B));
  }
  PetscCall(VecCopy(y,w));
  PetscCall(PetscFree(y_local));
  PetscCall(PetscFree(n_local));
  PetscCall(VecDestroy(&diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PoissonGibbs(PC pc, Vec b, Vec y)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscCall(PCPoissonGibbsSample(pc, b, y, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_PoissonGibbs(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_PoissonGibbs poissongibbs = pc->data;

  PetscFunctionBeginUser;
  poissongibbs->sample_index = 0;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(PCPoissonGibbsSample(pc, b, y, w));
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
  PetscCall(PetscFree(poissongibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_PoissonGibbs(PC pc)
{
  PC_PoissonGibbs poissongibbs = pc->data;
  
  PetscFunctionBeginUser;
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
