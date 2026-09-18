/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
    */

/** @file snes_poissonmala.c
    @brief Preconditioned MALA sampler for posterior obtained by conditioning a Gaussian prior
    on a Poisson process

    # Options database keys
    - `-poissonmala_epsilon` MALA stepsize

    # Notes

    TODO
*/

#include "parmgmc/snes/snes_poissonmala.h"
#include "parmgmc/poisson.h"

#include <petsc/private/snesimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscksp.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

/* Internal workspace for Poisson MALA sampler SNES */
typedef struct {
  Mat         Z_diag;  // Diagonal matrix Z
  Mat         G_lr;    // The n x m matrix G used for the low rank update
  KSP         ksp;     // internal linear solver
  PetscScalar epsilon; // MALA stepsize
} SNES_PoissonMALA;

/* Generate a new sample by updating theta -> theta' 
 *
 * This is the key computational routine for generating a new sample. 
 *
 * Parameters
 *   snes [inout] : underlying SNES object
 */
static PetscErrorCode SNESSample_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;
  KSP               ksp;
  PoissonCtx       *ctx;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  snes->reason = SNES_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reset SNES object
 *
 * Free contents of temporary workspace to prepare object for reuse
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESReset_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(KSPReset(poissonmala->ksp));
  PetscCall(MatDestroy(&poissonmala->Z_diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Destroy SNES object
 *
 * Free contents of temporary workspace with SNESReset_PoissonMALA() and in addition
 * deallocate the workspace itself.
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESDestroy_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESReset_PoissonMALA(snes));
  PetscCall(KSPDestroy(&poissonmala->ksp));
  PetscCall(PetscFree(poissonmala));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set up SNES object
 *   
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESSetUp_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;
  Vec               z_diag_inv;
  Mat               B_bar;
  Mat               B_meas_dense;
  Mat               S;
  PetscInt          n_row, n_col;
  KSP               ksp_dense;
  PC                pc_dense;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // Set linear operators of KSP
  PetscCall(KSPSetOperators(poissonmala->ksp, ctx->Q_prec, ctx->Q_prec));
  // Create diagonal matrix Z
  PetscCall(MatCreateDiagonal(ctx->event_counts, &poissonmala->Z_diag));
  // Create matrix bar(B) = Q^{-1} B
  PetscCall(MatGetSize(ctx->B_meas, &n_row, &n_col));
  PetscCall(MatConvert(ctx->B_meas, MATDENSE, MAT_INITIAL_MATRIX, &B_meas_dense));
  PetscCall(MatDuplicate(B_meas_dense, MAT_DO_NOT_COPY_VALUES, &B_bar));
  PetscCall(KSPMatSolve(poissonmala->ksp, B_meas_dense, B_bar));
  // Create matrix G = bar(B) (Z^{-1} + B^T bar(B))^{-1}
  PetscCall(MatCreateVecs(poissonmala->Z_diag, NULL, &z_diag_inv));
  PetscCall(MatGetDiagonal(poissonmala->Z_diag, z_diag_inv));
  PetscCall(VecReciprocal(z_diag_inv));
  PetscCall(MatTransposeMatMult(B_meas_dense, B_bar, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &S));
  PetscCall(MatDiagonalSet(S, z_diag_inv, ADD_VALUES));
  PetscCall(MatView(S, PETSC_VIEWER_STDOUT_WORLD));
  // Solve S G^T = bar(B)^T
  PetscCall(MatTranspose(B_bar, MAT_INPLACE_MATRIX, &B_bar));
  PetscCall(MatDuplicate(B_bar, MAT_DO_NOT_COPY_VALUES, &poissonmala->G_lr));
  PetscCall(KSPCreate(PetscObjectComm(snes), &ksp_dense));
  PetscCall(KSPSetType(ksp_dense, KSPPREONLY));
  PetscCall(KSPGetPC(ksp_dense, &pc_dense));
  PetscCall(PCSetType(pc_dense, PCLU));
  PetscCall(KSPSetOperators(ksp_dense, S, S));
  PetscCall(KSPMatSolve(ksp_dense, B_bar, poissonmala->G_lr));
  PetscCall(MatTranspose(poissonmala->G_lr, MAT_INPLACE_MATRIX, &poissonmala->G_lr));
  // Free memory
  PetscCall(MatDestroy(&B_meas_dense));
  PetscCall(MatDestroy(&B_bar));
  PetscCall(MatDestroy(&S));
  PetscCall(VecDestroy(&z_diag_inv));
  PetscCall(KSPDestroy(&ksp_dense));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Inspect options database to set SNES parameters
 *
 * The only parameter that can be set is the stepsize epsilon
 *
 * Parameters
 *   snes [inout] : SNES object
 *   PetscOptionsObject [in] : PETSc options object
 */
static PetscErrorCode SNESSetFromOptions_PoissonMALA(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBegin;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson MALA options");
  PetscCall(PetscOptionsReal("-poissonmala_epsilon", "Stepsize", NULL, poissonmala->epsilon, &poissonmala->epsilon, NULL));
  PetscCall(KSPAppendOptionsPrefix(poissonmala->ksp, "poissonmala_"));
  PetscCall(KSPSetFromOptions(poissonmala->ksp));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* View SNES object
 * 
 * Parameters
 *   snes [in] : SNES object
 *   viewer [inout] : viewer to user
 */
static PetscErrorCode SNESView_PoissonMALA(SNES snes, PetscViewer viewer)
{
  SNES_PoissonMALA *poissonmala;
  (void)poissonmala;
  (void)viewer;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(PetscViewerASCIIPrintf(viewer, "  stepsize=%g\n", (double)poissonmala->epsilon));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Linear solver\n"));
  PetscCall(KSPView(poissonmala->ksp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create unitialised SNES object
 *
 * Parameters
 *   snes [inout] : SNES object to create
 */
PetscErrorCode SNESCreate_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;
  PC                pc;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissonmala));
  snes->data = (void *)poissonmala;

  snes->ops->solve          = SNESSample_PoissonMALA;
  snes->ops->destroy        = SNESDestroy_PoissonMALA;
  snes->ops->reset          = SNESReset_PoissonMALA;
  snes->ops->setup          = SNESSetUp_PoissonMALA;
  snes->ops->setfromoptions = SNESSetFromOptions_PoissonMALA;
  snes->ops->view           = SNESView_PoissonMALA;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;
  PetscCall(KSPCreate(PetscObjectComm(snes), &poissonmala->ksp));

  PetscCall(KSPSetType(poissonmala->ksp, KSPPREONLY));
  PetscCall(KSPGetPC(poissonmala->ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));

  poissonmala->epsilon = 0.1;

  PetscFunctionReturn(PETSC_SUCCESS);
}
