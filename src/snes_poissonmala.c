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

/* Compute the MALA proposal bias Phi(theta) 
 *
 * Parameters
 *   snes [in] : underlying SNES object
 *   theta [in] : state theta for which to evaluate the bias
 *   f_rhs [in] : right hand side vector f
 *   phi [out] : resulting value of Phi(theta)
 */
static PetscErrorCode SNESPoissonMALAProposalBias_Private(SNES snes, Vec theta, Vec f_rhs, Vec phi)
{
  SNES_PoissonMALA *poissonmala;
  KSP               ksp;
  PoissonCtx       *ctx;
  Vec               kappa, sigma, s, tau;
  PetscScalar       alpha;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // Compute vector sigma
  PetscCall(VecDuplicate(ctx->event_counts, &sigma));
  PetscCall(VecCopy(ctx->event_counts, sigma));
  PetscCall(MatCreateVecs(ctx->B_meas, &kappa, NULL));
  PetscCall(MatMultTranspose(ctx->B_meas, theta, kappa));
  PetscCall(VecAXPY(kappa, -1.0, ctx->nu));
  PetscCall(VecExp(kappa));
  PetscCall(VecAXPY(sigma, -1.0, kappa));
  // Compute s = f - Q theta + B sigma
  PetscCall(VecDuplicate(theta, &tau));
  PetscCall(VecDuplicate(f_rhs, &s));
  PetscCall(VecCopy(f_rhs, s));
  PetscCall(MatMult(ctx->Q_prec, theta, tau)); // Use tau = Q theta as a temporary
  PetscCall(VecAXPY(s, -1.0, tau));
  PetscCall(MatMultAdd(ctx->B_meas, sigma, s, s));
  // Solve Q tau = s for tau
  PetscCall(KSPSolve(poissonmala->ksp, s, tau));
  // Compute phi = theta + epsilon^2/2 * (tau - G B^T tau)
  PetscCall(MatMultTranspose(ctx->B_meas, tau, kappa)); // use kappa = B^T tau as a temporary
  PetscCall(MatMult(poissonmala->G_lr, kappa, s));      // use s = G kappa = G B^T tau as a temporary
  PetscCall(VecAXPY(tau, -1.0, s));                     // tau -> tau - G B^T tau
  PetscCall(VecCopy(theta, phi));
  alpha = 0.5 * poissonmala->epsilon * poissonmala->epsilon;
  PetscCall(VecAXPY(phi, alpha, tau)); // phi = theta + epsilon^2/2 * (tau - G B^T tau)
  // Free memory
  PetscCall(VecDestroy(&kappa));
  PetscCall(VecDestroy(&sigma));
  PetscCall(VecDestroy(&s));
  PetscCall(VecDestroy(&tau));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESPoissonMALAProposalDelta_Private(SNES snes, Vec theta, Vec theta_star, Vec f_rhs, Vec phi, Vec phi_star, PetscScalar *delta)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;
  Vec               Q_theta, d_theta, B_T_d_theta, exp_B_theta, exp_B_theta_star, exp_nu, Delta, Q_Delta, sqrt_Z_B_Delta, sqrt_Z;
  PetscScalar       delta_1, delta_2_prime;
  PetscScalar       theta_Q_theta, f_theta, n_B_T_dtheta, exp_dot, Delta_Q_Delta, Delta_B_Z_B_T_Delta;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // delta_1 = log(p(theta*)/p(theta))
  delta_1 = 0.0;
  PetscCall(VecDuplicate(ctx->nu, &B_T_d_theta));
  PetscCall(VecDuplicate(ctx->nu, &exp_nu));
  PetscCall(VecDuplicate(ctx->nu, &exp_B_theta));
  PetscCall(VecDuplicate(ctx->nu, &exp_B_theta_star));
  PetscCall(VecDuplicate(ctx->nu, &sqrt_Z_B_Delta));
  PetscCall(VecDuplicate(ctx->event_counts, &sqrt_Z));
  PetscCall(VecDuplicate(theta, &Q_theta));
  PetscCall(VecDuplicate(theta, &Delta));
  PetscCall(MatMult(ctx->Q_prec, theta, Q_theta));
  PetscCall(VecDot(theta, Q_theta, &theta_Q_theta));
  delta_1 += 0.5 * theta_Q_theta; // + 1/2 theta^T Q theta
  PetscCall(MatMult(ctx->Q_prec, theta_star, Q_theta));
  PetscCall(VecDot(theta_star, Q_theta, &theta_Q_theta));
  delta_1 -= 0.5 * theta_Q_theta; // - 1/2 (theta*)^T Q theta*
  PetscCall(VecDuplicate(theta_star, &d_theta));
  PetscCall(VecCopy(theta_star, d_theta));
  PetscCall(VecAXPY(d_theta, -1.0, theta));
  PetscCall(VecDot(f_rhs, d_theta, &f_theta));
  delta_1 += f_theta; // + f^T (theta*-theta)
  PetscCall(MatMultTranspose(ctx->B_meas, d_theta, B_T_d_theta));
  PetscCall(VecDot(ctx->event_counts, B_T_d_theta, &n_B_T_dtheta));
  delta_1 += n_B_T_dtheta; // + n^T B^T (theta* - theta)
  PetscCall(MatMultTranspose(ctx->B_meas, theta, exp_B_theta));
  PetscCall(VecExp(exp_B_theta));
  PetscCall(MatMultTranspose(ctx->B_meas, theta_star, exp_B_theta_star));
  PetscCall(VecExp(exp_B_theta_star));
  PetscCall(VecAXPY(exp_B_theta, -1.0, exp_B_theta_star));
  PetscCall(VecCopy(ctx->nu, exp_nu));
  PetscCall(VecExp(exp_nu));
  PetscCall(VecDot(exp_nu, exp_B_theta, &exp_dot));
  delta_1 += exp_dot; // + (e^nu)^T (e^(B^T theta) - e^(B^T theta*))
  // delta_2 = log(pi(theta*|theta)/pi(theta|theta*))
  PetscCall(VecCopy(ctx->event_counts, sqrt_Z));
  PetscCall(VecSqrtAbs(sqrt_Z));
  delta_2_prime = 0;
  PetscCall(VecCopy(theta, Delta));
  PetscCall(VecAXPY(Delta, -1.0, phi_star));
  PetscCall(VecDuplicate(Delta, &Q_Delta));
  PetscCall(MatMult(ctx->Q_prec, Delta, Q_Delta));
  PetscCall(VecDot(Delta, Q_Delta, &Delta_Q_Delta));
  delta_2_prime += Delta_Q_Delta; // + (Delta*)^T Q Delta *
  PetscCall(MatMultTranspose(ctx->B_meas, Delta, sqrt_Z_B_Delta));
  PetscCall(VecPointwiseMult(sqrt_Z_B_Delta, sqrt_Z_B_Delta, sqrt_Z));
  PetscCall(VecNorm(sqrt_Z_B_Delta, NORM_2, &Delta_B_Z_B_T_Delta));
  delta_2_prime += Delta_B_Z_B_T_Delta;
  PetscCall(VecCopy(theta_star, Delta));
  PetscCall(VecAXPY(Delta, -1.0, phi));
  PetscCall(VecDuplicate(Delta, &Q_Delta));
  PetscCall(MatMult(ctx->Q_prec, Delta, Q_Delta));
  PetscCall(VecDot(Delta, Q_Delta, &Delta_Q_Delta));
  delta_2_prime -= Delta_Q_Delta; // + (Delta*)^T Q Delta *
  PetscCall(MatMultTranspose(ctx->B_meas, Delta, sqrt_Z_B_Delta));
  PetscCall(VecPointwiseMult(sqrt_Z_B_Delta, sqrt_Z_B_Delta, sqrt_Z));
  PetscCall(VecNorm(sqrt_Z_B_Delta, NORM_2, &Delta_B_Z_B_T_Delta));
  delta_2_prime -= Delta_B_Z_B_T_Delta;
  // Final result
  *delta = delta_1 + delta_2_prime / (2 * poissonmala->epsilon * poissonmala->epsilon);
  // Free memory
  PetscCall(VecDestroy(&Q_theta));
  PetscCall(VecDestroy(&d_theta));
  PetscCall(VecDestroy(&B_T_d_theta));
  PetscCall(VecDestroy(&exp_B_theta));
  PetscCall(VecDestroy(&exp_B_theta_star));
  PetscCall(VecDestroy(&exp_nu));
  PetscCall(VecDestroy(&Delta));
  PetscCall(VecDestroy(&Q_Delta));
  PetscCall(VecDestroy(&sqrt_Z));
  PetscCall(VecDestroy(&sqrt_Z_B_Delta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  Vec               theta, f_rhs, xi, phi, phi_star, theta_star;
  PetscScalar       delta;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  theta = snes->vec_sol;
  f_rhs = snes->vec_rhs;
  PetscCall(VecDuplicate(theta, &phi));
  PetscCall(VecDuplicate(theta, &phi_star));
  PetscCall(VecDuplicate(theta, &theta_star));
  PetscCall(VecDuplicate(theta, &xi));
  // Compute proposal bias
  PetscCall(SNESPoissonMALAProposalBias_Private(snes, theta, f_rhs, phi));
  // Draw xi ~ N(0,M^{-1})
  // TODO
  // Compute proposal theta^*+ epsilon*xi
  PetscCall(VecCopy(phi, theta_star));
  PetscCall(VecAXPY(theta_star, poissonmala->epsilon, xi));
  // Compute reverse proposal bias
  PetscCall(SNESPoissonMALAProposalBias_Private(snes, theta_star, f_rhs, phi_star));
  // Proposal delta
  PetscCall(SNESPoissonMALAProposalDelta_Private(snes, theta, theta_star, f_rhs, phi, phi_star, &delta));
  // Free memory
  PetscCall(VecDestroy(&phi));
  PetscCall(VecDestroy(&phi_star));
  PetscCall(VecDestroy(&theta_star));
  PetscCall(VecDestroy(&xi));

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
  // Solve S G^T = bar(B)^T
  PetscCall(MatTranspose(B_bar, MAT_INPLACE_MATRIX, &B_bar));
  PetscCall(MatDuplicate(B_bar, MAT_DO_NOT_COPY_VALUES, &poissonmala->G_lr));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_dense));
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
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &poissonmala->ksp));

  PetscCall(KSPSetType(poissonmala->ksp, KSPPREONLY));
  PetscCall(KSPGetPC(poissonmala->ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));

  poissonmala->epsilon = 0.1;

  PetscFunctionReturn(PETSC_SUCCESS);
}
