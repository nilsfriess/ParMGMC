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
#include "parmgmc/parmgmc.h"

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
  PetscRandom prand;             // Random number generator
  Vec         exp_nu;            // Vector exp(-nu) [data]
  Vec         sqrt_n;            // Vector sqrt(n)  [data]
  Mat         G_lr;              // The n x m matrix G used for the low rank update
  Vec        *work;              // Temporary vectors
                                 //   0: theta*    [state]
                                 //   1: phi       [state]
                                 //   2: phi*      [state]
                                 //   3: xi        [state]
                                 //   4: (various) [state]
                                 //   5: (various) [state]
                                 //   6: (various) [data]
                                 //   7: (various) [data]
  KSP         ksp;               // internal linear solver
  KSP         ksp_prior_sampler; // KSP for sampling from the prior
  PetscScalar epsilon;           // MALA stepsize
  PetscInt    its;               // Number of iterations
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
  PoissonCtx       *ctx;
  Vec               tmp_1, tmp_2, tmp_3, tmp_4;
  PetscScalar       alpha;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));

  // Temporary vectors
  tmp_1 = poissonmala->work[6];
  tmp_2 = poissonmala->work[7];
  tmp_3 = poissonmala->work[4];
  tmp_4 = poissonmala->work[5];
  // Construct vector sigma = n - exp(B^T theta-nu)
  PetscCall(VecCopy(ctx->event_counts, tmp_1));
  PetscCall(MatMultTranspose(ctx->B_meas, theta, tmp_2));
  PetscCall(VecAXPY(tmp_2, -1.0, ctx->nu));
  PetscCall(VecExp(tmp_2));
  PetscCall(VecAXPY(tmp_1, -1.0, tmp_2)); // tmp_1 = sigma
  // Compute s = f - Q theta + B sigma
  PetscCall(VecCopy(f_rhs, tmp_3));
  PetscCall(MatMult(ctx->Q_prec, theta, tmp_4)); // tmp_4 = Q theta
  PetscCall(VecAXPY(tmp_3, -1.0, tmp_4));
  PetscCall(MatMultAdd(ctx->B_meas, tmp_1, tmp_3, tmp_3)); // tmp_3 = f - Q theta + B sigma
  // Solve Q tau = s for tau
  PetscCall(KSPSolve(poissonmala->ksp, tmp_3, tmp_4)); // tmp_4 = tau = Q^{-1} (f - Q theta + B sigma)
  // Compute phi = theta + epsilon^2/2 * (tau - G B^T tau)
  PetscCall(MatMultTranspose(ctx->B_meas, tmp_4, tmp_1));
  PetscCall(MatMult(poissonmala->G_lr, tmp_1, tmp_3)); // tmp_3 = G B^T tau
  PetscCall(VecAXPY(tmp_4, -1.0, tmp_3));              // tmp_4 = tau - G B^T tau
  PetscCall(VecCopy(theta, phi));
  alpha = 0.5 * poissonmala->epsilon * poissonmala->epsilon;
  PetscCall(VecAXPY(phi, alpha, tmp_4)); // phi = theta + epsilon^2/2 * (tau - G B^T tau)
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute delta_1 = log(p(theta*)/p(theta)) */
static PetscErrorCode SNESPoissonMALAProposalDelta1_Private(SNES snes, Vec theta, Vec theta_star, Vec f_rhs, PetscScalar *delta_1)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;
  Vec               tmp_1, tmp_2, tmp_3;
  PetscScalar       dot_product;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // Temporary vectors
  tmp_1 = poissonmala->work[4];
  tmp_2 = poissonmala->work[6];
  tmp_3 = poissonmala->work[7];
  // Initialise
  *delta_1 = 0.0;
  // Step 1: + 1/2 theta^T Q theta
  PetscCall(MatMult(ctx->Q_prec, theta, tmp_1)); // tmp_1 = Q theta
  PetscCall(VecDot(theta, tmp_1, &dot_product));
  *delta_1 += 0.5 * dot_product;
  // Step 2: - 1/2 (theta*)^T Q theta*
  PetscCall(MatMult(ctx->Q_prec, theta_star, tmp_1)); // tmp_1 = Q theta*
  PetscCall(VecDot(theta_star, tmp_1, &dot_product));
  *delta_1 -= 0.5 * dot_product;
  // Step 3: + f^T (theta*-theta)
  PetscCall(VecCopy(theta_star, tmp_1));
  PetscCall(VecAXPY(tmp_1, -1.0, theta)); // tmp_1 = theta* - theta
  PetscCall(VecDot(f_rhs, tmp_1, &dot_product));
  *delta_1 += dot_product;
  // Step 4: + n^T B^T (theta* - theta)
  PetscCall(MatMultTranspose(ctx->B_meas, tmp_1, tmp_2)); // tmp_2 = B^T (theta* - theta)
  PetscCall(VecDot(ctx->event_counts, tmp_2, &dot_product));
  *delta_1 += dot_product;
  // Step 5: + (e^nu)^T (e^(B^T theta) - e^(B^T theta*))
  PetscCall(MatMultTranspose(ctx->B_meas, theta, tmp_2));
  PetscCall(VecExp(tmp_2)); // tmp_2 = exp(B^T theta)
  PetscCall(MatMultTranspose(ctx->B_meas, theta_star, tmp_3));
  PetscCall(VecExp(tmp_3));               // tmp_3 = exp(B^T theta*)
  PetscCall(VecAXPY(tmp_2, -1.0, tmp_3)); // tmp_2 = exp(B^T theta) - exp(B^T theta*)
  PetscCall(VecDot(poissonmala->exp_nu, tmp_2, &dot_product));
  *delta_1 += dot_product;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute delta_2 = log(pi(theta*|theta)/pi(theta|theta*)) */
static PetscErrorCode SNESPoissonMALAProposalDelta2_Private(SNES snes, Vec theta, Vec theta_star, Vec phi, Vec phi_star, PetscScalar *delta_2)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;
  Vec               tmp_1, tmp_2, tmp_3;
  PetscScalar       dot_product, nrm;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // Temporary vectors
  tmp_1 = poissonmala->work[4];
  tmp_2 = poissonmala->work[5];
  tmp_3 = poissonmala->work[6];
  // Initialise to zero
  *delta_2 = 0;
  // Step 1: + Delta^T Q Delta
  PetscCall(VecCopy(theta_star, tmp_1));
  PetscCall(VecAXPY(tmp_1, -1.0, phi));          // tmp_1 = Delta = theta* - Phi(theta)
  PetscCall(MatMult(ctx->Q_prec, tmp_1, tmp_2)); // tmp_2 = Q Delta
  PetscCall(VecDot(tmp_1, tmp_2, &dot_product));
  *delta_2 += dot_product;
  // Step 2: + (B^T Delta)^T Z (B^T Delta) = || Z^{1/2} B^T Delta ||_2^2
  PetscCall(MatMultTranspose(ctx->B_meas, tmp_1, tmp_3));
  PetscCall(VecPointwiseMult(tmp_3, tmp_3, poissonmala->sqrt_n));
  PetscCall(VecNorm(tmp_3, NORM_2, &nrm));
  *delta_2 += nrm * nrm;
  // Step 3: - (Delta*)^T Q Delta*
  PetscCall(VecCopy(theta, tmp_1));
  PetscCall(VecAXPY(tmp_1, -1.0, phi_star));     // tmp_1 = Delta* = theta - Phi(theta*)
  PetscCall(MatMult(ctx->Q_prec, tmp_1, tmp_2)); // tmp_2 = Q Delta*
  PetscCall(VecDot(tmp_1, tmp_2, &dot_product));
  *delta_2 -= dot_product;
  // Step 4: - (B^T Delta*)^T Z (B^T Delta*) = || Z^{1/2} B^T Delta* ||_2^2
  PetscCall(MatMultTranspose(ctx->B_meas, tmp_1, tmp_3));
  PetscCall(VecPointwiseMult(tmp_3, tmp_3, poissonmala->sqrt_n));
  PetscCall(VecNorm(tmp_3, NORM_2, &nrm));
  *delta_2 -= nrm * nrm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESPoissonMALAProposalDraw_Private(SNES snes, Vec xi)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;
  Vec               tmp_1, tmp_2;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  // Temporary vectors
  tmp_1 = poissonmala->work[6];
  tmp_2 = poissonmala->work[4];
  // Draw zeta ~ N(0,I)
  PetscCall(VecSetRandomStandardNormal(tmp_1, poissonmala->prand));
  // Construct RHS f_rhs = B Z^{1/2} zeta
  PetscCall(VecPointwiseMult(tmp_1, tmp_1, poissonmala->sqrt_n));
  PetscCall(MatMult(ctx->B_meas, tmp_1, tmp_2));
  // Sample xi* ~ N(Q^{-1}f_rhs,Q^{-1})
  PetscCall(KSPSolve(poissonmala->ksp_prior_sampler, tmp_2, xi));
  // Low-rank update
  PetscCall(MatMultTranspose(ctx->B_meas, xi, tmp_1));
  PetscCall(MatMult(poissonmala->G_lr, tmp_1, tmp_2));
  PetscCall(VecAXPY(xi, -1.0, tmp_2));
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
  PoissonCtx       *ctx;
  Vec               theta, f_rhs;
  Vec               xi, phi, phi_star, theta_star;
  PetscScalar       delta_1, delta_2, delta, u_random;
  PetscBool         accepted;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  theta      = snes->vec_sol;
  f_rhs      = snes->vec_rhs;
  theta_star = poissonmala->work[0];
  phi        = poissonmala->work[1];
  phi_star   = poissonmala->work[2];
  xi         = poissonmala->work[3];
  for (PetscInt it = 0; it < poissonmala->its; ++it) {
    // Compute proposal bias
    PetscCall(SNESPoissonMALAProposalBias_Private(snes, theta, f_rhs, phi));
    // Draw xi ~ N(0,M^{-1})
    PetscCall(SNESPoissonMALAProposalDraw_Private(snes, xi));
    // Compute proposal theta^*+ epsilon*xi
    PetscCall(VecCopy(phi, theta_star));
    PetscCall(VecAXPY(theta_star, poissonmala->epsilon, xi));
    // Compute reverse proposal bias
    PetscCall(SNESPoissonMALAProposalBias_Private(snes, theta_star, f_rhs, phi_star));
    // Proposal delta
    PetscCall(SNESPoissonMALAProposalDelta1_Private(snes, theta, theta_star, f_rhs, &delta_1));
    PetscCall(SNESPoissonMALAProposalDelta2_Private(snes, theta, theta_star, phi, phi_star, &delta_2));
    delta    = delta_1 + delta_2 / (2 * poissonmala->epsilon * poissonmala->epsilon);
    accepted = true;
    if (delta < 0) {
      PetscCall(PetscRandomGetValueReal(poissonmala->prand, &u_random));
      accepted = log(1 - u_random) < delta; // Use 1-u since u is in [0,1)
    }
    if (accepted) PetscCall(VecCopy(theta_star, theta));
  }
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
  PetscCall(PetscRandomDestroy(&poissonmala->prand));
  PetscCall(KSPReset(poissonmala->ksp));
  PetscCall(KSPReset(poissonmala->ksp_prior_sampler));
  if (poissonmala->exp_nu) PetscCall(VecDestroy(&poissonmala->exp_nu));
  if (poissonmala->sqrt_n) PetscCall(VecDestroy(&poissonmala->sqrt_n));
  if (poissonmala->work) {
    for (PetscInt i = 0; i < 8; ++i) {
      if (poissonmala->work[i] != NULL) PetscCall(VecDestroy(&poissonmala->work[i]));
    }
    PetscCall(PetscFree(poissonmala->work));
  }
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
  PetscCall(KSPDestroy(&poissonmala->ksp_prior_sampler));
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
  if (!poissonmala->prand) PetscCall(ParMGMCGetPetscRandom(&poissonmala->prand));
  // Create temporary work vectors
  PetscCall(PetscMalloc1(8, &poissonmala->work));
  for (int i = 0; i < 6; ++i) { PetscCall(MatCreateVecs(ctx->B_meas, NULL, &poissonmala->work[i])); }
  for (int i = 6; i < 8; ++i) { PetscCall(MatCreateVecs(ctx->B_meas, &poissonmala->work[i], NULL)); }
  // Set linear operators of KSPs
  PetscCall(KSPSetOperators(poissonmala->ksp, ctx->Q_prec, ctx->Q_prec));
  PetscCall(KSPSetOperators(poissonmala->ksp_prior_sampler, ctx->Q_prec, ctx->Q_prec));
  PetscCall(KSPSetUp(poissonmala->ksp));
  PetscCall(KSPSetUp(poissonmala->ksp_prior_sampler));
  // Create vector exp(-nu)
  PetscCall(VecDuplicate(ctx->nu, &poissonmala->exp_nu));
  PetscCall(VecCopy(ctx->nu, poissonmala->exp_nu));
  PetscCall(VecScale(poissonmala->exp_nu, -1.0));
  PetscCall(VecExp(poissonmala->exp_nu));
  // Create vector sqrt(n)
  PetscCall(VecDuplicate(ctx->event_counts, &poissonmala->sqrt_n));
  PetscCall(VecCopy(ctx->event_counts, poissonmala->sqrt_n));
  PetscCall(VecSqrtAbs(poissonmala->sqrt_n));
  // Create matrix bar(B) = Q^{-1} B
  PetscCall(MatGetSize(ctx->B_meas, &n_row, &n_col));
  PetscCall(MatConvert(ctx->B_meas, MATDENSE, MAT_INITIAL_MATRIX, &B_meas_dense));
  PetscCall(MatDuplicate(B_meas_dense, MAT_DO_NOT_COPY_VALUES, &B_bar));
  PetscCall(KSPMatSolve(poissonmala->ksp, B_meas_dense, B_bar));
  // Create matrix G = bar(B) (Z^{-1} + B^T bar(B))^{-1}
  z_diag_inv = poissonmala->work[6];
  PetscCall(VecCopy(ctx->event_counts, z_diag_inv));
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
  PetscCall(PetscOptionsInt("-poissonmala_its", "Number of Poisson MALA iterations", NULL, poissonmala->its, &poissonmala->its, NULL));
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
  PetscCall(PetscViewerASCIIPrintf(viewer, "  number of iterations=%" PetscInt_FMT "\n", poissonmala->its));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Linear solver\n"));
  PetscCall(KSPView(poissonmala->ksp, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Prior sampler\n"));
  PetscCall(KSPView(poissonmala->ksp_prior_sampler, viewer));
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

  poissonmala->its = 1;

  // Create KSP for prior solver
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &poissonmala->ksp));
  PetscCall(KSPSetType(poissonmala->ksp, KSPPREONLY));
  PetscCall(KSPGetPC(poissonmala->ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));

  // Create KSP for prior sampler
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &poissonmala->ksp_prior_sampler));
  PetscCall(KSPSetType(poissonmala->ksp_prior_sampler, KSPPREONLY));
  PetscCall(KSPGetPC(poissonmala->ksp_prior_sampler, &pc));
  PetscCall(PCSetType(pc, PCCHOLSAMPLER));

  poissonmala->epsilon = 0.1;

  PetscFunctionReturn(PETSC_SUCCESS);
}
