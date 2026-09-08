/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.

    Non-linear Gibbs sampler for posterior obtained by conditioning a Gaussian prior
    on a Poisson process. The precision matrix of the prior is assumed to be the
    N x N sparse matrix Q. The N x m measurement matrix B describes the coupling between
    the vector of unknowns theta and the observations. More specifically, we have for the
    posterior:

      theta ~ product_{k=1}^{m} P(Lambda_k,n_k) N(mu,Q^{-1})

    where P(Lambda,n) = Lambda^n/n! e^{-Lambda} is the Poisson probability density and
    N(mu,Q^{-1}) is a multivariate normal probability distribution. The rate Lambda_k is given
    by
    
      Lambda_k(theta) = exp(b^{(k)} theta - nu_k) 
    
    where b^{(k)} is the k-th column of B.

    The vectors n (size m), nu (size m) and the matrices Q (shape N x N) and B (shape N x m)
    are collected in the user context PoissonGibbsCtx defined in snes_poissongibbs.h.

    The mean mu is provided via the right hand side vector f = Q.mu which is passed to the
    solve routine.

    The code can be run in two setups:

      1. Both the solution y=theta and the right hand side b=f are vectors of size N.
         In this case, the vector nu provided in the user context is used.
      2. The solution y=(theta,z) and the right hand side b=(f,nu) are nested vectors with
         two components of size N and m respectively. In this setup, the nu in the user context
         is ignored, and the nu in the right-hand side vector b is used.

    The second configuration is required for the hierarchical extension of the algorithm,
    which requires passing a modified nu' = nu - B^T theta to the coarser level samplers.

    The number its of Gibbs- sweeps over the unknowns can be set with 

      -poissongibbs_its its
*/

#include "parmgmc/snes/snes_poissongibbs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/snesimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

/* Internal workspace for Poisson Gibbs sampler SNES */
typedef struct {
  PetscRandom prand;            // Random number generator
  Vec         random_workspace; // Workspace vector containing normally distributed random numbers
  PetscInt    random_work_ptr;  // Pointer to current entry in random number vector
  PetscInt    its;              // Number of iterations
} SNES_PoissonGibbs;

#define RANDOM_BUFFER_SIZE 64 // Size of workspace vector with random numbers

/* Compute largest number of nonzeros per row in a given matrix
 *
 * This information can be used to create a sufficiently large buffer which can store
 * all possible matrix rows.
 * 
 * Parameters
 *   mat [in] : matrix to process
 *   max_nnz_per_row [out]: reference to variable which will
 *                           contain the result
 */
static PetscErrorCode SNESPoissonGibbs_GetMaxNnzPerRow(Mat mat, PetscInt *max_nnz_per_row)
{
  PetscInt        nnz;
  const PetscInt *row_ptr;
  PetscBool       done;

  PetscFunctionBeginUser;
  *max_nnz_per_row = 0;
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i = 1; i <= nnz; ++i) { *max_nnz_per_row = PetscMax(*max_nnz_per_row, row_ptr[i] - row_ptr[i - 1]); }
  PetscCall(MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Return single normally distributed random number
 * 
 * The random number z ~ N(0,1) has mean zero and variance 1. Internally, this function
 * populates a buffer with random numbers of size RANDOM_BUFFER_SIZE and then extracts individual
 * random numbers by iterating over this buffer, repopulating it on the fly once the samples
 * are exhausted.
 * 
 * Parameters
 *   snes [in] : the SNES object
 *   r [out] : reference to variable which will contain the resulting random number 
 */
static PetscErrorCode SNESPoissonGibbs_StandardNormal(SNES snes, PetscScalar *r)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;

  PetscFunctionBeginUser;
  if (poissongibbs->random_work_ptr >= RANDOM_BUFFER_SIZE) {
    PetscCall(VecSetRandomStandardNormal(poissongibbs->random_workspace, poissongibbs->prand));
    poissongibbs->random_work_ptr = 0;
  }
  PetscCall(VecGetValues(poissongibbs->random_workspace, 1, &poissongibbs->random_work_ptr, r));
  poissongibbs->random_work_ptr++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient dphi/dtheta(theta) of function which defines the one-dimensional pdf
 *
 * Compute the gradient 
 *
 *   dphi/dtheta = sum_{k=1}^{n_k} B_{ik} exp(B_{ik} theta + nu_k ) +( theta - bar(mu))/sigma^2
 * 
 * Parameters
 *   theta [in] : current value of theta
 *   mu_bar [in] : mean value of local Gaussian prior
 *   sigma [in] : variance of local Gaussian prior
 *   n_k [in] : number of measurements to include
 *   nu [in] : array with offset values nu_k
 *   b [in] : array with measurements B_{ik} for a fixed i; this is a columns of the measurement
 *       matrix B
 * 
 * Returns
 *   gradient dphi/dtheta
 */
static PetscScalar grad_phi(const PetscScalar theta, const PetscScalar mu_bar, const PetscScalar sigma, const PetscInt n_k, const PetscScalar *nu, const PetscScalar *b)
{
  PetscScalar g = (theta - mu_bar) / (sigma * sigma);
  for (PetscInt k = 0; k < n_k; ++k) { g += b[k] * exp(b[k] * theta + nu[k]); }
  return g;
}

/* Find maximum of the function phi which defines the one-dimensional pdf
 *
 * Uses bisection to find the maximum of phi, which can then be used to
 * obtain a better envelope.
 * 
 * Parameters
 *   mu_bar [in] : mean value of local Gaussian prior
 *   sigma [in] : variance of local Gaussian prior
 *   n_k [in]: number of measurements to include
 *   nu [in] : array with offsets nu
 *   b [in] : array with measurements B_{ik} for a fixed i; this is a columns of the measurement
 *            matrix B
 * Returns
 *   Location of maximum of function phi
 */
static PetscScalar find_argmax_phi(const PetscScalar mu_bar, const PetscScalar sigma, const PetscInt n_k, const PetscScalar *nu, const PetscScalar *b)
{
  PetscScalar theta, theta_old, theta_left, theta_right, theta_mid;
  PetscScalar g, g_old, g_left, g_mid;
  PetscScalar bracketing_tolerance = 1.E-12;
  PetscScalar bisection_tolerance  = 1.E-10;
  PetscScalar delta;

  theta     = mu_bar;
  g         = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
  theta_old = theta;
  g_old     = g;
  // Step 1: bracket minimum, i.e. find [a,b] such that dphi/dtheta changes sign in
  //         this interval
  delta = sigma;
  while (true) {
    if (g > 0) {
      theta -= delta;
    } else {
      theta += delta;
    }
    delta *= 2;
    g = grad_phi(theta, mu_bar, sigma, n_k, nu, b);
    // stop when derivative changes sign
    if (((g > 0) && (g_old < 0)) || ((g < 0) && (g_old > 0))) break;
    if (fabs(g) < bracketing_tolerance) break;
    theta_old = theta;
    g_old     = g;
  }
  // Step 2: iteratively bisect [a,b] to find the point where dphi/dtheta = 0
  if (theta_old < theta) {
    theta_left  = theta_old;
    theta_right = theta;
    g_left      = g_old;
  } else {
    theta_left  = theta;
    theta_right = theta_old;
    g_left      = g;
  }
  while ((theta_right - theta_left) / fmax(fabs(theta_right), fabs(theta_left)) > bisection_tolerance) {
    theta_mid = 0.5 * (theta_left + theta_right);
    g_mid     = grad_phi(theta_mid, mu_bar, sigma, n_k, nu, b);
    if (((g_left > 0) && (g_mid > 0)) || ((g_left < 0) && (g_mid < 0))) {
      theta_left = theta_mid;
      g_left     = g_mid;
    } else {
      theta_right = theta_mid;
    }
  }
  return 0.5 * (theta_right + theta_left);
}

/* Generate a new sample by updating theta -> theta' with a number of Gibbs sweeps
 *
 * This is the key computational routine for generating a new sample. It iterates over
 * all unknowns and updates these unknowns individually by drawing from the one-
 * dimensional distribution which is conditioned on all other unknowns and the 
 * measurements.
 * The number of sweeps is controlled by the parameter its.
 *
 * Parameters
 *   snes [inout] : underlying SNES object
 */
static PetscErrorCode SNESSample_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;
  Vec                theta;
  Vec                f_rhs;
  Vec                nu;
  PetscInt           rstart, rend, ncols_Q, ncols_B, max_nnz_per_row;
  const PetscInt    *cols_Q;
  const PetscScalar *vals_Q;
  const PetscInt    *cols_B;
  const PetscScalar *vals_B;
  PetscScalar        sigma;
  PetscScalar        mu_bar;
  PetscScalar       *theta_array;
  PetscScalar       *n_local;
  PetscScalar       *nu_local;
  PetscScalar        theta_bar;
  Vec                v_diag, nu_tilde;
  const PetscScalar *diag;
  const PetscScalar *f_rhs_array;
  PetscScalar        r, theta_prime;
  Mat                Q_prec;
  Mat                B_meas;
  PetscInt           it;
  PoissonGibbsCtx   *ctx;

  PetscFunctionBeginUser;

  PetscCall(SNESGetApplicationContext(snes, &ctx));
  Q_prec = ctx->Q_prec;
  B_meas = ctx->B_meas;
  // Check whether RHS is a nested vector and extract RHS and solution
  PetscBool is_nest;
  PetscCall(PetscObjectTypeCompare((PetscObject)snes->vec_rhs, VECNEST, &is_nest));
  if (is_nest) {
    PetscCall(VecNestGetSubVec(snes->vec_sol, 0, &theta));
    PetscCall(VecNestGetSubVec(snes->vec_rhs, 0, &f_rhs));
    PetscCall(VecNestGetSubVec(snes->vec_rhs, 1, &nu));
  } else {
    theta = snes->vec_sol;
    f_rhs = snes->vec_rhs;
    nu    = ctx->nu;
  }

  PetscCall(VecDuplicate(nu, &nu_tilde));
  PetscCall(MatMultTransposeAdd(ctx->B_meas, theta, nu, nu_tilde));

  // Storage for local part of vectors
  PetscCall(SNESPoissonGibbs_GetMaxNnzPerRow(Q_prec, &max_nnz_per_row));
  PetscCall(SNESPoissonGibbs_GetMaxNnzPerRow(B_meas, &max_nnz_per_row));
  PetscCall(PetscMalloc1(max_nnz_per_row, &n_local));
  PetscCall(PetscMalloc1(max_nnz_per_row, &nu_local));

  PetscCall(VecDuplicate(theta, &v_diag));
  PetscCall(MatGetDiagonal(Q_prec, v_diag));
  PetscCall(VecGetArrayRead(v_diag, &diag));
  PetscCall(VecGetArrayRead(f_rhs, &f_rhs_array));
  PetscCall(VecGetArray(theta, &theta_array));
  PetscCall(MatGetOwnershipRange(Q_prec, &rstart, &rend));
  // Loop over sweeps
  for (it = 0; it < poissongibbs->its; ++it) {
    // Iterate over unknowns
    for (PetscInt i = rstart; i < rend; ++i) {
      // Construct mean and variance of 1d Gaussian distribution
      PetscInt iloc = i - rstart;
      sigma         = 1. / sqrt(diag[iloc]);
      PetscCall(MatGetRow(Q_prec, i, &ncols_Q, &cols_Q, &vals_Q));
      PetscCall(MatGetRow(B_meas, i, &ncols_B, &cols_B, &vals_B));
      PetscCall(VecGetValues(ctx->event_counts, ncols_B, cols_B, n_local));
      PetscCall(VecGetValues(nu_tilde, ncols_B, cols_B, nu_local));
      for (PetscInt k = 0; k < ncols_B; ++k) { nu_local[k] -= theta_array[iloc] * vals_B[k]; }
      mu_bar = f_rhs_array[iloc];
      for (PetscInt j = 0; j < ncols_Q; ++j) {
        if (cols_Q[j] != i) mu_bar -= vals_Q[j] * theta_array[cols_Q[j] - rstart];
      }
      for (PetscInt j = 0; j < ncols_B; ++j) { mu_bar += vals_B[j] * n_local[j]; }
      mu_bar *= sigma * sigma;
      if (ncols_B > 0) {
        // Sample with rejection sampling if unknown couples to at least one
        // measurement
        theta_bar          = find_argmax_phi(mu_bar, sigma, ncols_B, nu_local, vals_B);
        PetscBool accepted = PETSC_FALSE;
        while (!accepted) {
          PetscCall(SNESPoissonGibbs_StandardNormal(snes, &r));
          theta_prime = theta_bar + sigma * r;
          PetscCall(PetscRandomGetValueReal(poissongibbs->prand, &r));
          PetscScalar Fbar = 0;
          for (PetscInt k = 0; k < ncols_B; ++k) { Fbar += exp(vals_B[k] * theta_prime + nu_local[k]) + ((theta_bar - theta_prime) * vals_B[k] - 1.0) * exp(vals_B[k] * theta_bar + nu_local[k]); }
          if (isnan(Fbar) || isinf(Fbar)) { return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_FP, PETSC_ERROR_INITIAL, "Encountered invalid Fbar value (NaN or Inf) in Poisson-Gibbs rejection step"); }
          accepted = (log(r) <= -Fbar);
        }
      } else {
        // Otherwise just draw a Gaussian random variable with mean mu_bar and width sigma
        PetscCall(SNESPoissonGibbs_StandardNormal(snes, &r));
        theta_prime = mu_bar + sigma * r;
      }
      theta_array[iloc] = theta_prime;
      for (PetscInt k = 0; k < ncols_B; ++k) { nu_local[k] += theta_array[iloc] * vals_B[k]; }
      // Restore values
      PetscCall(VecSetValues(nu_tilde, ncols_B, cols_B, nu_local, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(nu_tilde));
      PetscCall(VecAssemblyEnd(nu_tilde));
      PetscCall(MatRestoreRow(Q_prec, i, &ncols_Q, &cols_Q, &vals_Q));
      PetscCall(MatRestoreRow(B_meas, i, &ncols_B, &cols_B, &vals_B));
    }
  }
  snes->reason = SNES_CONVERGED_ITS;
  PetscCall(VecRestoreArrayRead(v_diag, &diag));
  // Restore solution and right hand side vectors
  PetscCall(VecRestoreArrayRead(f_rhs, &f_rhs_array));
  PetscCall(VecRestoreArray(theta, &theta_array));
  // Free temporary storage
  PetscCall(PetscFree(n_local));
  PetscCall(PetscFree(nu_local));
  PetscCall(VecDestroy(&v_diag));
  PetscCall(VecDestroy(&nu_tilde));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reset SNES object
 *
 * Free contents of temporary workspace to prepare object for reuse
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESReset_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&poissongibbs->prand));
  PetscCall(VecDestroy(&poissongibbs->random_workspace));
  PetscCall(SNESPoissonGibbsSetIterations(snes, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Destroy SNES object
 *
 * Free contents of temporary workspace and deallocate the workspace
 * itself.
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESDestroy_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;

  PetscFunctionBeginUser;
  PetscCall(SNESReset_PoissonGibbs(snes));
  PetscCall(PetscFree(poissongibbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set up SNES object
 * 
 * Create temporary workspace which will be used for random
 * number generation
 * 
 * Parameters
 *   snes : SNES object
 */
static PetscErrorCode SNESSetUp_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;

  PetscFunctionBeginUser;
  // Create random number generator and vector which will store
  // random numbers
  if (!poissongibbs->prand) PetscCall(ParMGMCGetPetscRandom(&poissongibbs->prand));
  if (!poissongibbs->random_workspace) PetscCall(VecCreateSeq(PETSC_COMM_SELF, RANDOM_BUFFER_SIZE, &poissongibbs->random_workspace));
  // Point to the end of vector to trigger re-population when the
  // next random number if requested
  poissongibbs->random_work_ptr = RANDOM_BUFFER_SIZE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set the number of Gibbs sweeps
 *
 * Parameters
 *   snes : SNES object
 *   its : new number of sweeps 
 */
PetscErrorCode SNESPoissonGibbsSetIterations(SNES snes, PetscInt its)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;
  PetscFunctionBeginUser;
  poissongibbs->its = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Inspect options database to set SNES parameters
 *
 * The only parameter that can be set is the number of sweeps
 *
 * Parameters
 *   snes [inout] : SNES object
 *   PetscOptionsObject [in] : PETSc options object
 */
static PetscErrorCode SNESSetFromOptions_PoissonGibbs(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;
  (void)poissongibbs;
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson Gibbs options");
  PetscCall(PetscOptionsInt("-poissongibbs_its", "Number of Poisson Gibbs iterations", NULL, poissongibbs->its, &poissongibbs->its, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* View SNES object
 * 
 * Parameters
 *   snes [in] : SNES object
 *   viewer [inout] : viewer to user
 */
static PetscErrorCode SNESView_PoissonGibbs(SNES snes, PetscViewer viewer)
{
  SNES_PoissonGibbs *poissongibbs = (SNES_PoissonGibbs *)snes->data;
  (void)poissongibbs;
  (void)viewer;
  PetscCall(PetscViewerASCIIPrintf(viewer, "  number of iterations=%" PetscInt_FMT "\n", poissongibbs->its));
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create unitialised SNES object
 *
 * Parameters
 *   snes [inout] : SNES object to create
 */
PetscErrorCode SNESCreate_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs *poissongibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissongibbs));
  snes->data = (void *)poissongibbs;

  snes->ops->solve          = SNESSample_PoissonGibbs;
  snes->ops->destroy        = SNESDestroy_PoissonGibbs;
  snes->ops->reset          = SNESReset_PoissonGibbs;
  snes->ops->setup          = SNESSetUp_PoissonGibbs;
  snes->ops->setfromoptions = SNESSetFromOptions_PoissonGibbs;
  snes->ops->view           = SNESView_PoissonGibbs;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  // Set default value value
  PetscCall(SNESPoissonGibbsSetIterations(snes, 1));

  PetscFunctionReturn(PETSC_SUCCESS);
}
