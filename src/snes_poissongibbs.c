/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.

    Non-linear Gibbs sampler for posterior obtained by contitioning a Gaussian prior
    on a Poisson process.
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

typedef struct {
  PetscRandom prand; // Random numbers
  Vec random_workspace;
  PetscInt random_work_ptr;
  PetscInt sample_index;  
  PetscInt its;
}  SNES_PoissonGibbs;

#define RANDOM_BUFFER_SIZE 64

static PetscErrorCode SNESPoissonGibbs_GetMaxNnzPerRow(Mat mat, PetscInt *max_nnz_per_row) {
  PetscInt nnz;
  const PetscInt* row_ptr;
  PetscBool done;

  PetscFunctionBeginUser;
  // Work out maximum number of entries per row for Q_prec and B
  *max_nnz_per_row = 0;
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  for (PetscInt i=1; i<=nnz; ++i) {
    *max_nnz_per_row = PetscMax(*max_nnz_per_row,row_ptr[i]-row_ptr[i-1]);
  }
  PetscCall(MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &nnz, &row_ptr, NULL, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESPoissonGibbs_StandardNormal(SNES snes, PetscScalar *r) {
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;

  PetscFunctionBeginUser;  
  if (poissongibbs->random_work_ptr >= RANDOM_BUFFER_SIZE) {
    PetscCall(VecSetRandomStandardNormal(poissongibbs->random_workspace, poissongibbs->prand));
    poissongibbs->random_work_ptr=0;
  }
  PetscCall(VecGetValues(poissongibbs->random_workspace, 1, &poissongibbs->random_work_ptr, r));
  poissongibbs->random_work_ptr++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESPoissonGibbs_Uniform(SNES snes, PetscScalar *r) {
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;

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

static PetscErrorCode SNESPoissonGibbs_FindMaximum(const PetscScalar mu_bar,
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
static PetscErrorCode SNESSample_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;
  Vec y;
  Vec f_rhs;
  Vec nu;
  PetscInt rstart, rend, ncols_Q, ncols_B, max_nnz_per_row;
  const PetscInt *cols_Q;
  const PetscScalar *vals_Q;
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
  const PetscScalar* f_rhs_array;
  PetscScalar r, theta_prime;
  Mat Q_prec;
  Mat B_meas;
  PetscInt it;
  PoissonGibbsCtx* ctx;  

  PetscFunctionBeginUser;
  y = snes->vec_sol;
  PetscCall(VecNestGetSubVec(snes->vec_rhs,0,&f_rhs));
  PetscCall(VecNestGetSubVec(snes->vec_rhs,1,&nu));

  PetscCall(SNESGetApplicationContext(snes, &ctx));
  Q_prec = ctx->Q_prec;
  B_meas = ctx->B_meas;

  PetscCall(VecDuplicate(nu, &nu_tilde));
  PetscCall(MatMultTransposeAdd(ctx->B_meas, y, nu, nu_tilde));
  
  // Storage for local part of vectors
  PetscCall(SNESPoissonGibbs_GetMaxNnzPerRow(Q_prec, &max_nnz_per_row));
  PetscCall(SNESPoissonGibbs_GetMaxNnzPerRow(B_meas, &max_nnz_per_row));
  PetscCall(PetscMalloc1(max_nnz_per_row, &n_local));
  PetscCall(PetscMalloc1(max_nnz_per_row, &nu_local));
  
  PetscCall(VecDuplicate(y, &v_diag));
  PetscCall(MatGetDiagonal(Q_prec, v_diag));
  PetscCall(VecGetArrayRead(v_diag, &diag));
  PetscCall(VecGetArrayRead(f_rhs, &f_rhs_array));
  PetscCall(VecGetArray(y,&theta));
  PetscCall(MatGetOwnershipRange(Q_prec, &rstart, &rend));

  for (it=0; it<poissongibbs->its; ++it) {
    for (PetscInt i=rstart; i<rend; ++i) {
      PetscInt iloc = i - rstart;
      sigma = 1./sqrt(diag[iloc]);
      PetscCall(MatGetRow(Q_prec, i, &ncols_Q, &cols_Q, &vals_Q));
      PetscCall(MatGetRow(B_meas, i, &ncols_B, &cols_B, &vals_B));
      PetscCall(VecGetValues(ctx->event_counts, ncols_B, cols_B, n_local));
      PetscCall(VecGetValues(nu_tilde, ncols_B, cols_B, nu_local));
      for (PetscInt k=0; k<ncols_B; ++k) {
        nu_local[k] -= theta[iloc]*vals_B[k];
      }
      mu_bar = f_rhs_array[iloc];
      for (PetscInt j=0; j<ncols_Q; ++j) {
        if (cols_Q[j] != i)
          mu_bar -= vals_Q[j]*theta[cols_Q[j]-rstart];
      }
      for (PetscInt j=0; j<ncols_B; ++j) {
        mu_bar += vals_B[j]*n_local[j];
      }
      mu_bar *= sigma*sigma;
      if (ncols_B > 0) {
        // sample with rejection sampling
        PetscCall(SNESPoissonGibbs_FindMaximum(mu_bar, sigma, ncols_B, nu_local, vals_B, &theta_bar));
        PetscBool accepted = PETSC_FALSE;
        while (!accepted) {
          PetscCall(SNESPoissonGibbs_StandardNormal(snes, &r));
          theta_prime = theta_bar + sigma*r;
          PetscCall(SNESPoissonGibbs_Uniform(snes, &r));
          PetscScalar Fbar = 0;
          for (PetscInt k=0; k<ncols_B; ++k) {
            Fbar += exp(vals_B[k]*theta_prime+nu_local[k]) + ((theta_bar - theta_prime)*vals_B[k]-1.0)*exp(vals_B[k]*theta_bar+nu_local[k]);
          }
          if (isnan(Fbar) || isinf(Fbar)) {
            return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_FP, PETSC_ERROR_INITIAL, "Encountered invalid Fbar value (NaN or Inf) in Poisson-Gibbs rejection step");
          }
          accepted = (log(r) <= -Fbar);
        }
      } else {
        // just draw a Gaussian random variable with mean mu_bar and width sigma
        PetscCall(SNESPoissonGibbs_StandardNormal(snes, &r));
        theta_prime = mu_bar + sigma*r;
      }
      theta[iloc] = theta_prime;
      for (PetscInt k=0; k<ncols_B; ++k) {
        nu_local[k] += theta[iloc]*vals_B[k];
      }
      PetscCall(VecSetValues(nu_tilde, ncols_B, cols_B, nu_local, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(nu_tilde));
      PetscCall(VecAssemblyEnd(nu_tilde));
      PetscCall(MatRestoreRow(Q_prec, i, &ncols_Q, &cols_Q, &vals_Q));
      PetscCall(MatRestoreRow(B_meas, i, &ncols_B, &cols_B, &vals_B));
    }
  }
  snes->reason = SNES_CONVERGED_ITS;
  PetscCall(VecRestoreArrayRead(v_diag, &diag));
  PetscCall(VecRestoreArrayRead(f_rhs, &f_rhs_array));
  PetscCall(VecRestoreArray(y, &theta));
  PetscCall(PetscFree(n_local));
  PetscCall(PetscFree(nu_local));
  PetscCall(VecDestroy(&v_diag));
  PetscCall(VecDestroy(&nu_tilde));  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&poissongibbs->prand));  
  PetscCall(VecDestroy(&poissongibbs->random_workspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESDestroy_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&poissongibbs->prand));
  PetscCall(VecDestroy(&poissongibbs->random_workspace));
  PetscCall(PetscFree(poissongibbs));
  PetscFunctionReturn(PETSC_SUCCESS);  
}

static PetscErrorCode SNESSetUp_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;
  
  PetscFunctionBeginUser;
  if (!poissongibbs->prand) PetscCall(ParMGMCGetPetscRandom(&poissongibbs->prand));
  PetscCall(VecCreate(MPI_COMM_WORLD, &poissongibbs->random_workspace));
  PetscCall(VecSetSizes(poissongibbs->random_workspace, RANDOM_BUFFER_SIZE, PETSC_DETERMINE));
  PetscCall(VecSetType(poissongibbs->random_workspace, VECSEQ));
  poissongibbs->random_work_ptr = RANDOM_BUFFER_SIZE;  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_PoissonGibbs(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;
  (void)poissongibbs;
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson Gibbs options");
  PetscCall(PetscOptionsInt("-poissongibbs_its", "Number of Poisson Gibbs iterations", NULL, poissongibbs->its, &poissongibbs->its, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_PoissonGibbs(SNES snes, PetscViewer viewer)
{
  SNES_PoissonGibbs* poissongibbs = (SNES_PoissonGibbs*)snes->data;
  (void)poissongibbs;
  (void)viewer;
  PetscCall(PetscViewerASCIIPrintf(viewer, "  number of iterations=%" PetscInt_FMT "\n", poissongibbs->its));
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESCreate_PoissonGibbs(SNES snes)
{
  SNES_PoissonGibbs* poissongibbs;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissongibbs));
  snes->data       = (void*)poissongibbs;
  
  snes->ops->solve           = SNESSample_PoissonGibbs;
  snes->ops->destroy         = SNESDestroy_PoissonGibbs;
  snes->ops->reset           = SNESReset_PoissonGibbs;
  snes->ops->setup           = SNESSetUp_PoissonGibbs;
  snes->ops->setfromoptions  = SNESSetFromOptions_PoissonGibbs;
  snes->ops->view            = SNESView_PoissonGibbs;  

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}
