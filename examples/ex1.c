/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Samples from a Gaussian random field with Matern covariance using standalone
 *  Gibbs and Cholesky samplers, and the GAMGMC Multigrid Monte Carlo sampler.
 *  The precision operator is discretised using using finite differences.
 *  For GAMGMC, this tests both the fully algrabic variant and the geometric
 *  variant.
 */

/**************************** Test specification ****************************/
// Gibbs with default omega
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -skip_petscrc

// Gibbs with backward sweep
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -pc_gibbs_backward -skip_petscrc

// Gibbs with symmetric sweep
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -pc_gibbs_symmetric -skip_petscrc

// Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type cholsampler -skip_petscrc

// Algebraic MGMC using PCGAMGMC with coarse Gibbs
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_coarse_ksp_max_it 2 -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -gamgmc_pc_mg_galerkin both -skip_petscrc

// Algebraic MGMC using PCGAMGMC with coarse Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type preonly -gamgmc_mg_coarse_pc_type cholsampler -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -skip_petscrc -log_view

// Geometric MGMC using PCGAMGMC with coarse Gibbs
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_pc_mg_levels 3 -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_coarse_ksp_max_it 2 -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -skip_petscrc

// Geometric MGMC using PCGAMGMC with coarse Cholesky
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_pc_mg_levels 3 -da_grid_x 3 -da_grid_y 3 -gamgmc_mg_levels_ksp_type richardson -gamgmc_mg_levels_pc_type gibbs -gamgmc_mg_coarse_ksp_type preonly -gamgmc_mg_coarse_pc_type cholsampler -gamgmc_mg_levels_ksp_max_it 2 -da_refine 2 -skip_petscrc
/****************************************************************************/

#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petscdm.h>
#include <petscmath.h>
#include <petscsystypes.h>
#include <petscksp.h>
#include <petscvec.h>

static PetscErrorCode SampleCallback(PetscInt it, Vec y, void *ctx)
{
  Vec mean = ctx;

  PetscFunctionBeginUser;
  PetscCall(VecAXPBY(mean, 1. / (it + 1), it / (it + 1.), y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM        da;
  Mat       A;
  Vec       b, x, mean, ex_mean;
  KSP       ksp;
  PC        pc;
  PetscReal err, ex_mean_norm;
  PetscInt  n_samples = 500000;
  PetscInt  n_burnin  = 1000;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-samples", &n_samples, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-burnin", &n_burnin, NULL));

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 10, A));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(DMCreateGlobalVector(da, &mean));
  PetscCall(KSPGetPC(ksp, &pc));

  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &ex_mean));
  PetscCall(VecSet(b, 1));
  PetscCall(VecSet(x, 0));

  {
    KSP ksp2;

    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp2));
    PetscCall(KSPSetOperators(ksp2, A, A));
    PetscCall(KSPSetTolerances(ksp2, 1e-12, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSolve(ksp2, b, ex_mean));
    PetscCall(KSPDestroy(&ksp2));
  }

  // Burn-in phase: advance the chain without recording samples
  PetscCall(KSPSetTolerances(ksp, 0, 0, 0, n_burnin));
  PetscCall(KSPSolve(ksp, b, x));

  // Sampling phase
  PetscCall(PCSetSampleCallback(pc, SampleCallback, mean, NULL));
  PetscCall(KSPSetTolerances(ksp, 0, 0, 0, n_samples));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecAXPY(mean, -1, ex_mean));
  PetscCall(VecNorm(mean, NORM_2, &err));
  PetscCall(VecNorm(ex_mean, NORM_2, &ex_mean_norm));

  PetscCheck(PetscIsCloseAtTol(err / ex_mean_norm, 0, 0.01, 0.01), MPI_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sample mean has not converged: rel. error %.5f", err / ex_mean_norm);
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Rel. mean error: %.5f\n", err / ex_mean_norm));

  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&ex_mean));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&da));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
}
