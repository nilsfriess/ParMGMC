/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Tests the different samplers with low-rank updates.
 */

/**************************** Test specification ****************************/
// MGMC sampler with low-rank update
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -gamgmc_mg_coarse_ksp_type richardson -gamgmc_mg_coarse_pc_type gibbs -gamgmc_mg_coarse_ksp_max_it 2 -dm_refine_hierarchy 3  %opts -ksp_norm_type none -with_lr -ksp_convergence_test skip 

// Gibbs sampler with low-rank update
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gibbs -pc_gibbs_symmetric -dm_refine_hierarchy 3 -with_lr %opts -ksp_max_it 50000 -ksp_norm_type none -ksp_convergence_test skip 

// Cholesky sampler with low-rank update
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type cholsampler -dm_refine 2 -with_lr  %opts -ksp_convergence_test skip 
/****************************************************************************/

#include <parmgmc/mc_sor.h>
#include <parmgmc/ms.h>
#include <parmgmc/obs.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscksp.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>

typedef struct _SampleCtx {
  Vec         mean, y, mean_exact, tmp;
  PetscScalar mean_exact_norm;
} *SampleCtx;

static PetscErrorCode SampleCtxCreate(DM dm, SampleCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscNew(ctx));
  PetscCall(DMCreateGlobalVector(dm, &(*ctx)->mean));
  PetscCall(DMCreateGlobalVector(dm, &(*ctx)->y));
  PetscCall(DMCreateGlobalVector(dm, &(*ctx)->mean_exact));
  PetscCall(DMCreateGlobalVector(dm, &(*ctx)->tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleCtxDestroy(void *sctx)
{
  SampleCtx *ctx = sctx;

  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&(*ctx)->mean));
  PetscCall(VecDestroy(&(*ctx)->y));
  PetscCall(VecDestroy(&(*ctx)->mean_exact));
  PetscCall(VecDestroy(&(*ctx)->tmp));
  PetscCall(PetscFree(*ctx));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SampleCallbackKSP(PetscInt it, Vec y, void *ctx)
{
  SampleCtx *sctx = ctx;
  Vec        mean = (*sctx)->mean;

  PetscFunctionBeginUser;
  PetscCall(VecScale(mean, it));
  PetscCall(VecAXPY(mean, 1., y));
  PetscCall(VecScale(mean, 1. / (it + 1)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM             dm;
  SampleCtx      samplectx;
  Mat            A, B, Aop;
  Vec            S, b, x, f;
  KSP            ksp;
  PC             pc;
  MS             ms;
  PetscBool      with_lr = PETSC_FALSE;
  const PetscInt nobs    = 3;
  PetscScalar    obs[3 * nobs], radii[nobs], obsvals[nobs], err, exact_mean_norm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, &A));
  PetscCall(MSGetDM(ms, &dm));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_lr", &with_lr, NULL));
  if (with_lr) {
    PetscReal obsval = 1;

    obs[0] = 0.25;
    obs[1] = 0.25;
    obs[2] = 0.75;
    obs[3] = 0.75;
    obs[4] = 0.25;
    obs[5] = 0.75;

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-obsval", &obsval, NULL));
    obsvals[0] = obsval;
    radii[0]   = 0.1;
    obsvals[1] = -1;
    radii[1]   = 0.15;
    obsvals[2] = obsval;
    radii[2]   = 0.1;

    PetscCall(MakeObservationMats(dm, nobs, 1e-4, obs, radii, obsvals, &B, &S, &f));
    PetscCall(MatCreateLRC(A, B, S, B, &Aop));
  } else Aop = A;

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, Aop, Aop));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedSkip, NULL, NULL));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &x));

  if (with_lr) {
    b = f;
  } else {
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &f));
    PetscCall(VecSetRandom(f, NULL));
    PetscCall(VecSet(f, 1));
    PetscCall(MatMult(Aop, f, b));
  }

  PetscCall(SampleCtxCreate(dm, &samplectx));
  {
    KSP ksp2;

    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp2));
    PetscCall(KSPSetOperators(ksp2, Aop, Aop));
    PetscCall(KSPSetTolerances(ksp2, 1e-12, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSolve(ksp2, b, samplectx->mean_exact));
    PetscCall(KSPDestroy(&ksp2));
  }

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetSampleCallback(pc, SampleCallbackKSP, &samplectx, SampleCtxDestroy));
  PetscCall(KSPSolve(ksp, b, x));

  {
    PetscViewer viewer;
    char        filename[512] = "solution.vtu";

    PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

    PetscCall(PetscObjectSetName((PetscObject)(x), "solution"));
    PetscCall(VecView(x, viewer));

    PetscCall(PetscObjectSetName((PetscObject)(samplectx->mean), "mean"));
    PetscCall(VecView(samplectx->mean, viewer));

    PetscCall(PetscObjectSetName((PetscObject)(samplectx->mean_exact), "mean_exact"));
    PetscCall(VecView(samplectx->mean_exact, viewer));

    PetscCall(VecAXPY(samplectx->mean, -1, samplectx->mean_exact));
    PetscCall(PetscObjectSetName((PetscObject)(samplectx->mean), "error"));
    PetscCall(VecView(samplectx->mean, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  // Mean is now error
  PetscCall(VecNorm(samplectx->mean, NORM_2, &err));
  PetscCall(VecNorm(samplectx->mean_exact, NORM_2, &exact_mean_norm));
  err /= exact_mean_norm;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Relative mean norm: %.5f\n", err));
  /* PetscCheck(PetscIsCloseAtTol(err, 0, 0.05, 0.05), MPI_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sample mean has not converged: got %.4f, expected %.4f", err, 0.f); */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  if (with_lr) {
    PetscCall(VecDestroy(&S));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&Aop));
  } else {
    PetscCall(VecDestroy(&f));
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MSDestroy(&ms));
  PetscCall(PetscFinalize());
  return 0;
}
