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
// All samplers run with the low-rank update (-with_lr) except the Cholesky
// reference: a direct factorisation of A + B Sigma^-1 B^T would assemble the
// dense low-rank term, which is exactly what the LR machinery avoids, so that
// combination is intentionally omitted.  -nburnin discards the initial
// transient (the chain starts at x = 0); -tol is sized to the converged
// sample-mean error at the given -ksp_max_it with some margin.
//
// -box_faces 2 keeps the coarsest mesh (8 cells) large enough to distribute
// across the MPI ranks: the geometric hierarchy is built by refining the
// distributed coarse mesh, so a coarse mesh with fewer cells than ranks would
// leave ranks empty.  These tolerances are calibrated for both np 1 and np 4.

// Geometric MGMC, low-rank update, SOR-Gibbs coarse sampler
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_mg_coarse_pc_type sorgibbs -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 500 -ksp_max_it 2000 -tol 0.10 %opts -ksp_norm_type none -ksp_convergence_test skip

// Geometric MGMC, low-rank update, MulticolorGibbs coarse sampler
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_mg_coarse_pc_type mcgibbs -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 500 -ksp_max_it 2000 -tol 0.10 %opts -ksp_norm_type none -ksp_convergence_test skip

// Geometric MGMC, low-rank update, Cholesky coarse sampler (coarse grid only -- cheap)
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_mg_coarse_pc_type cholsampler -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 500 -ksp_max_it 2000 -tol 0.10 %opts -ksp_norm_type none -ksp_convergence_test skip

// Geometric MGMC, NO low-rank update, SOR-Gibbs coarse sampler.  Without the LR
// term (which conditions the operator) the kappa=1 Matern system mixes too slowly
// for the weaker parallel smoother, so use a larger kappa to keep it diagonally
// dominant; np 1 and np 4 then agree.
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_mg_coarse_pc_type sorgibbs -box_faces 2 -dm_refine_hierarchy 2 -matern_kappa 10 -nburnin 500 -ksp_max_it 2000 -tol 0.05 %opts -ksp_norm_type none -ksp_convergence_test skip

// Algebraic MGMC (GAMG), low-rank update -- aggressive coarsening needs more smoothing to mix
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type gamg -gamgmc_mg_levels_ksp_max_it 10 -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 500 -ksp_max_it 5000 -tol 0.05 %opts -ksp_norm_type none -ksp_convergence_test skip

// MulticolorGibbs sampler with low-rank update
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type mcgibbs -pc_mcgibbs_symmetric -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 2000 -ksp_max_it 20000 -tol 0.05 %opts -ksp_norm_type none -ksp_convergence_test skip

// SOR-Gibbs sampler with low-rank update
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type sorgibbs -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 2000 -ksp_max_it 20000 -tol 0.05 %opts -ksp_norm_type none -ksp_convergence_test skip

// Cholesky sampler (exact reference, no low-rank update -- see note above)
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type cholsampler -box_faces 2 -dm_refine 2 -nburnin 200 -ksp_max_it 5000 -tol 0.05 %opts -ksp_norm_type none -ksp_convergence_test skip
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
  PetscInt    nburnin; /* number of initial samples to discard before averaging */
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
  SampleCtx *sctx    = ctx;
  Vec        mean    = (*sctx)->mean;
  PetscInt   nburnin = (*sctx)->nburnin;
  PetscInt   k;

  PetscFunctionBeginUser;
  /* Discard the first nburnin samples: the chain starts from x = 0, so early
     samples are far from stationarity and would skew the running mean. */
  if (it < nburnin) PetscFunctionReturn(PETSC_SUCCESS);
  k = it - nburnin;
  PetscCall(VecScale(mean, k));
  PetscCall(VecAXPY(mean, 1., y));
  PetscCall(VecScale(mean, 1. / (k + 1)));
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
  PetscInt       nburnin = 0;
  PetscReal      tol     = 0.1;
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
  PetscCall(KSPSetDM(ksp, dm));
#if PETSC_VERSION_GT(3, 24, 5)
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
#else
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
#endif
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
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nburnin", &nburnin, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  samplectx->nburnin = nburnin;
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
  PetscCheck(PetscRealPart(err) <= tol, MPI_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sample mean has not converged: got %.4f, expected <= %.4f", (double)PetscRealPart(err), (double)tol);

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
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
  return 0;
}
