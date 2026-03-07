/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Tests the multicolour SOR method.
 */

/**************************** Test specification ****************************/
// Omega = 1
// RUN1: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -ksp_error_if_not_converged -dm_refine 2

// Omega = 1.2
// RUN1: %cc %s -o %t %flags && %mpirun -np %NP %t -mc_sor_omega 1.2 -ksp_type richardson -ksp_error_if_not_converged -dm_refine 2

// Standalone SOR with low-rank update
// RUN1: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type richardson -dm_refine 3 -with_lr -ksp_error_if_not_converged

// FGMRES + SOR with low-rank update
// RUN: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type fgmres -dm_refine 4 -with_lr %opts

// FGMRES + SSOR
// RUN: %cc %s -o %t %flags -g && %mpirun -np %NP %t -ksp_type fgmres -dm_refine 4 -sor_symmetric %opts
/****************************************************************************/

#include <parmgmc/mc_sor.h>
#include <parmgmc/ms.h>
#include <parmgmc/obs.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/problems.h>

#include <petsc.h>
#include <petsc/private/kspimpl.h>
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

typedef struct {
  MCSOR mc;
} *AppCtx;

static PetscErrorCode apply(PC pc, Vec x, Vec y)
{
  AppCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MCSORApply(ctx->mc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM             dm;
  Mat            A, B, Aop;
  Vec            S, b, x, f;
  KSP            ksp;
  PC             pc;
  MS             ms;
  AppCtx         appctx;
  PetscBool      with_lr = PETSC_FALSE, sor_symmetric = PETSC_FALSE;
  const PetscInt nobs = 3;
  PetscScalar    obs[3 * nobs], radii[nobs], obsvals[nobs];

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
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
    obsvals[1] = obsval;
    radii[1]   = 0.1;
    obsvals[2] = obsval;
    radii[2]   = 0.1;

    PetscCall(MakeObservationMats(dm, nobs, 1e-3, obs, radii, obsvals, &B, &S, &f));
    PetscCall(MatCreateLRC(A, B, S, B, &Aop));
  } else Aop = A;

  PetscCall(PetscNew(&appctx));
  PetscCall(MCSORCreate(Aop, &appctx->mc));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-sor_symmetric", &sor_symmetric, NULL));
  if (sor_symmetric) PetscCall(MCSORSetSweepType(appctx->mc, SOR_SYMMETRIC_SWEEP));
  PetscCall(MCSORSetUp(appctx->mc));

  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, Aop, Aop));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetApply(pc, apply));
  PetscCall(PCShellSetContext(pc, appctx));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &x));

  if (with_lr) {
    b = f;
  } else {
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &f));
    PetscCall(VecSetRandom(f, NULL));
    PetscCall(MatMult(Aop, f, b));
    PetscCall(VecDestroy(&f));
  }

  PetscCall(KSPSolve(ksp, b, x));

  {
    PetscViewer viewer;
    char        filename[512] = "solution.vtu";

    PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

    PetscCall(PetscObjectSetName((PetscObject)(x), "solution"));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(MCSORDestroy(&appctx->mc));
  PetscCall(PetscFree(appctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  if (with_lr) {
    PetscCall(VecDestroy(&S));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&Aop));
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MSDestroy(&ms));
  PetscCall(PetscFinalize());
  return 0;
}
