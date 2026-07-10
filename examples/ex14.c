/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Tests Poisson Gibbs sampler
 */

// ex14 -ksp_type richardson -pc_type gamgmc -pc_gamgmc_mg_type mg -gamgmc_mg_coarse_pc_type sorgibbs -box_faces 2 -dm_refine_hierarchy 2 -with_lr -nburnin 500 -ksp_max_it 2000 -tol 0.10

#include <parmgmc/mc_sor.h>
#include <parmgmc/ms.h>
#include <parmgmc/obs.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/pc/pc_poissongibbs.h>
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

int main(int argc, char *argv[])
{
  DM             dm;
  Mat            A;
  Vec            x, f;
  KSP            ksp;
  PC             pc;
  MS             ms;
  PetscInt       nobs = 4;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());
  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, &A));
  PetscCall(MSGetDM(ms, &dm));
  PetscInt ndof,m;
  PetscCall(MatGetSize(A,&ndof,&m));
  printf("Matrix size = %d x %d\n",ndof,m);
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  // Observations
  PoissonGibbsCtx ctx;
  PetscCall(VecCreate(MPI_COMM_WORLD, &ctx.event_counts));
  PetscCall(VecSetSizes(ctx.event_counts, PETSC_DECIDE, nobs));
  PetscCall(VecSetFromOptions(ctx.event_counts));
  PetscCall(VecSet(ctx.event_counts,2));
  PetscCall(VecCreate(MPI_COMM_WORLD, &ctx.nu));
  PetscCall(VecSetSizes(ctx.nu, PETSC_DECIDE, nobs));
  PetscCall(VecSetFromOptions(ctx.nu));
  PetscCall(MatCreateSeqAIJ(MPI_COMM_WORLD,ndof,nobs,nobs,NULL,&ctx.B));
  for (PetscInt j=0; j<nobs; ++j) {
    PetscInt i = (PetscInt) ndof*(1.0*j/nobs);
    PetscCall(MatSetValue(ctx.B, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(ctx.B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx.B, MAT_FINAL_ASSEMBLY));
  // Create vectors
  PetscCall(VecCreate(MPI_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, ndof));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreate(MPI_COMM_WORLD, &f));
  PetscCall(VecSetSizes(f, PETSC_DECIDE, ndof));
  PetscCall(VecSetFromOptions(f));

#ifdef PARMGMC_PETSC_KSP_DMACTIVE_3ARG
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
#else
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
#endif
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedSkip, NULL, NULL));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &x));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPSolve(ksp, f, x));

  {
    PetscViewer viewer;
    char        filename[512] = "solution.vtu";

    PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
    PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

    PetscCall(PetscObjectSetName((PetscObject)(x), "solution"));
    PetscCall(VecView(x, viewer));

    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&ctx.event_counts));
  PetscCall(VecDestroy(&ctx.nu));
  PetscCall(MatDestroy(&ctx.B));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MSDestroy(&ms));
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
  return 0;
}