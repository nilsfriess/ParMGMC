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

// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -snes_type poissongibbs -snes_poissongibbs_its 1 -snes_view :snes_view.txt

/*
Command line options:

  ./build/examples/ex14 \
      -snes_type poissongibbs \
      -snes_poissongibbs_its 1 \
      -snes_view :snes_view.txt

*/

#include <parmgmc/ms.h>
#include <parmgmc/obs.h>
#include <parmgmc/parmgmc.h>
#include <parmgmc/snes/snes_poissongibbs.h>
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
#include <petscsnes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>

PetscErrorCode initialise_ctx(Mat Q_prec, PetscInt nobs, PoissonGibbsCtx* ctx) {
  PetscInt ndof, m;

  PetscFunctionBeginUser;  
  PetscCall(MatGetSize(Q_prec,&ndof,&m));
// Observations
  PetscCall(VecCreate(MPI_COMM_WORLD, &ctx->event_counts));
  PetscCall(VecSetSizes(ctx->event_counts, PETSC_DECIDE, nobs));
  PetscCall(VecSetFromOptions(ctx->event_counts));
  PetscCall(VecSet(ctx->event_counts,2.0));
  PetscCall(VecCreate(MPI_COMM_WORLD, &ctx->nu));
  PetscCall(VecSetSizes(ctx->nu, PETSC_DECIDE, nobs));
  PetscCall(VecSetFromOptions(ctx->nu));
  PetscCall(VecSet(ctx->nu,0));
  PetscCall(MatCreateSeqAIJ(MPI_COMM_WORLD,ndof,nobs,nobs,NULL,&ctx->B_meas));
  for (PetscInt j=0; j<nobs; ++j) {
    PetscInt i = (PetscInt) ndof*(1.0*j/nobs);
    PetscCall(MatSetValue(ctx->B_meas, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(ctx->B_meas, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->B_meas, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreate(MPI_COMM_WORLD, &ctx->f_rhs));
  PetscCall(VecSetSizes(ctx->f_rhs, PETSC_DECIDE, ndof));
  PetscCall(VecSetFromOptions(ctx->f_rhs));
  PetscCall(VecZeroEntries(ctx->f_rhs));
  ctx->Q_prec = Q_prec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM             dm;
  Mat            Q_prec;
  Vec            y;
  SNES           snes;
  MS             ms;
  PetscInt       nobs = 4;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());
  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, &Q_prec));
  PetscCall(MSGetDM(ms, &dm));
  PetscInt ndof, m;
  PetscCall(MatGetSize(Q_prec,&ndof,&m));
  printf("Matrix size = %d x %d\n",ndof,m);
  PetscCall(SNESCreate(MPI_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  
  PoissonGibbsCtx ctx;
  initialise_ctx(Q_prec,nobs,&ctx);

  // Create sample vector
  PetscCall(VecCreate(MPI_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, ndof));
  PetscCall(VecSetFromOptions(y));
  
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));
  PetscCall(DMCreateGlobalVector(dm, &y));
  PetscCall(SNESSetApplicationContext(snes, &ctx));
  
  PetscViewer viewer;
  char        filename[512] = "solution.vtu";

  PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
  PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

  PetscInt n_samples = 64;
  for (int k=0;k<n_samples;++k)
  {
    PetscCall(SNESSolve(snes, NULL, y));
    char field_label[100];
    sprintf (field_label, "sample_%03d",k);
    PetscCall(PetscObjectSetName((PetscObject)(y), field_label));
    PetscCall(VecView(y, viewer));
  }
  
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ctx.f_rhs));
  PetscCall(VecDestroy(&ctx.event_counts));
  PetscCall(VecDestroy(&ctx.nu));
  PetscCall(MatDestroy(&ctx.Q_prec));
  PetscCall(MatDestroy(&ctx.B_meas));
  PetscCall(SNESDestroy(&snes));
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
  return 0;
}