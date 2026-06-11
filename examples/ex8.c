/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Provides the full code for the first code listing in the Algebraic MGMC paper
 *
 */

// Gibbs sampler
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -pc_type sorgibbs

// MGMC sampler, set up manually using GAMG
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -ksp_max_it 100 -pc_type gamg -prefix_push mg_levels_ -ksp_type richardson -ksp_max_it 1 -pc_type sorgibbs -prefix_pop -prefix_push mg_coarse_ -ksp_type richardson -ksp_max_it 1 -pc_type cholsampler -prefix_pop

// MGMC sampler, set up using the GAMGMC PC
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -ksp_type richardson -ksp_max_it 100 -pc_type gamgmc

#include <parmgmc/problems.h>
#include <parmgmc/parmgmc.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  DM  da;
  Mat A;
  KSP ksp;
  Vec y, f;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 256, 256, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 0.0001, A));
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(DMCreateGlobalVector(da, &y));
  PetscCall(VecDuplicate(y, &f));
  PetscCall(KSPSolve(ksp, f, y));
  PetscCall(DMDestroy(&da));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&f));
  PetscCall(ParMGMCFinalize());
  PetscCall(PetscFinalize());
}
