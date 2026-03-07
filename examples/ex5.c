#include "parmgmc/mc_sor.h"
#include "parmgmc/parmgmc.h"
#include "parmgmc/problems.h"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Checks that a symmetric Gauss-Seidel sweep is the same as a forward sweep,
 *  followed by a backward sweep.
 */

/**************************** Test specification ****************************/
// MulticolorGibbs with default omega
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t
/****************************************************************************/

int main(int argc, char *argv[])
{
  Mat         A;
  DM          da;
  MCSOR       mc;
  Vec         x, y, b;
  PetscScalar err;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatAssembleShiftedLaplaceFD(da, 1, A));

  PetscCall(MCSORCreate(A, &mc));
  PetscCall(MCSORSetUp(mc));

  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(DMCreateGlobalVector(da, &y));
  PetscCall(DMCreateGlobalVector(da, &b));
  PetscCall(VecSetRandom(b, NULL));
  PetscCall(VecSetRandom(x, NULL));
  PetscCall(VecCopy(x, y));

  PetscCall(MCSORSetSweepType(mc, SOR_FORWARD_SWEEP));
  PetscCall(MCSORApply(mc, b, x));
  PetscCall(MCSORSetSweepType(mc, SOR_BACKWARD_SWEEP));
  PetscCall(MCSORApply(mc, b, x));

  PetscCall(MCSORSetSweepType(mc, SOR_SYMMETRIC_SWEEP));
  PetscCall(MCSORApply(mc, b, y));

  PetscCall(VecAXPY(x, -1, y));
  PetscCall(VecNorm(x, NORM_2, &err));
  PetscCheck(PetscAbs(err) < 1e-15, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Forward+Backward sweep is not the same as symmetric sweep");

  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MCSORDestroy(&mc));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
}
