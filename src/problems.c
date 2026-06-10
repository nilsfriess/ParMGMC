/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include "parmgmc/problems.h"

#include <petscdmda.h>
#include <petscerror.h>

PetscErrorCode MatAssembleShiftedLaplaceFD(DM dm, PetscReal kappa, Mat mat)
{
  MatStencil row, cols[5];
  PetscReal  hinv2, vals[5];
  PetscInt   mx, my, xm, ym, xs, ys;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(dm, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(dm, &xs, &ys, 0, &xm, &ym, 0));

  hinv2 = 1. / ((mx - 1) * (mx - 1));
  for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
      PetscReal diag = kappa * kappa;
      PetscInt  n    = 0;

      row.j = j;
      row.i = i;

      if (j > 0) {
        cols[n].j = j - 1;
        cols[n].i = i;
        vals[n]   = -hinv2;
        diag += hinv2;
        n++;
      }
      if (i > 0) {
        cols[n].j = j;
        cols[n].i = i - 1;
        vals[n]   = -hinv2;
        diag += hinv2;
        n++;
      }
      if (j < my - 1) {
        cols[n].j = j + 1;
        cols[n].i = i;
        vals[n]   = -hinv2;
        diag += hinv2;
        n++;
      }
      if (i < mx - 1) {
        cols[n].j = j;
        cols[n].i = i + 1;
        vals[n]   = -hinv2;
        diag += hinv2;
        n++;
      }

      cols[n].j = j;
      cols[n].i = i;
      vals[n]   = diag;
      n++;

      PetscCall(MatSetValuesStencil(mat, 1, &row, n, cols, vals, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(mat, MAT_SPD, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
