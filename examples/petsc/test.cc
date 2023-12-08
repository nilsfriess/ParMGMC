#include <iostream>

#include <memory>
#include <petscdm.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <random>

#include "parmgmc/grid/grid.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/sor.hh"

using namespace parmgmc;

PetscErrorCode assemble(Mat &mat, const Grid &grid) {
  MatStencil row_stencil;

  MatStencil col_stencil[5]; // At most 5 non-zero entries per row
  PetscScalar values[5];

  PetscFunctionBeginUser;

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(grid.get_dm(), &info));

  for (auto i = info.xs; i < info.xs + info.xm; ++i) {
    for (auto j = info.ys; j < info.ys + info.ym; ++j) {
      row_stencil.i = i;
      row_stencil.j = j;

      PetscInt k = 0;

      if (i != 0) {
        values[k] = -1;
        col_stencil[k].i = i - 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (i != info.mx - 1) {
        values[k] = -1;
        col_stencil[k].i = i + 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (j != 0) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j - 1;
        ++k;
      }

      if (j != info.my - 1) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j + 1;
        ++k;
      }

      col_stencil[k].i = i;
      col_stencil[k].j = j;
      values[k] = static_cast<PetscScalar>(k);
      ++k;

      PetscCall(MatSetValuesStencil(
          mat, 1, &row_stencil, k, col_stencil, values, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetOption(mat, MAT_SYMMETRIC, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, (char *)0, "PETSc test");

  PetscFunctionBeginUser;

  int n_vertices = 100;
  auto grid_operator =
    std::make_shared<GridOperator>(n_vertices, n_vertices, assemble);

  std::mt19937_64 engine{123};
  SORSampler sampler{grid_operator, &engine};

  Vec x;
  PetscCall(MatCreateVecs(grid_operator->get_matrix(), &x, NULL));

  sampler.sample(x, 100);

  PetscReal norm;
  VecNorm(x, NORM_2, &norm);
  
  std::cout << "||x|| = " << norm << "\n";
  
  PetscCall(VecDestroy(&x));
  PetscFinalize();
}
