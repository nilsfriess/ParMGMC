#include <iostream>
#include <memory>
#include <random>

#include <pcg_random.hpp>

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

#include "parmgmc/grid/grid.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/cholesky.hh"
#include "parmgmc/samplers/sample_chain.hh"
#include "parmgmc/samplers/sor.hh"

using namespace parmgmc;

PetscErrorCode assemble(Mat &mat, const Grid &grid) {
  MatStencil row_stencil;

  MatStencil col_stencil[5]; // At most 5 non-zero entries per row
  PetscScalar values[5];

  PetscFunctionBeginUser;

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(grid.get_dm(), &info));

  PetscReal noise_var = 1e-4;

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
      values[k] = static_cast<PetscScalar>(k) + noise_var;
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

  int n_vertices = 10;
  auto grid_operator =
      std::make_shared<GridOperator>(n_vertices, n_vertices, assemble);

  pcg32 engine;
  pcg_extras::seed_seq_from<std::random_device> seed_source;

  engine.seed(seed_source);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  engine.set_stream(rank);

  Vec sample;
  PetscCall(MatCreateVecs(grid_operator->get_matrix(), &sample, NULL));

  SampleChain<SORSampler<pcg32>> chain{grid_operator, &engine, 1.9852};

  const std::size_t n_burnin = 100;
  const std::size_t n_samples = 1000;

  PetscReal norm;

  chain.disable_save();
  chain.sample(sample, n_burnin);
  chain.enable_save();

  Vec mean;
  VecDuplicate(sample, &mean);

  for (std::size_t n = 0; n < n_samples; ++n) {
    chain.sample(sample);

    chain.get_mean(mean);

    VecNorm(mean, NORM_2, &norm);
    std::cout << norm << "\n";
  }

  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&mean));
  PetscFinalize();
}
