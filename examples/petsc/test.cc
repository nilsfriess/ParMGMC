#include <iostream>
#include <random>

#include <pcg_random.hpp>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/petsc_helper.hh"
#include "parmgmc/samplers/cholesky.hh"
#include "parmgmc/samplers/multigrid.hh"
#include "parmgmc/samplers/sample_chain.hh"
#include "parmgmc/samplers/sor.hh"

using namespace parmgmc;

PetscErrorCode assemble(Mat mat, DM dm) {
  MatStencil row_stencil;

  MatStencil col_stencil[5]; // At most 5 non-zero entries per row
  PetscScalar values[5];

  PetscFunctionBeginUser;

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(dm, &info));

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
  PetscHelper helper(&argc, &argv);
  PetscFunctionBeginUser;

  int n_vertices = (1 << 8) + 1;
  int n_levels = 3;
  PetscBool found;

  PetscOptionsGetInt(NULL, NULL, "-n_vertices", &n_vertices, &found);
  PetscOptionsGetInt(NULL, NULL, "-n_levels", &n_levels, &found);

  auto grid_operator =
      std::make_shared<GridOperator>(n_vertices, n_vertices, assemble);

  pcg32 engine;
  pcg_extras::seed_seq_from<std::random_device> seed_source;

  engine.seed(seed_source);
  // engine.seed(0xCAFEBEEF);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  engine.set_stream(rank);

  Vec sample;
  Vec rhs;
  PetscCall(MatCreateVecs(grid_operator->mat, &sample, NULL));
  PetscCall(MatCreateVecs(grid_operator->mat, &rhs, NULL));

  PetscInt size;
  PetscCall(VecGetLocalSize(rhs, &size));
  PetscCall(fill_vec_rand(rhs, size, engine));

  // SampleChain<MultigridSampler<pcg32>> chain{
  //     grid_operator, &engine, n_levels, assemble};
  SampleChain<SORSampler<pcg32>> chain{grid_operator, &engine, 1.9852};

  const std::size_t n_burnin = 0;
  PetscInt n_samples = 1;
  PetscOptionsGetInt(NULL, NULL, "-n_samples", &n_samples, &found);

  chain.sample(sample, rhs, n_burnin);
  chain.enable_est_mean_online();

  Vec mean;
  VecDuplicate(sample, &mean);

  Vec prec_x_mean;
  VecDuplicate(mean, &prec_x_mean);

  PetscScalar err;
  PetscReal rhs_norm;
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));

  for (PetscInt n = 0; n < n_samples; ++n) {
    chain.sample(sample, rhs);

    chain.get_mean(mean);

    PetscCall(MatMult(grid_operator->mat, mean, prec_x_mean));
    PetscCall(VecAXPY(prec_x_mean, -1., rhs));
    PetscCall(VecNorm(prec_x_mean, NORM_2, &err));

    if (rhs_norm < 1e-18)
      std::cout << err << "\n";
    else
      std::cout << err << ", " << err / rhs_norm << "\n";
  }

  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&prec_x_mean));

  PetscFunctionReturn(PETSC_SUCCESS);
}
