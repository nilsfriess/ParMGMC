#pragma once

#include "parmgmc/grid/grid_operator.hh"

#include <algorithm>
#include <memory>
#include <random>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class CholeskySampler {
public:
  CholeskySampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine)
      : engine{engine} {
    auto check_error = [&](auto err) {
      if (err != PETSC_SUCCESS)
        throw std::runtime_error("Error while creating CholeskySampler\n");
    };

    const auto &mat = grid_operator->get_matrix();

    IS rowperm, colperm;
    check_error(MatGetOrdering(mat, MATORDERINGNATURAL, &rowperm, &colperm));

    MatFactorInfo info;
    check_error(MatFactorInfoInitialize(&info));
    info.fill = 2.0;
    info.diagonal_fill = 0;
    info.zeropivot = 0.0;

    check_error(MatGetFactor(mat, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &L));
    check_error(MatCholeskyFactorSymbolic(L, mat, rowperm, &info));
    check_error(MatCholeskyFactorNumeric(L, mat, &info));
  }

  PetscErrorCode sample(Vec sample, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    if (first_sample)
      PetscCall(VecDuplicate(sample, &rand));

    for (std::size_t n = 0; n < n_samples; ++n) {
      PetscScalar *rand_arr;
      PetscCall(VecGetArray(rand, &rand_arr));
      PetscInt vec_local_size;
      PetscCall(VecGetLocalSize(rand, &vec_local_size));
      std::generate_n(
          rand_arr, vec_local_size, [&]() { return normal_dist(*engine); });
      PetscCall(VecRestoreArray(rand, &rand_arr));

      PetscCall(MatBackwardSolve(L, rand, sample));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  Mat L;

  Engine *engine;
  std::normal_distribution<PetscReal> normal_dist;

  Vec rand;
  bool first_sample = true;
};
} // namespace parmgmc
