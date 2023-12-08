#pragma once

#include "parmgmc/grid/grid_operator.hh"

#include <algorithm>
#include <memory>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <random>
#include <stdexcept>

namespace parmgmc {
template <class Engine> class SORSampler {
public:
  SORSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine)
      : grid_operator{grid_operator}, engine{engine} {
    auto check_error = [&](auto err) {
      if (err != PETSC_SUCCESS)
        throw std::runtime_error("Error while creating SORSampler\n");
    };

    /* We create a full Krylov solver since PETSc does not expose a SOR solver
     * directly, only as a preconditioner for a Krylov solver. However, we can
     * tell PETSc to only run the preconditioner, not the full solver. */
    check_error(KSPCreate(MPI_COMM_WORLD, &ksp));
    check_error(KSPSetType(ksp, KSPPREONLY));

    PC prec;
    check_error(KSPGetPC(ksp, &prec));
    check_error(PCSetType(prec, PCSOR));

    check_error(KSPSetOperators(
        ksp, grid_operator->get_matrix(), grid_operator->get_matrix()));

    check_error(MatCreateVecs(grid_operator->get_matrix(), &sqrt_diag, NULL));
    check_error(MatGetDiagonal(grid_operator->get_matrix(), sqrt_diag));
    check_error(VecSqrtAbs(sqrt_diag));
  }

  void sample(Vec sample, std::size_t n_steps = 1) {
    if (first_sample)
      VecDuplicate(sample, &rand);

    for (std::size_t n = 0; n < n_steps; ++n) {
      PetscScalar *rand_arr;
      VecGetArray(rand, &rand_arr);
      PetscInt vec_local_size;
      VecGetLocalSize(rand, &vec_local_size);
      std::generate_n(
          rand_arr, vec_local_size, [&]() { return normal_dist(*engine); });
      VecRestoreArray(rand, &rand_arr);
      VecPointwiseMult(rand, rand, sqrt_diag);

      KSPSolve(ksp, rand, sample);
    }
  }

private:
  std::shared_ptr<GridOperator> grid_operator;

  Engine *engine;
  std::normal_distribution<PetscReal> normal_dist;

  KSP ksp;
  Vec sqrt_diag;

  Vec rand;
  bool first_sample = true;
};
} // namespace parmgmc
