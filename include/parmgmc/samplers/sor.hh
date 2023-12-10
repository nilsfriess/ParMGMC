#pragma once

#include "parmgmc/grid/grid_operator.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class SORSampler {
public:
  SORSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
             PetscReal omega = 1.)
      : engine{engine}, omega{omega} {
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
    check_error(PCSORSetOmega(prec, omega));

    check_error(KSPSetOperators(
        ksp, grid_operator->get_matrix(), grid_operator->get_matrix()));

    // Extract matrix diagonal and multiply by sqrt((2-omega)/omega)
    check_error(
        MatCreateVecs(grid_operator->get_matrix(), &scaled_sqrt_diag, NULL));
    check_error(MatGetDiagonal(grid_operator->get_matrix(), scaled_sqrt_diag));
    check_error(VecSqrtAbs(scaled_sqrt_diag));
    check_error(VecScale(scaled_sqrt_diag, std::sqrt((2 - omega) / omega)));
  }

  PetscErrorCode sample(Vec sample, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    if (first_sample) {
      PetscCall(VecDuplicate(sample, &rand));
      PetscCall(VecGetLocalSize(sample, &vec_local_size));
    }

    for (std::size_t n = 0; n < n_samples; ++n) {
      PetscCall(fill_rand());
      PetscCall(VecPointwiseMult(rand, rand, scaled_sqrt_diag));
      PetscCall(KSPSolve(ksp, rand, sample));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  PetscErrorCode fill_rand() {
    PetscFunctionBeginUser;

    PetscScalar *rand_arr;
    PetscCall(VecGetArray(rand, &rand_arr));
    std::generate_n(
        rand_arr, vec_local_size, [&]() { return normal_dist(*engine); });
    PetscCall(VecRestoreArray(rand, &rand_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Engine *engine;
  std::normal_distribution<PetscReal> normal_dist;

  KSP ksp;
  Vec scaled_sqrt_diag;

  Vec rand;
  PetscInt vec_local_size;

  double omega;

  bool first_sample = true;
};
} // namespace parmgmc
