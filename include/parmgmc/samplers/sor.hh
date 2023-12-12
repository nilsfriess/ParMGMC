#pragma once

#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/sor_preconditioner.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

#include <petscpctypes.h>
#include <petscsys.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class SORSampler {
public:
  SORSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine, PetscReal omega = 1.) {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    PetscFunctionBeginUser;

    call(KSPCreate(MPI_COMM_WORLD, &ksp));
    call(KSPSetType(ksp, KSPRICHARDSON));
    call(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    call(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    call(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));

    PC pc;
    call(KSPGetPC(ksp, &pc));
    call(PCSetType(pc, PCSHELL));

    auto *context =
        new SORRichardsonContext<Engine>(engine, grid_operator->mat, omega);

    call(PCShellSetContext(pc, context));
    call(PCShellSetApplyRichardson(pc, sor_pc_richardson_apply<Engine>));
    call(PCShellSetDestroy(pc, sor_pc_richardson_destroy<Engine>));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    for (std::size_t n = 0; n < n_samples; ++n)
      PetscCall(KSPSolve(ksp, rhs, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~SORSampler() {
    KSPDestroy(&ksp);
  }

private:
  KSP ksp;
};
} // namespace parmgmc
