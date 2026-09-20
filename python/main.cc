/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include <petscsys.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "petsc_caster.hh"
#include "parmgmc/parmgmc.h"
#include "parmgmc/poisson.h"
#include "parmgmc/snes/snes_poissonmala.h"

namespace py = pybind11;

std::function<void(PetscInt, Vec)> *cb;

PYBIND11_MODULE(pymgmc, m)
{
  PetscCallVoid(ParMGMCInitialize());

  // Register cleanup so ParMGMCFinalize is called when the module is unloaded.
  m.add_object("_cleanup", py::capsule([]() { PetscCallVoid(ParMGMCFinalize()); }));

  m.def("PCSetSampleCallback", [&](PC pc, std::function<void(PetscInt, Vec)> &cb_) {
    PetscFunctionBegin;
    cb = new std::function<void(PetscInt, Vec)>(cb_); // TODO: This leaks memory but just copying didn't work and caused a segfault during Python runtime shutdown.

    PetscCallVoid(PCSetSampleCallback(
      pc,
      [](PetscInt i, Vec v, void *) {
        PetscFunctionBegin;
        (*cb)(i, v);
        PetscFunctionReturn(PETSC_SUCCESS);
      },
      nullptr, nullptr));
    PetscFunctionReturnVoid();
  });

  m.def("seed", [](unsigned long s) {
    PetscRandom pr;

    PetscFunctionBegin;
    PetscCallVoid(ParMGMCGetPetscRandom(&pr));
    PetscCallVoid(PetscRandomSetSeed(pr, s));
    PetscCallVoid(PetscRandomSeed(pr));
    PetscCallVoid(PetscRandomDestroy(&pr)); // release the ref given by ParMGMCGetPetscRandom
    PetscFunctionReturnVoid();
  });
  m.def("SetPoissonCtx", [](SNES snes, Vec event_counts, Vec measurement_intervals, PetscScalar beta, Mat Q_prec, Mat B_meas) {
    PoissonCtx *ctx;
    Vec         marker; // marker which is attached to SNES to signal that user context has been set

    PetscFunctionBegin;
    // Check whether context has already been attached to this SNES
    // We do not want to set the user context twice since currently the snes setup routine uses information
    // from this context, and this would become stale if the context is changed later
    PetscCallVoid(PetscObjectQuery((PetscObject)snes, "ctx_set", (PetscObject *)&marker));
    if (marker) {
      PetscCallVoid(PetscErrorPrintf("User context for SNES already set\n"));
      MPI_Abort(PETSC_COMM_WORLD, PETSC_ERR_COR);
    } else {
      PetscCallVoid(VecCreateSeq(PETSC_COMM_SELF, 1, &marker));
      PetscCallVoid(PetscObjectCompose((PetscObject)snes, "ctx_set", (PetscObject)marker));
      PetscCallVoid(VecDestroy(&marker));
    }

    PetscCallVoid(PetscNew(&ctx));
    PetscCallVoid(PetscObjectReference((PetscObject)event_counts));
    PetscCallVoid(PetscObjectReference((PetscObject)Q_prec));
    PetscCallVoid(PetscObjectReference((PetscObject)B_meas));

    ctx->event_counts = event_counts;
    ctx->Q_prec       = Q_prec;
    ctx->B_meas       = B_meas;
    // nu = - beta - log(t_k)
    PetscCallVoid(VecDuplicate(measurement_intervals, &ctx->nu));
    PetscCallVoid(VecCopy(measurement_intervals, ctx->nu));
    PetscCallVoid(VecLog(ctx->nu));
    PetscCallVoid(VecShift(ctx->nu, beta));
    PetscCallVoid(VecScale(ctx->nu, -1.0));
    PetscCallVoid(SNESSetApplicationContext(snes, ctx));
    PetscFunctionReturnVoid();
  });
  m.def("GetMALAAcceptanceRate", [](SNES snes) -> double {
    PetscScalar acceptance_rate;
    PetscCallAbort(PETSC_COMM_WORLD, SNESPoissonMALAGetAcceptanceStatistics(snes, NULL, NULL, &acceptance_rate));
    return (double)acceptance_rate;
  });
};
