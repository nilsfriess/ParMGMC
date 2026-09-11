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
#include "parmgmc/snes/snes_poissongibbs.h"

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
  m.def("SNESPoissonSetAppCtx", [](SNES snes, Vec event_counts, Vec time_intervals, PetscScalar beta, Mat Q_prec, Mat B_meas) {
    PoissonGibbsCtx *ctx;

    PetscFunctionBegin;
    PetscCallVoid(PetscNew(&ctx));
    PetscCallVoid(PetscObjectReference((PetscObject)event_counts));
    PetscCallVoid(PetscObjectReference((PetscObject)Q_prec));
    PetscCallVoid(PetscObjectReference((PetscObject)B_meas));

    ctx->event_counts = event_counts;
    ctx->Q_prec       = Q_prec;
    ctx->B_meas       = B_meas;
    // nu = - beta - log(t_k)
    PetscCallVoid(VecDuplicate(time_intervals, &ctx->nu));
    PetscCallVoid(VecCopy(time_intervals, ctx->nu));
    PetscCallVoid(VecLog(ctx->nu));
    PetscCallVoid(VecShift(ctx->nu, beta));
    PetscCallVoid(VecScale(ctx->nu, -1.0));
    PetscCallVoid(SNESSetApplicationContext(snes, ctx));
    PetscFunctionReturnVoid();
  });
};
