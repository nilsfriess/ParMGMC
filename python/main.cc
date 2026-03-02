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

namespace py = pybind11;

std::function<void(PetscInt, Vec)> *cb;

PYBIND11_MODULE(pymgmc, m)
{
  PetscCallVoid(ParMGMCInitialize());

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
};
