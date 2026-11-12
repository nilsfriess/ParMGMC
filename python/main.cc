/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#include <petscsys.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "petsc_caster.hh"
#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_patchgibbs.h"

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

  // Patches of a PCPATCHGIBBS sampler in CSR layout: the DOFs of patch p are
  // indices[offsets[p]:offsets[p+1]].
  m.def("PCPatchGibbsSetPatches", [](PC pc, py::array_t<PetscInt, py::array::c_style | py::array::forcecast> offsets, py::array_t<PetscInt, py::array::c_style | py::array::forcecast> indices) {
    PetscFunctionBegin;
    if (offsets.ndim() != 1 || indices.ndim() != 1 || offsets.size() < 1) throw std::invalid_argument("offsets and indices must be 1D arrays, offsets non-empty");
    const PetscInt npatches = static_cast<PetscInt>(offsets.size() - 1);
    if (offsets.at(npatches) != static_cast<PetscInt>(indices.size())) throw std::invalid_argument("offsets[-1] must equal len(indices)");
    PetscCallVoid(PCPatchGibbsSetPatches(pc, npatches, offsets.data(), indices.data()));
    PetscFunctionReturnVoid();
  });

  m.def("seed", [](unsigned long s) {
    PetscFunctionBegin;
    PetscCallVoid(ParMGMCSetSeed(s));
    PetscFunctionReturnVoid();
  });
};
