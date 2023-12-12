#pragma once

#include <iostream>

#include "parmgmc/grid/grid.hh"

#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>

namespace parmgmc {
struct GridOperator {
  /* Constructs a GridOperator instance for a 2d structured grid of size
   * global_x*global_y and a matrix representing an operator defined on that
   * grid. The parameter mat_assembler must be a function with signature `void
   * mat_assembler(Mat &, const Grid &)` that assembles the matrix. Note that
   * the nonzero pattern is alredy set before this function is called, so only
   * the values have to be set (e.g., using PETSc's MatSetValuesStencil).
   */
  template <class MatAssembler>
  GridOperator(PetscInt global_x, PetscInt global_y,
               MatAssembler &&mat_assembler) {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    PetscFunctionBeginUser;
    call(DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,
                      DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      global_x,
                      global_y,
                      PETSC_DECIDE,
                      PETSC_DECIDE,
                      dof_per_node,
                      stencil_width,
                      NULL,
                      NULL,
                      &dm));
    call(DMSetUp(dm));

    // Allocate memory for matrix and initialise non-zero pattern
    call(DMCreateMatrix(dm, &mat));

    // Call provided assembly functor to fill matrix
    call(mat_assembler(mat, dm));

    PetscFunctionReturnVoid();
  }

  GridOperator() = default;

  ~GridOperator() {
    MatDestroy(&mat);
    DMDestroy(&dm);
  }

  DM dm;
  Mat mat;
};
} // namespace parmgmc
