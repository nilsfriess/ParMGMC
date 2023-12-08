#pragma once

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <stdexcept>

namespace parmgmc {
/*
  Wrapper around PETSc DMDA grid, representing a 2d structured (i.e.,
  topologically cartesian) grid.
*/
class Grid {
public:
  Grid(PetscInt global_x, PetscInt global_y)
      : global_x{global_x}, global_y{global_y} {
    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    auto err = DMDACreate2d(PETSC_COMM_WORLD,
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
                            &dmgrid);
    if (err != PETSC_SUCCESS)
      throw std::runtime_error(
          "parmgmc::Grid(): An error occured when trying to create the grid.");

    err = DMSetUp(dmgrid);
    if (err != PETSC_SUCCESS)
      throw std::runtime_error(
          "parmgmc::Grid(): An error occured when trying to create the grid.");
  }

  PetscErrorCode initialise_matrix(Mat &mat) {
    PetscFunctionBeginUser;
    PetscCall(DMCreateMatrix(dmgrid, &mat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt get_nx_vertices() const { return global_x; }
  PetscInt get_ny_vertices() const { return global_y; }

  const DM &get_dm() const { return dmgrid; }

private:
  DM dmgrid;

  PetscInt global_x;
  PetscInt global_y;
};
} // namespace parmgmc
