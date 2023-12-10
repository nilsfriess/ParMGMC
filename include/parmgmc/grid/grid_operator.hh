#pragma once

#include "parmgmc/grid/grid.hh"

#include <petscmat.h>

namespace parmgmc {
class GridOperator {
public:
  /* Constructs a GridOperator instance for a 2d structured grid of size
   * global_x*global_y and a matrix representing an operator defined on that
   * grid. The parameter mat_assembler must be a function with signature `void
   * mat_assembler(Mat &, const Grid &)` that assembles the matrix. Note that
   * the nonzero pattern is alredy set before this function is called, so only
   * the values have to be set (e.g., using PETSc's MatSetValuesStencil).
   */
  template <class MatAssembler>
  GridOperator(PetscInt global_x, PetscInt global_y,
               MatAssembler &&mat_assembler)
      : grid{global_x, global_y} {
    grid.initialise_matrix(mat);
    mat_assembler(mat, grid);
  }

  const Mat& get_matrix() const { return mat; }
  const Grid &get_grid() const { return grid; }

private:
  Grid grid;
  Mat mat;
};
} // namespace parmgmc
