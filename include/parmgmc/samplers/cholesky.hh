#pragma once

#include <petscconf.h>
#if (PETSC_HAVE_MKL_CPARDISO == 1)

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/linear_operator.hh"

#include <memory>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class CholeskySampler {
public:
  CholeskySampler(const std::shared_ptr<LinearOperator> &linear_operator,
                  Engine *engine)
      : linear_operator{linear_operator}, engine{engine} {
    PetscFunctionBegin;
    // Compute cholesky decomposition of mat
    PARMGMC_INFO << "Computing Cholesky factorisation...\n";
    Timer timer;

    PARMGMC_INFO << "\t Converting matrix to right format...";
    Mat smat;
    PetscCallVoid(MatConvert(
        linear_operator->get_mat(), MATSBAIJ, MAT_INITIAL_MATRIX, &smat));
    PARMGMC_INFO_NP << "done. Took " << timer.elapsed() << " seconds.\n";
    timer.reset();

    PetscCallVoid(MatGetFactor(
        smat, MATSOLVERMKL_CPARDISO, MAT_FACTOR_CHOLESKY, &factor));

    PetscCallVoid(
        MatMkl_CPardisoSetCntl(factor, 51, 1)); // Use MPI parallel solver
    int mpi_size;
    PetscCallVoid(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    // TODO: On a cluster, it might be necessary to set these values differently
    PetscCallVoid(MatMkl_CPardisoSetCntl(
        factor, 52, mpi_size)); // Set numper of MPI ranks
    PetscCallVoid(MatMkl_CPardisoSetCntl(
        factor, 3, 1)); // Set number of OpenMP processes per rank

    // PetscCallVoid(MatMkl_CPardisoSetCntl(factor, 68, 1)); // Message level
    // info

    IS rowperm, colperm;
    PetscCallVoid(MatGetOrdering(smat, MATORDERINGNATURAL, &rowperm, &colperm));

    MatFactorInfo info;
    PetscCallVoid(MatCholeskyFactorSymbolic(factor, smat, rowperm, &info));
    PetscCallVoid(MatCholeskyFactorNumeric(factor, smat, &info));

    PetscCallVoid(ISDestroy(&rowperm));
    PetscCallVoid(ISDestroy(&colperm));
    
    PetscCallVoid(MatDestroy(&smat));

    PARMGMC_INFO << "Done. Cholesky factorisation took " << timer.elapsed()
                 << " seconds\n";

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, const Vec rhs) {
    PetscFunctionBeginUser;

    if (v == nullptr) {
      PetscCall(VecDuplicate(rhs, &v));
      PetscCall(VecDuplicate(rhs, &r));
    }

    PetscCall(fill_vec_rand(r, *engine));
    PetscCall(MatForwardSolve(factor, rhs, v));

    PetscCall(VecAXPY(v, 1., r));

    PetscCall(MatBackwardSolve(factor, v, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

~CholeskySampler() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&r));
    PetscCallVoid(VecDestroy(&v));

    PetscCallVoid(MatDestroy(&factor));

    PetscFunctionReturnVoid();
}

private:
  std::shared_ptr<LinearOperator> linear_operator;
  Engine *engine;

  Mat factor;

  Vec v = nullptr;
  Vec r = nullptr;
};
} // namespace parmgmc
#endif