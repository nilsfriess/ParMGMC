#pragma once

#include <iostream>
#include <mpi.h>

#include <petscerror.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

#include <algorithm>
#include <random>

namespace parmgmc {
template <class Engine> struct SORRichardsonContext {
  SORRichardsonContext(Engine *engine, Mat mat, PetscReal omega)
      : engine{engine} {
    auto call = [&](auto err) { PetscCallAbort(MPI_COMM_WORLD, err); };

    call(MatCreateVecs(mat, &sqrt_diag, NULL));
    call(MatGetDiagonal(mat, sqrt_diag));
    call(VecSqrtAbs(sqrt_diag));
    call(VecScale(sqrt_diag, std::sqrt((2 - omega) / omega)));

    // Setup Gauss-Seidel preconditioner
    call(PCCreate(MPI_COMM_WORLD, &pc));
    call(PCSetType(pc, PCSOR));
    call(PCSetOperators(pc, mat, mat));

    call(VecDuplicate(sqrt_diag, &z));

    call(VecGetLocalSize(sqrt_diag, &vec_size));
  }

  ~SORRichardsonContext() {
    VecDestroy(&sqrt_diag);
    VecDestroy(&z);

    PCDestroy(&pc);
  }

  Engine *engine;
  std::normal_distribution<PetscReal> dist;
  Vec sqrt_diag;
  Vec z;
  PetscInt vec_size;

  PC pc;
};

template <class Engine>
inline PetscErrorCode
sor_pc_richardson_apply(PC pc, Vec b, Vec x, Vec r, PetscReal rtol,
                        PetscReal abstol, PetscReal dtol, PetscInt maxits,
                        PetscBool zeroinitialguess, PetscInt *its,
                        PCRichardsonConvergedReason *reason) {
  /* We ignore all the provided tolerances since this is only supposed to be
   * used within MGMC
   */
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)maxits;

  // We also assume x is not zero
  (void)zeroinitialguess;

  // Always return one iteration
  *its = 1;
  *reason = PCRICHARDSON_CONVERGED_ITS;

  PetscFunctionBeginUser;

  // Get context object that contains rng and sqrt of the diagonal of the matrix
  SORRichardsonContext<Engine> *context;
  PetscCall(PCShellGetContext(pc, &context));

  // Below we set: r <- b + sqrt((2-omega)/omega) * D^1/2 * c, where c ~ N(0,I)
  fill_vec_rand(r, context->vec_size, *context->engine);

  PetscCall(VecPointwiseMult(r, r, context->sqrt_diag));
  PetscCall(VecAXPY(r, 1., b));

  // Perform actual Richardson step
  Mat A;
  PetscCall(PCGetOperators(pc, &A, NULL));

  // r <- r - A x
  PetscCall(VecScale(r, -1));
  PetscCall(MatMultAdd(A, x, r, r));
  PetscCall(VecScale(r, -1));

  // Apply (SOR) preconditioner
  PetscCall(PCApply(context->pc, r, context->z));

  PetscCall(VecAXPY(x, 1., context->z));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode sor_pc_richardson_destroy(PC pc) {
  PetscFunctionBeginUser;

  SORRichardsonContext<Engine> *context;
  PetscCall(PCShellGetContext(pc, &context));

  delete context;

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
