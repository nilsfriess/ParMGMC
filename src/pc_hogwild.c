#include "parmgmc/pc/pc_hogwild.h"

#include <petsc/private/pcmgimpl.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <stdio.h>

typedef struct {
  Vec         sqrtdiag;
  PetscRandom prand;
} PC_Hogwild;

static PetscErrorCode PCApplyRichardson_Hogwild(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;

  PC_Hogwild *hw = pc->data;

  PetscFunctionBeginUser;
  for (PetscInt it = 0; it < its; ++it) {
    PetscCall(VecSetRandom(w, hw->prand));
    PetscCall(VecPointwiseMult(w, w, hw->sqrtdiag));
    PetscCall(VecAXPY(w, 1., b));
    PetscCall(MatSOR(pc->pmat, w, 1., SOR_LOCAL_FORWARD_SWEEP, 0., 1., 1., y));

    PetscReal norm;
    PetscCall(VecNorm(y, NORM_2, &norm));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", norm));
  }

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_Hogwild(PC pc)
{
  PC_Hogwild *hw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&hw));
  pc->data = hw;

  PetscCall(MatCreateVecs(pc->pmat, &hw->sqrtdiag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, hw->sqrtdiag));
  PetscCall(VecSqrtAbs(hw->sqrtdiag));

  // TODO: Allow user to pass own PetscRandom
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)pc), &hw->prand));
  PetscCall(PetscRandomSetType(hw->prand, "ziggurat"));

  pc->ops->applyrichardson = PCApplyRichardson_Hogwild;
  PetscFunctionReturn(PETSC_SUCCESS);
}