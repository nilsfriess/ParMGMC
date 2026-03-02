/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
 */

#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/pc/pc_gibbs.h"
#include "parmgmc/pc/pc_hogwild.h"
#include "parmgmc/random/ziggurat.h"

#include <petsc/private/pcimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>

/** @file
    @brief This file contains general purpose functions for the ParMGMC library.
*/

PetscClassId  PARMGMC_CLASSID;
PetscLogEvent MULTICOL_SOR;

PetscRandom parmgmc_rand = NULL;

static PetscErrorCode ParMGMCRegisterPCAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PCRegister(PCHOGWILD, PCCreate_Hogwild));
  PetscCall(PCRegister(PCGIBBS, PCCreate_Gibbs));
  PetscCall(PCRegister(PCGAMGMC, PCCreate_GAMGMC));
  PetscCall(PCRegister(PCCHOLSAMPLER, PCCreate_CholSampler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParMGMCRegisterPetscRandomAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PetscRandomRegister(PARMGMC_ZIGGURAT, PetscRandomCreate_Ziggurat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCGetPetscRandom(PetscRandom *pr)
{
  PetscFunctionBegin;
  if (!parmgmc_rand) {
    PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &parmgmc_rand));
    PetscCall(PetscRandomSetType(parmgmc_rand, PARMGMC_ZIGGURAT));
  }
  /* Bump the reference count so that callers can call PetscRandomDestroy on
     the returned object independently of the global parmgmc_rand lifetime. */
  PetscCall(PetscObjectReference((PetscObject)parmgmc_rand));
  *pr = parmgmc_rand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCInitialize(void)
{
  PetscFunctionBeginUser;
  PetscCall(ParMGMCRegisterPCAll());
  PetscCall(ParMGMCRegisterPetscRandomAll());

  PetscCall(PetscClassIdRegister("ParMGMC", &PARMGMC_CLASSID));
  PetscCall(PetscLogEventRegister("MulticolSOR", PARMGMC_CLASSID, &MULTICOL_SOR));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCFinalize(void)
{
  PetscFunctionBeginUser;
  if (parmgmc_rand) PetscCall(PetscRandomDestroy(&parmgmc_rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCRegisterSetSampleCallback(PC pc, PetscErrorCode (*set)(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *)))
{
  PetscFunctionBeginUser;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSetSampleCallback_C", set));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetSampleCallback(PC pc, PetscErrorCode (*cb)(PetscInt, Vec, void *), void *ctx, PetscErrorCode (*deleter)(void *))
{
  PetscFunctionBeginUser;
  PetscUseMethod((PetscObject)pc, "PCSetSampleCallback_C", (PC, PetscErrorCode(*)(PetscInt, Vec, void *), void *, PetscErrorCode (*)(void *)), (pc, cb, ctx, deleter));
  PetscFunctionReturn(PETSC_SUCCESS);
}
