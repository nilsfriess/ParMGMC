/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
 */

#include "parmgmc/parmgmc.h"
#include "parmgmc/pc/pc_chols.h"
#include "parmgmc/pc/pc_gamgmc.h"
#include "parmgmc/pc/pc_mcgibbs.h"
#include "parmgmc/pc/pc_parsor.h"
#include "parmgmc/pc/pc_sorgibbs.h"

#include <petsc/private/pcimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>

#ifdef PARMGMC_HAVE_MKL
  #include <mkl_vsl.h>
#endif

/** @file
    @brief This file contains general purpose functions for the ParMGMC library.
*/

PetscClassId  PARMGMC_CLASSID;
PetscLogEvent MULTICOL_SOR;
PetscLogEvent VEC_SET_RANDOM_NORMAL;

PetscRandom parmgmc_rand = NULL;

#ifdef PARMGMC_HAVE_MKL
static VSLStreamStatePtr parmgmc_vsl_stream = NULL;
#endif

static PetscErrorCode ParMGMCRegisterPCAll(void)
{
  PetscFunctionBeginUser;
  PetscCall(PCRegister(PCSORGIBBS, PCCreate_SORGibbs));
  PetscCall(PCRegister(PCMCGIBBS, PCCreate_MulticolorGibbs));
  PetscCall(PCRegister(PCGAMGMC, PCCreate_GAMGMC));
  PetscCall(PCRegister(PCCHOLSAMPLER, PCCreate_CholSampler));
  PetscCall(PCRegister(PCPARSOR, PCCreate_PARSOR));
  PetscCall(PCRegister(PCWOODBURY, PCCreate_Woodbury));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCGetPetscRandom(PetscRandom *pr)
{
  PetscFunctionBegin;
  if (!parmgmc_rand) {
    PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &parmgmc_rand));
    PetscCall(PetscRandomSetFromOptions(parmgmc_rand));
  }
  /* Bump the reference count so that callers can call PetscRandomDestroy on
     the returned object independently of the global parmgmc_rand lifetime. */
  PetscCall(PetscObjectReference((PetscObject)parmgmc_rand));
  *pr = parmgmc_rand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetRandomStandardNormal(Vec v, PetscRandom r)
{
  PetscInt     n;
  PetscScalar *array;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(VEC_SET_RANDOM_NORMAL, v, r, 0, 0));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &array));

#if defined(PARMGMC_HAVE_MKL) && !defined(PETSC_USE_COMPLEX)
  if (!parmgmc_vsl_stream) {
    PetscInt64 seed;
    int        rank;
    PetscCall(PetscRandomGetSeed(r, &seed));
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MKL_INT err = vslNewStream(&parmgmc_vsl_stream, VSL_BRNG_MT19937, (MKL_UINT)((unsigned long)seed ^ (unsigned long)rank));
    PetscCheck(err == VSL_ERROR_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "vslNewStream failed (error %d)", (int)err);
  }
  {
    MKL_INT err;
  #ifdef PETSC_USE_REAL_SINGLE
    err = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, parmgmc_vsl_stream, (MKL_INT)n, array, 0.0f, 1.0f);
  #else
    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, parmgmc_vsl_stream, (MKL_INT)n, array, 0.0, 1.0);
  #endif
    PetscCheck(err == VSL_ERROR_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "vdRngGaussian failed (error %d)", (int)err);
  }
#else
  /* Box-Muller: each (u1, u2) pair yields two independent normals */
  for (PetscInt i = 0; i < n; i += 2) {
    PetscReal u1, u2, radius, theta;

    PetscCall(PetscRandomGetValueReal(r, &u1));
    PetscCall(PetscRandomGetValueReal(r, &u2));

    radius   = PetscSqrtReal(-2.0 * PetscLogReal(u1));
    theta    = 2.0 * PETSC_PI * u2;
    array[i] = radius * PetscCosReal(theta);
    if (i + 1 < n) array[i + 1] = radius * PetscSinReal(theta);
  }
#endif

  PetscCall(VecRestoreArray(v, &array));
  PetscCall(PetscLogEventEnd(VEC_SET_RANDOM_NORMAL, v, r, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCInitialize(void)
{
  PetscFunctionBeginUser;
  PetscCall(ParMGMCRegisterPCAll());

  PetscCall(PetscClassIdRegister("ParMGMC", &PARMGMC_CLASSID));
  PetscCall(PetscLogEventRegister("MulticolSOR", PARMGMC_CLASSID, &MULTICOL_SOR));
  PetscCall(PetscLogEventRegister("VecSetRandN", PARMGMC_CLASSID, &VEC_SET_RANDOM_NORMAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParMGMCFinalize(void)
{
  PetscFunctionBeginUser;
  if (parmgmc_rand) PetscCall(PetscRandomDestroy(&parmgmc_rand));
#if defined(PARMGMC_HAVE_MKL) && !defined(PETSC_USE_COMPLEX)
  if (parmgmc_vsl_stream) vslDeleteStream(&parmgmc_vsl_stream);
#endif
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
