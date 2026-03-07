/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petsclogtypes.h>
#include <petscmacros.h>
#include <petscpctypes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscversion.h>

/* In PETSc >= 3.23, PetscOptionItems became an opaque pointer typedef;
   before that it was a plain struct and callbacks received a pointer. */
#if PETSC_VERSION_LT(3, 23, 0)
  #define PetscOptionItems_ARG PetscOptionItems *
#else
  #define PetscOptionItems_ARG PetscOptionItems
#endif

#define PARMGMC_ZIGGURAT "ziggurat"

#define PCMCGIBBS     "mcgibbs"
#define PCGAMGMC      "gamgmc"
#define PCSORGIBBS    "sorgibbs"
#define PCCHOLSAMPLER "cholsampler"
#define KSPCGSAMPLER  "cgsampler"

PETSC_EXTERN PetscClassId  PARMGMC_CLASSID;
PETSC_EXTERN PetscLogEvent MULTICOL_SOR;

PETSC_EXTERN PetscErrorCode ParMGMCInitialize(void);
PETSC_EXTERN PetscErrorCode ParMGMCFinalize(void);

PETSC_EXTERN PetscErrorCode PCRegisterSetSampleCallback(PC, PetscErrorCode (*)(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *, PetscErrorCode (*)(void *)));
PETSC_EXTERN PetscErrorCode PCSetSampleCallback(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *, PetscErrorCode (*)(void *));

PETSC_EXTERN PetscErrorCode ParMGMCGetPetscRandom(PetscRandom *);
