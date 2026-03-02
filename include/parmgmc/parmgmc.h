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

#define PARMGMC_ZIGGURAT "ziggurat"

#define PCGIBBS       "gibbs"
#define PCGAMGMC      "gamgmc"
#define PCHOGWILD     "hogwild"
#define PCCHOLSAMPLER "cholsampler"
#define KSPCGSAMPLER  "cgsampler"

PETSC_EXTERN PetscClassId  PARMGMC_CLASSID;
PETSC_EXTERN PetscLogEvent MULTICOL_SOR;

PETSC_EXTERN PetscErrorCode ParMGMCInitialize(void);
PETSC_EXTERN PetscErrorCode ParMGMCFinalize(void);

PETSC_EXTERN PetscErrorCode PCRegisterSetSampleCallback(PC, PetscErrorCode (*)(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *, PetscErrorCode (*)(void *)));
PETSC_EXTERN PetscErrorCode PCSetSampleCallback(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *, PetscErrorCode (*)(void *));

PETSC_EXTERN PetscErrorCode ParMGMCGetPetscRandom(PetscRandom *);
