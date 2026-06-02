/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscmacros.h>
#include <petscpctypes.h>
#include <petscsystypes.h>

PETSC_EXTERN PetscErrorCode PCGAMGMCSetLevels(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCCreate_GAMGMC(PC);
PETSC_EXTERN PetscErrorCode PCGAMGMCGetInternalPC(PC, PC *);
PETSC_EXTERN PetscErrorCode PCGAMGMCSetInternalPC(PC, PC);
