/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2026  Nils Schiefer-Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscis.h>
#include <petscmacros.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsystypes.h>

PETSC_EXTERN PetscErrorCode PCCreate_SchwarzMGMC(PC);
PETSC_EXTERN PetscErrorCode PCSchwarzMGMCSetCoarseSpace(PC, Mat);
PETSC_EXTERN PetscErrorCode PCSchwarzMGMCSetSubdomains(PC, PetscInt, const IS[]);
PETSC_EXTERN PetscErrorCode PCSchwarzMGMCGetMG(PC, PC *);
