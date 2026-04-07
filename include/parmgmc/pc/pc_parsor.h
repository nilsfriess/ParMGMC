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
#include <petscvec.h>

PETSC_EXTERN PetscErrorCode PCCreate_PARSOR(PC);
PETSC_EXTERN PetscErrorCode PCPARSORSetOmega(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCPARSORSetIterations(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCPARSORApplySOR(PC, Vec, PetscInt, PetscBool, Vec);
PETSC_EXTERN PetscErrorCode PCPARSORSetOmega(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCPARSORSetIterations(PC, PetscInt);
