/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscksp.h>
#include <petscmacros.h>
#include <petscsystypes.h>
#include <petscvec.h>

PETSC_EXTERN PetscErrorCode KSPCreate_CGSampler(KSP);
PETSC_EXTERN PetscErrorCode CGSamplerGetSample(KSP, Vec);
