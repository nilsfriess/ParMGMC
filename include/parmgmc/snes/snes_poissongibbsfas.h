/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscvec.h>
#include <petscmat.h>
#include <petscsnes.h>

/* User context for Poisson Gibbs sampler */
typedef struct {  
} PoissonGibbsFASCtx;

PETSC_EXTERN PetscErrorCode SNESCreate_PoissonGibbsFAS(SNES snes);