/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscsnes.h>

// Create hierarchical Poisson Gibbs SNES
PETSC_EXTERN PetscErrorCode SNESCreate_PoissonGibbsFAS(SNES snes);
