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
  Vec event_counts; // measured event counts
  Mat Q_prec;       // precision matrix
  Mat B_meas;       // measurement matrix
  Vec nu;           // offset vector
} PoissonGibbsCtx;

// Create Poisson Gibbs SNES
PETSC_EXTERN PetscErrorCode SNESCreate_PoissonGibbs(SNES snes);
// Create hierarchical Poisson Gibbs SNES
PETSC_EXTERN PetscErrorCode SNESCreate_PoissonGibbsFAS(SNES snes);
// Set number of Gibbs sweeps
PETSC_EXTERN PetscErrorCode SNESPoissonGibbsSetIterations(SNES snes, PetscInt its);
