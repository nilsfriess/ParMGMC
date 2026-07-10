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
#include <petscmat.h>

/* User context for Poisson Gibbs sampler */
typedef struct {
   Vec event_counts; // measured event counts
   Vec nu;           // Shift vector of length m
   Mat B;            // observation matrix of size n x m
} PoissonGibbsCtx;


PETSC_EXTERN PetscErrorCode PCCreate_PoissonGibbs(PC pc);
