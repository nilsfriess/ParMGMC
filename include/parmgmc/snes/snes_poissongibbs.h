/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

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
  Vec f_rhs;        // right hand side vector  
  Vec nu;           // Shift vector of length m
} PoissonGibbsCtx;

PETSC_EXTERN PetscErrorCode SNESCreate_PoissonGibbs(SNES snes);
