/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petscistypes.h>
#include <petscmacros.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

typedef struct _MCSOR {
  void *ctx;
} *MCSOR;

PETSC_EXTERN PetscErrorCode MCSORCreate(Mat, MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORSetUp(MCSOR);
PETSC_EXTERN PetscErrorCode MCSORDestroy(MCSOR *);
PETSC_EXTERN PetscErrorCode MCSORApply(MCSOR, Vec, Vec);
PETSC_EXTERN PetscErrorCode MCSORSetOmega(MCSOR, PetscReal);
PETSC_EXTERN PetscErrorCode MCSORSetSweepType(MCSOR, MatSORType);
PETSC_EXTERN PetscErrorCode MCSORGetSweepType(MCSOR, MatSORType *);
PETSC_EXTERN PetscErrorCode MCSORGetISColoring(MCSOR, ISColoring *);
PETSC_EXTERN PetscErrorCode MCSORGetNumColors(MCSOR, PetscInt *);
PETSC_EXTERN PetscErrorCode MCSORBuildLRCCorrection(PetscErrorCode (*det_sor)(void *, Vec, Vec), void *, Mat, Mat, Vec, Mat *);
