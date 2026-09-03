/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.

    Non-linear Gibbs sampler for posterior obtained by contitioning a Gaussian prior
    on a Poisson process.
*/

#include "parmgmc/snes/snes_poissongibbsfas.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/snesimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

typedef struct {  
}  SNES_PoissonGibbsFAS;

/* Generate a new sample (computational routine) */
static PetscErrorCode SNESSample_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  
  PetscFunctionBeginUser;  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;

  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESDestroy_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;

  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);  
}

static PetscErrorCode SNESSetUp_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_PoissonGibbsFAS(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_PoissonGibbsFAS(SNES snes, PetscViewer viewer)
{
  SNES_PoissonGibbsFAS* poissongibbsfas = (SNES_PoissonGibbsFAS*)snes->data;
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESCreate_PoissonGibbsFAS(SNES snes)
{
  SNES_PoissonGibbsFAS* poissongibbsfas;
  
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissongibbsfas));
  snes->data       = (void*)poissongibbsfas;
  
  snes->ops->solve           = SNESSample_PoissonGibbsFAS;
  snes->ops->destroy         = SNESDestroy_PoissonGibbsFAS;
  snes->ops->reset           = SNESReset_PoissonGibbsFAS;
  snes->ops->setup           = SNESSetUp_PoissonGibbsFAS;
  snes->ops->setfromoptions  = SNESSetFromOptions_PoissonGibbsFAS;
  snes->ops->view            = SNESView_PoissonGibbsFAS;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}
