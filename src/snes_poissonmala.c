/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess, Eike Mueller

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
    */

/** @file snes_poissonmala.c
    @brief Preconditioned MALA sampler for posterior obtained by conditioning a Gaussian prior
    on a Poisson process

    # Options database keys
    - `-poissonmala_epsilon` MALA stepsize

    # Notes

    TODO
*/

#include "parmgmc/snes/snes_poissonmala.h"
#include "parmgmc/poisson.h"

#include <petsc/private/snesimpl.h>
#include <petscerror.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stddef.h>
#include <string.h>

/* Internal workspace for Poisson MALA sampler SNES */
typedef struct {
  PetscScalar epsilon; // MALA stepsize
} SNES_PoissonMALA;

/* Generate a new sample by updating theta -> theta' 
 *
 * This is the key computational routine for generating a new sample. 
 *
 * Parameters
 *   snes [inout] : underlying SNES object
 */
static PetscErrorCode SNESSample_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;
  PoissonCtx       *ctx;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESGetApplicationContext(snes, &ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reset SNES object
 *
 * Free contents of temporary workspace to prepare object for reuse
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESReset_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Destroy SNES object
 *
 * Free contents of temporary workspace with SNESReset_PoissonMALA() and in addition
 * deallocate the workspace itself.
 *
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESDestroy_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(SNESReset_PoissonMALA(snes));
  PetscCall(PetscFree(poissonmala));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set up SNES object
 *   
 * Parameters
 *   snes [inout] : SNES object
 */
static PetscErrorCode SNESSetUp_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Inspect options database to set SNES parameters
 *
 * The only parameter that can be set is the stepsize epsilon
 *
 * Parameters
 *   snes [inout] : SNES object
 *   PetscOptionsObject [in] : PETSc options object
 */
static PetscErrorCode SNESSetFromOptions_PoissonMALA(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBegin;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscOptionsHeadBegin(PetscOptionsObject, "Poisson MALA options");
  PetscCall(PetscOptionsReal("-poissonmala_epsilon", "Stepsize", NULL, poissonmala->epsilon, &poissonmala->epsilon, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* View SNES object
 * 
 * Parameters
 *   snes [in] : SNES object
 *   viewer [inout] : viewer to user
 */
static PetscErrorCode SNESView_PoissonMALA(SNES snes, PetscViewer viewer)
{
  SNES_PoissonMALA *poissonmala;
  (void)poissonmala;
  (void)viewer;

  PetscFunctionBeginUser;
  poissonmala = (SNES_PoissonMALA *)snes->data;
  PetscCall(PetscViewerASCIIPrintf(viewer, "  stepsize=%" PetscReal_FMT "\n", poissonmala->epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create unitialised SNES object
 *
 * Parameters
 *   snes [inout] : SNES object to create
 */
PetscErrorCode SNESCreate_PoissonMALA(SNES snes)
{
  SNES_PoissonMALA *poissonmala;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&poissonmala));
  snes->data = (void *)poissonmala;

  snes->ops->solve          = SNESSample_PoissonMALA;
  snes->ops->destroy        = SNESDestroy_PoissonMALA;
  snes->ops->reset          = SNESReset_PoissonMALA;
  snes->ops->setup          = SNESSetUp_PoissonMALA;
  snes->ops->setfromoptions = SNESSetFromOptions_PoissonMALA;
  snes->ops->view           = SNESView_PoissonMALA;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  poissonmala->epsilon = 0.1;

  PetscFunctionReturn(PETSC_SUCCESS);
}
