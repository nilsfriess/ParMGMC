/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
 */

#include "parmgmc/parmgmc.h"
#include "parmgmc/random/ziggurat.h"

#include <math.h>
#include <petsc/private/randomimpl.h>
#include <petscerror.h>
#include <petscmacros.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stdint.h>

// Ziggurat implementation taken from: https://people.sc.fsu.edu/~jburkardt/c_src/ziggurat/ziggurat.html
static float r4_uni(uint32_t *jsr)
{
  uint32_t jsr_input;
  float    value;

  jsr_input = *jsr;

  *jsr = (*jsr ^ (*jsr << 13));
  *jsr = (*jsr ^ (*jsr >> 17));
  *jsr = (*jsr ^ (*jsr << 5));

  value = fmod(0.5 + (float)(jsr_input + *jsr) / 65536.0 / 65536.0, 1.0);

  return value;
}

static uint32_t shr3_seeded(uint32_t *jsr)
{
  uint32_t value;

  value = *jsr;

  *jsr = (*jsr ^ (*jsr << 13));
  *jsr = (*jsr ^ (*jsr >> 17));
  *jsr = (*jsr ^ (*jsr << 5));

  value = value + *jsr;

  return value;
}

static float r4_nor(uint32_t *jsr, uint32_t kn[128], float fn[128], float wn[128])
{
  int         hz;
  uint32_t    iz;
  const float r = 3.442620;
  float       value;
  float       x;
  float       y;

  hz = (int)shr3_seeded(jsr);
  iz = (hz & 127);

  if (fabsf((float)hz) < kn[iz]) {
    value = (float)(hz)*wn[iz];
  } else {
    for (;;) {
      if (iz == 0) {
        for (;;) {
          x = -0.2904764 * logf(r4_uni(jsr));
          y = -logf(r4_uni(jsr));
          if (x * x <= y + y) { break; }
        }

        if (hz <= 0) {
          value = -r - x;
        } else {
          value = +r + x;
        }
        break;
      }

      x = (float)(hz)*wn[iz];

      if (fn[iz] + r4_uni(jsr) * (fn[iz - 1] - fn[iz]) < exp(-0.5 * x * x)) {
        value = x;
        break;
      }

      hz = (int)shr3_seeded(jsr);
      iz = (hz & 127);

      if (fabsf((float)hz) < kn[iz]) {
        value = (float)(hz)*wn[iz];
        break;
      }
    }
  }

  return value;
}

static void r4_nor_setup(uint32_t kn[128], float fn[128], float wn[128])
{
  double       dn = 3.442619855899;
  int          i;
  const double m1 = 2147483648.0;
  double       q;
  double       tn = 3.442619855899;
  const double vn = 9.91256303526217E-03;

  q = vn / exp(-0.5 * dn * dn);

  kn[0] = (uint32_t)((dn / q) * m1);
  kn[1] = 0;

  wn[0]   = (float)(q / m1);
  wn[127] = (float)(dn / m1);

  fn[0]   = 1.0;
  fn[127] = (float)(exp(-0.5 * dn * dn));

  for (i = 126; 1 <= i; i--) {
    dn        = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
    kn[i + 1] = (uint32_t)((dn / tn) * m1);
    tn        = dn;
    fn[i]     = (float)(exp(-0.5 * dn * dn));
    wn[i]     = (float)(dn / m1);
  }

  return;
}

typedef struct {
  uint32_t seed;
  uint32_t KN[128];
  float    FN[128], WN[128];
} ZigguratData;

static PetscErrorCode PetscRandomSeed_Ziggurat(PetscRandom r)
{
  ZigguratData *zig = r->data;
  PetscFunctionBegin;
  /* The XOR-shift PRNG has 0 as a fixed point: seeding with 0 would produce
     only zeros. Fall back to 1 in that case. */
  zig->seed = r->seed ? (uint32_t)r->seed : 1u;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRandomGetValueReal_Ziggurat(PetscRandom r, PetscReal *val)
{
  ZigguratData *zig = r->data;

  PetscFunctionBeginUser;
  *val = r4_nor(&zig->seed, zig->KN, zig->FN, zig->WN);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRandomDestroy_Ziggurat(PetscRandom r)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(r->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed, PetscRandomSeed_Ziggurat),
  PetscDesignatedInitializer(getvalue, PetscRandomGetValueReal_Ziggurat),
  PetscDesignatedInitializer(destroy, PetscRandomDestroy_Ziggurat),
};

PetscErrorCode PetscRandomCreate_Ziggurat(PetscRandom r)
{
  ZigguratData *z;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&z));
  r4_nor_setup(z->KN, z->FN, z->WN);
  r->data   = z;
  r->ops[0] = PetscRandomOps_Values;
  PetscCall(PetscObjectChangeTypeName((PetscObject)r, PARMGMC_ZIGGURAT));
  PetscFunctionReturn(PETSC_SUCCESS);
}
