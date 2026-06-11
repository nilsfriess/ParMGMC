"""Assembled geometric-multigrid interpolation for Firedrake.

Firedrake builds the multigrid prolongation as a matrix-free ``MATPYTHON``
wrapping the ``TransferManager`` (see ``firedrake.mg.ufl_utils``).  That makes
``-pc_mg_galerkin`` impossible: PETSc forms coarse operators A_c = P^T A P with
``MatPtAP``, which has no implementation for ``(seqaij, python)`` and raises
``PETSC_ERR_SUP``.

Importing this module monkeypatches ``create_interpolation`` so the DM hands
PETSc an *assembled* AIJ prolongation instead.  Then ``-pc_mg_galerkin both``
works out of the box -- ``PCSetUp_MG`` attaches the coarse DMs and forms the
Galerkin operators itself, with no manual prolongation surgery.

Import it before building the ``MeshHierarchy`` (``attach_hooks`` resolves
``create_interpolation`` lazily, so the patch must be in place first).

Limitations: assembles via the generic ``interpolate`` path (slower than a
hand-rolled nested assembly) and does not carry Dirichlet-BC row/column
elimination.  Adequate for natural-BC demonstrations; not a general fix.
"""

import firedrake as fd
import firedrake.mg.ufl_utils as _ufl_utils
from firedrake.dmhooks import get_appctx


def _assembled_create_interpolation(dmc, dmf):
    V_c = get_appctx(dmc)._problem.u_restrict.function_space()
    V_f = get_appctx(dmf)._problem.u_restrict.function_space()
    P = fd.assemble(fd.interpolate(fd.TrialFunction(V_c), V_f))
    return P.petscmat, None


_ufl_utils.create_interpolation = _assembled_create_interpolation
