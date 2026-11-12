"""Firedrake wrapper for ParMGMC's PCPATCHGIBBS (overlapping star-patch Gibbs sampler).

StarPatchGibbsPC uses exactly the patches of firedrake.ASMStarPC (the DOFs in the topological
star of each mesh entity of dimension -pc_star_construct_dim, translated to global numbers as
ASMStarPC does for PCASM) and hands them to a PCPATCHGIBBS, which performs the multiplicative
overlapping Schwarz Gibbs sweep without any per-patch PETSc objects.  With the same patch order
this is the same splitting as ASMStarPC with -pc_star_sub_pc_asm_local_type multiplicative and a
Cholesky sampler (or, with -pc_star_pgibbs_pc_patchgibbs_deterministic, an exact Cholesky solve)
on every patch.

Options (prefix <pc prefix>pc_star_):
    construct_dim, mat_ordering_type    as for ASMStarPC
    pgibbs_pc_patchgibbs_sweep          forward | backward | symmetric
    pgibbs_pc_patchgibbs_deterministic  no noise (deterministic smoother)

Sequential only.
"""

import numpy as np
import pymgmc
from firedrake.dmhooks import get_function_space
from firedrake.petsc import PETSc
from firedrake.preconditioners.asm import ASMStarPC


class StarPatchGibbsPC(ASMStarPC):
    _prefix = "pc_star_"

    def initialize(self, pc):
        _, P = pc.getOperators()
        self.prefix = (pc.getOptionsPrefix() or "") + self._prefix
        V = get_function_space(pc.getDM())

        ises = self.get_patches(V)  # local numbering, as in ASMPatchPC
        lgmap = V.dof_dset.lgmap
        ises = [lgmap.applyIS(iset) for iset in ises]  # as ASMPatchPC does for PCASM
        sizes = np.array([iset.getSize() for iset in ises], dtype=PETSc.IntType)
        offsets = np.zeros(len(ises) + 1, dtype=PETSc.IntType)
        np.cumsum(sizes, out=offsets[1:])
        indices = (np.concatenate([iset.getIndices() for iset in ises]).astype(PETSc.IntType)
                   if ises else np.zeros(0, dtype=PETSc.IntType))
        for iset in ises:
            iset.destroy()

        inner = PETSc.PC().create(comm=pc.comm)
        inner.incrementTabLevel(1, parent=pc)
        inner.setOptionsPrefix(self.prefix + "pgibbs_")
        inner.setType("patchgibbs")
        inner.setOperators(*pc.getOperators())
        pymgmc.PCPatchGibbsSetPatches(inner, offsets, indices)
        inner.setFromOptions()
        inner.setUp()
        self.inner = inner

    def update(self, pc):
        self.inner.setOperators(*pc.getOperators())
        self.inner.setUp()

    def apply(self, pc, x, y):
        self.inner.apply(x, y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("StarPatchGibbsPC has no transpose")

    def view(self, pc, viewer=None):
        if viewer is not None and hasattr(self, "inner"):
            viewer.printfASCII("Star patch Gibbs sampler (PCPATCHGIBBS on the ASMStarPC patches)\n")
            self.inner.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "inner"):
            self.inner.destroy()
