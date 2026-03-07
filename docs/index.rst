Parallel Multigrid Monte Carlo
==============================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Overview

   installation
   api/modules
   examples/examples

ParMGMC is a parallel implementation of the Multigrid Monte Carlo method using the Portable Extensible Toolkit for Scientific computing (`PETSc <https://petsc.org/>`_) for sampling from high-dimensional Gaussian distributions.

The implementation includes

* a parallel MulticolorGibbs sampler,
  
* a geometric Multigrid Monte Carlo sampler to generate samples on simple structured grids,
  
* a fully algebraic sampler to generate random samples on graphs.

All three variants can be used to sample from Gaussian distributions with given precision matrix and for linear Bayesian inverse problems with Gaussian priors.

It is written in C but can also be used from C++ and has Python bindings for usage with ``petsc4py``.

.. note::

   This library is under active development and its interface may change at any time.
