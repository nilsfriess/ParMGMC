#!/usr/bin/env bash

INTEL_MKL_PATH=/opt/intel/oneapi/mkl/latest/
PYTHON=python3.12
PARMGMC_COMMIT=164f2a4212e278318a7396cff33603799083c67a

# Setting these is only necessary for PNetCDF which firedrake pulls in.
# It ignores the CC, CXX etc. compilers that PETSc sets and only checks
# for MPICC, MPICXX etc. So we set it here manually and pass it later.
PNETCDF_MPICC=$(pwd)/petsc/arch-firedrake-default/bin/mpicc
PNETCDF_MPICXX=$(pwd)/petsc/arch-firedrake-default/bin/mpicxx
PNETCDF_MPIF90=$(pwd)/petsc/arch-firedrake-default/bin/mpif90

NPROC=16

set -xe

# Download firedrake configure script and clone PETSc
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure
git clone --branch $($PYTHON firedrake-configure --os unknown --show-petsc-version) https://gitlab.com/petsc/petsc.git
cd petsc
$PYTHON ../firedrake-configure --os unknown --show-petsc-configure-options | xargs -L1 ./configure --with-cuda=0 --with-cudac=0 --with-hip=0 --with-hipc=0 --download-mpich --with-mkl_cpardiso --with-mkl_pardiso --with-scalapack-dir=$INTEL_MKL_PATH --with-blaslapack-dir=$INTEL_MKL_PATH --download-pnetcdf-configure-arguments="MPICC=$PNETCDF_MPICC MPICXX=$PNETCDF_MPICXX MPIF77=$PNETCDF_MPIF90 MPIF90=$PNETCDF_MPIF90"
# The last argument above is to fix some pnetcdf weirdness: it ignores the CC=... that PETSc sets and then uses the wrong compiler

# Build PETSc
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-firedrake-default all -j$NPROC
cd ..

# Prepare firedrake install
$PYTHON -m venv venv-firedrake
. venv-firedrake/bin/activate

pip cache purge
export $(python firedrake-configure --os unknown --show-env)
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
export CC=$PETSC_DIR/$PETSC_ARCH/bin/mpicc
export CXX=$PETSC_DIR/$PETSC_ARCH/bin/mpicxx
export FC=$PETSC_DIR/$PETSC_ARCH/bin/mpif90

# Install firedrake
# mpi4py >= 4.1 understands MPICH 5; build it against PETSc's MPICH
pip install --no-binary mpi4py 'mpi4py>=4.1'
# h5py build deps, then build h5py against that mpi4py instead of its pinned one
pip install setuptools cython numpy pkgconfig
pip install --no-build-isolation --no-binary h5py h5py
pip install --no-binary h5py 'firedrake[check]'

# Other packages needed for the ParMGMC examples
pip install numpy scipy pyvista emcee matplotlib pandas

# Download ParMGMC
git clone https://github.com/nilsfriess/ParMGMC.git
cd ParMGMC
git checkout $PARMGMC_COMMIT
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_PREFIX_PATH=$PETSC_DIR/$PETSC_ARCH/ -DCMAKE_BUILD_TYPE=Release -DMKL_DIR=$INTEL_MKL_PATH/lib/cmake/mkl -DPARMGMC_BUILD_EXAMPLES=On
make -j$NPROC

cd ../..
cat >> venv-firedrake/bin/activate <<EOF
export PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
export PATH=\$PETSC_DIR/\$PETSC_ARCH/bin:\$PATH
EOF
