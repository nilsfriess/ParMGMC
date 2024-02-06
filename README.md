# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C++17 implementation of the Multigrid Monte Carlo method to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Installation and usage
ParMGMC is a header-only library and as such requires no installation. If you want to use the library, simply copy the folder `parmgmc` inside the `include` directory somewhere on your computer. ParMGMC has the following dependencies:
- An MPI installation (e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/))
- [PETSc](https://petsc.org/)  (version >= 3.17)

To compile and run the tests and/or the examples, you need CMake. To compile and run the tests first clone the repository and configure the build with
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
```
To specify a custom compiler (e.g., a MPI compiler wrapper) add `-DCMAKE_CXX_COMPILER=mpic++`. To add custom compiler flags, use `-DCMAKE_CXX_FLAGS="-O3 -march=native`.
If the CMake configuartion finished successfully, compile the tests using 
```bash
$ make tests
```
The execute 
```bash
$ ./tests/tests
```
to run the serial tests and
```bash
$ mpirun -np 4 ./tests/tests [mpi]
```
to run the parallel test suite.

## Running the MFEM examples
Optionally, one can pass `-Dmfem_DIR=/path/to/mfem_install/lib/cmake/mfem` during CMake configuration to build the examples that use [MFEM](https://mfem.org/). Note that this requires a MFEM installation built with MPI, METIS, and PETSc support enabled (refer to the MFEM documentation for details). Configure the project as described above and then run 
```bash
$ make examples
```
to build the examples.
