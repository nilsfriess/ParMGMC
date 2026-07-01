cmake_minimum_required(VERSION 3.20)
project(ParMGMC
	LANGUAGES C CXX
	VERSION 0.0.1
	DESCRIPTION "Implementation of the Multigrid Monte Carlo method in PETSc"
	HOMEPAGE_URL "https://github.com/nilsfriess/ParMGMC.git")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)

if (${PETSC_FOUND})
  message(STATUS "Found PETSc version ${PETSC_VERSION}, include dirs: ${PETSC_INCLUDE_DIRS}")
endif()

find_package(MPI REQUIRED)
message(STATUS "mpiexec executable located at ${MPIEXEC_EXECUTABLE}")

find_package(MKL CONFIG)
if(MKL_FOUND)
  message(STATUS "Found Intel MKL — VSL random number generation enabled")
else()
  message(STATUS "Intel MKL not found — Box-Muller fallback used for random number generation")
endif()

# pc_chols.c calls BLAS/LAPACK; link the same one PETSc uses (read from
# petscvariables) so it resolves at link time on macOS without an ABI mismatch.
pkg_get_variable(PETSC_PREFIX PETSc prefix)
set(PETSC_BLASLAPACK_LIB "")
if(PETSC_PREFIX AND EXISTS "${PETSC_PREFIX}/lib/petsc/conf/petscvariables")
  file(STRINGS "${PETSC_PREFIX}/lib/petsc/conf/petscvariables" _blas_line REGEX "^BLASLAPACK_LIB = ")
  string(REGEX REPLACE "^BLASLAPACK_LIB = " "" _blas_line "${_blas_line}")
  separate_arguments(PETSC_BLASLAPACK_LIB UNIX_COMMAND "${_blas_line}")
endif()
if(PETSC_BLASLAPACK_LIB)
  message(STATUS "Using PETSc's BLAS/LAPACK: ${PETSC_BLASLAPACK_LIB}")
  set(PARMGMC_BLAS_LINK ${PETSC_BLASLAPACK_LIB})
else()
  message(STATUS "PETSc BLAS/LAPACK not recorded; falling back to find_package(BLAS)")
  find_package(BLAS REQUIRED)
  set(PARMGMC_BLAS_LINK BLAS::BLAS)
endif()

# iact.c needs a double-precision FFTW.
find_package(FFTW COMPONENTS DOUBLE_LIB REQUIRED)

# PETSc changed the signature of KSPSetDMActive() in some version. Previously we used
# PETSC_VERSION_GT to check but that unconditionally returns `true` for non-release
# builds (e.g. builds from a PETSc main branch). So here we try to compile a program
# with the new signature and set a compile definition that can then be checked in the
# code. This is ugly and we can of course not do this for every function that PETSc
# decides to change; arguably this is a flaw in PETSc's PETSC_VERSION_GT macro.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_C)
check_c_source_compiles("
#include <petscksp.h>
int main(void){ KSP ksp; return KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE); }
" PARMGMC_PETSC_KSP_DMACTIVE_3ARG)
unset(CMAKE_REQUIRED_LIBRARIES)

# Injected into parmgmc.pc so the standalone lit test builds (which compile the
# examples via pkg-config rather than by linking the parmgmc target) see it too.
set(PARMGMC_PC_CFLAGS "")
if(PARMGMC_PETSC_KSP_DMACTIVE_3ARG)
  set(PARMGMC_PC_CFLAGS "-DPARMGMC_PETSC_KSP_DMACTIVE_3ARG")
endif()
if(PETSC_BLASLAPACK_LIB)
  message(STATUS "Using PETSc's BLAS/LAPACK: ${PETSC_BLASLAPACK_LIB}")
  set(PARMGMC_BLAS_LINK ${PETSC_BLASLAPACK_LIB})
else()
  message(STATUS "PETSc BLAS/LAPACK not recorded; falling back to find_package(BLAS)")
  find_package(BLAS REQUIRED)
  set(PARMGMC_BLAS_LINK BLAS::BLAS)
endif()

# iact.c needs a double-precision FFTW.
find_package(FFTW COMPONENTS DOUBLE_LIB REQUIRED)

add_library(parmgmc SHARED)

if(PARMGMC_PETSC_KSP_DMACTIVE_3ARG)
  # PUBLIC so it also reaches the in-tree example/benchmark targets that link parmgmc.
  target_compile_definitions(parmgmc PUBLIC PARMGMC_PETSC_KSP_DMACTIVE_3ARG)
endif()

if(MKL_FOUND)
  target_compile_definitions(parmgmc PRIVATE PARMGMC_HAVE_MKL)
  # Only pull in headers; PETSc is already linked against MKL so we rely on
  # its transitive link for symbol resolution (direct MKL::MKL linkage causes
  # double-init segfaults in PETSc builds that bundle MKL).
  target_include_directories(parmgmc SYSTEM PRIVATE
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

file(GLOB_RECURSE PARMGMC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/parmgmc/*.h")
message(STATUS ${PARMGMC_INCLUDES})
target_sources(parmgmc
	PRIVATE
	    src/pc_sorgibbs.c
	    src/pc_mcgibbs.c
	    src/pc_gamgmc.c
	    src/pc_chols.c
	    src/pc_parsor.c
	    src/mc_sor.c
            src/woodbury.c
	    src/parmgmc.c
	    src/problems.c
	    src/ms.c
	    src/obs.c
	    src/iact.c
	    src/stats.c
	 PUBLIC
	    FILE_SET parmgmcheaders
	    BASE_DIRS include
	    TYPE HEADERS
	       FILES ${PARMGMC_INCLUDES})
# target_include_directories(parmgmc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(parmgmc SYSTEM PUBLIC ${PETSC_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS})
target_compile_options(parmgmc PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes)
target_compile_options(parmgmc PUBLIC ${PETSC_CFLAGS})
target_link_libraries(parmgmc PUBLIC PkgConfig::PETSC MPI::MPI_C FFTW::Double ${PARMGMC_BLAS_LINK})

add_executable(benchmark examples/benchmark/main.cc)
target_include_directories(benchmark PRIVATE include)
target_compile_options(benchmark PRIVATE -Wall -Wextra -Wpedantic -pedantic)
target_link_libraries(benchmark PRIVATE parmgmc m MPI::MPI_CXX MPI::MPI_C)

# add_executable(test test.c)
# target_include_directories(test PRIVATE include)
# target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes)
# target_link_libraries(test parmgmc m)

set(MFEM_DIR "" CACHE PATH "Path to the MFEM build or install prefix.")
find_package(MFEM)
if(MFEM_FOUND)
  message(STATUS "Found MFEM (include dirs: ${MFEM_INCLUDE_DIRS})")

  find_package(MPI REQUIRED)
  message(STATUS "mpiexec executable located at ${MPIEXEC_EXECUTABLE}")

  # add_executable(mfem_ex examples/mfem/main.cc)
  # target_include_directories(mfem_ex PRIVATE include ${MFEM_INCLUDE_DIRS})
  # target_compile_options(mfem_ex PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wshadow -Wpointer-arith -Wcast-qual)
  # target_link_libraries(mfem_ex PUBLIC parmgmc PRIVATE m ${MFEM_LIBRARIES} MPI::MPI_CXX MPI::MPI_C)

  target_include_directories(benchmark SYSTEM PRIVATE ${MFEM_INCLUDE_DIRS})
  target_link_libraries(benchmark PRIVATE ${MFEM_LIBRARIES})
  target_compile_definitions(benchmark PUBLIC PARMGMC_HAVE_MFEM)
else()
  message(STATUS "Could not find MFEM")
endif()

include(GNUInstallDirs)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parmgmc 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	FILE_SET parmgmcheaders
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/share/pkgconfig/parmgmc.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/parmgmc.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parmgmc.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(NOT DEFINED PARMGMC_ENABLE_PYTHON_BINDINGS)
  set(PARMGMC_ENABLE_PYTHON_BINDINGS False)
endif()

if(${PARMGMC_ENABLE_PYTHON_BINDINGS})
  add_subdirectory(python)
endif()

if(NOT DEFINED PARMGMC_BUILD_DOCS)
  set(PARMGMC_BUILD_DOCS False)
endif()

if(${PARMGMC_BUILD_DOCS})
  add_subdirectory(docs)
endif()

if(NOT DEFINED PARMGMC_BUILD_DOCS)
  set(PARMGMC_BUILD_DOCS False)
endif()

if(${PARMGMC_BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()
