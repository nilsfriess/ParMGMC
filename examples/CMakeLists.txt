cmake_minimum_required(VERSION 3.20)
project(ParMGMC_Tests
	LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_search_module(parmgmc parmgmc>=0.0.1 REQUIRED)
pkg_search_module(PETSc PETSc>=3.19 REQUIRED)
pkg_search_module(MPICH mpich REQUIRED)

pkg_get_variable(MPIRUN_PREFIX ${MPICH_MODULE_NAME} prefix)
set(MPIRUN_PATH "${MPIRUN_PREFIX}/bin/mpirun")
set(PARMGMC_CC  "${MPIRUN_PREFIX}/bin/mpicc")
message(STATUS "MPICH prefix: ${MPIRUN_PREFIX}")

set(PARMGMC_CFLAGS "${parmgmc_CFLAGS} ${PETSc_CFLAGS}")
set(PARMGMC_CFLAGS "${PARMGMC_CFLAGS} -Wl,-rpath,${parmgmc_LIBRARY_DIRS},-rpath,${PETSc_LIBRARY_DIRS}")
set(PARMGMC_CFLAGS "${PARMGMC_CFLAGS} -L${parmgmc_LIBRARY_DIRS} -L${PETSc_LIBRARY_DIRS}")
set(PARMGMC_CFLAGS "${PARMGMC_CFLAGS} -l${parmgmc_LIBRARIES} -l${PETSc_LIBRARIES}")
string(REPLACE ";" " " PARMGMC_CFLAGS "${PARMGMC_CFLAGS}")
set(PARMGMC_COMP "${PARMGMC_CFLAGS} ${CMAKE_C_FLAGS}")
set(PARMGMC_CFLAGS "${PARMGMC_CFLAGS} ${PARMGMC_ADD_CFLAGS}")
message(STATUS "LIT compile line: ${PARMGMC_CC} ${PARMGMC_CFLAGS}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-seq
		  COMMENT "Running sequential tests"
		  COMMAND "lit" "${CMAKE_CURRENT_BINARY_DIR}" -a --time-tests --filter-out=benchmark.c -DNP=1)
add_custom_target(check-par
		  COMMENT "Running parallel tests"
		  COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}" -a --time-tests --filter-out=benchmark.c -DNP=4)

add_custom_target(check
		  DEPENDS check-par check-seq)
