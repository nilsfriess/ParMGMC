cmake_minimum_required(VERSION 3.20)
project(ParMGMC_Tests
        LANGUAGES C)

if (PROJECT_IS_TOP_LEVEL)
   find_package(PkgConfig REQUIRED)
   pkg_search_module(parmgmc parmgmc>=0.0.1 REQUIRED)
   pkg_search_module(PETSc PETSc>=3.19 REQUIRED)

   # Recent PETSc exposes the mpiexec it was configured against directly in
   # petsc.pc; use it to launch the parallel tests so they run against the same
   # MPI implementation PETSc was built with. Older petsc.pc files don't carry
   # this variable, in which case we fall back to a plain mpirun on PATH.
   pkg_get_variable(PETSC_MPIEXEC PETSc mpiexec)
   if(PETSC_MPIEXEC)
      set(MPIRUN_PATH "${PETSC_MPIEXEC}")
      get_filename_component(MPI_BIN_DIR "${PETSC_MPIEXEC}" DIRECTORY)
   else()
      set(MPIRUN_PATH "mpirun")
   endif()

   # Compile the tests with the matching mpicc when we can locate it next to
   # mpiexec, otherwise rely on a plain mpicc on PATH.
   if(MPI_BIN_DIR AND EXISTS "${MPI_BIN_DIR}/mpicc")
      set(PARMGMC_CC "${MPI_BIN_DIR}/mpicc")
   else()
      set(PARMGMC_CC "mpicc")
   endif()
   message(STATUS "MPI launcher: ${MPIRUN_PATH}")
   message(STATUS "MPI C compiler: ${PARMGMC_CC}")

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
                COMMAND lit -j 4 "${CMAKE_CURRENT_BINARY_DIR}" -a --time-tests -DNP=1)
   add_custom_target(check-par
                COMMENT "Running parallel tests"
                COMMAND lit -j 2 "${CMAKE_CURRENT_BINARY_DIR}" -a --time-tests --filter-out=ex6.c -DNP=4)

   add_custom_target(check
                DEPENDS check-par check-seq)
else()
   file(GLOB ex_files "ex*.c")
   message(STATUS "Found example files ${ex_files}")
   foreach(file ${ex_files})
      get_filename_component(test_name ${file} NAME_WE)
      add_executable(${test_name} ${file})
      target_include_directories(${test_name} PRIVATE ../include)
      target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -pedantic)
      target_link_libraries(${test_name} PRIVATE parmgmc m)
   endforeach()
endif()
