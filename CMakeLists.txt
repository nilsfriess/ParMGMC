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

find_package(FFTW)
message(STATUS ${FFTW_INCLUDE_DIRS} ${FFTW_LIBRARIES})
if (DEFINED FFTW_INCLUDE_DIRS AND NOT DEFINED FFTW_LIBRARIES)
  message(STATUS "Found FFTW include dirs but no libs, assuming that Intel MKL FFTW is used")
  find_package(MKL CONFIG REQUIRED)
  set(FFTW_DOUBLE_LIB "") # TODO: If we set this to $<LINK_ONLY:MKL::MKL> or MKL::MKL, then the program segfaults. Just leaving it empty seems to work (maybe because PETSc is linked against MKL already?).
endif()

add_library(parmgmc SHARED)
file(GLOB_RECURSE PARMGMC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/parmgmc/*.h")
message(STATUS ${PARMGMC_INCLUDES})
target_sources(parmgmc
	PRIVATE
	    src/pc_hogwild.c
	    src/pc_mcgibbs.c
	    src/pc_gamgmc.c
	    src/pc_chols.c
	    src/ziggurat.c
	    src/mc_sor.c
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
target_link_libraries(parmgmc PUBLIC PkgConfig::PETSC MPI::MPI_C )

add_executable(benchmark examples/benchmark/main.cc)
target_include_directories(benchmark PRIVATE include)
target_compile_options(benchmark PRIVATE -Wall -Wextra -Wpedantic -pedantic)
target_link_libraries(benchmark PRIVATE parmgmc m MPI::MPI_CXX MPI::MPI_C ${FFTW_DOUBLE_LIB})

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

