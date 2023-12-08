add_library(parmgmc
  parmgmc/lattice/lattice.cc
  parmgmc/mpi_helper.cc
  parmgmc/common/helpers.cc)

if (${USE_MPI})
  target_compile_definitions(parmgmc PUBLIC USE_MPI=true)
else()
  target_compile_definitions(parmgmc PUBLIC USE_MPI=false)
  target_include_directories(parmgmc SYSTEM PUBLIC "${CMAKE_SOURCE_DIR}/external")
endif()

# TODO: Make this more robust/ use a FindMETIS.cmake from somewhere
if (${USE_METIS})
   target_compile_definitions(parmgmc PUBLIC USE_METIS=true)

   target_include_directories(parmgmc SYSTEM PUBLIC "${METIS_ROOT}/include/")
   target_link_directories(parmgmc PUBLIC "${METIS_ROOT}/lib")
   target_link_libraries(parmgmc PUBLIC metis GKlib)
endif()

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc>=3.1)

target_include_directories(parmgmc PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_compile_options(parmgmc PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(parmgmc PUBLIC Eigen3::Eigen ${MPI_TARGET} PkgConfig::PETSc)
