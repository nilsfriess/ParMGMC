find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

find_package(pybind11 REQUIRED)
pybind11_add_module(pymgmc main.cc)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(pymgmc PRIVATE ${PETSC4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "petsc4py could not be found.")
endif()

target_include_directories(pymgmc PUBLIC ${PETSC_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ../include)
target_link_libraries(pymgmc PUBLIC pybind11::module parmgmc)
