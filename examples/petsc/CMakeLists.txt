add_executable(petsc_test test.cc)

target_compile_options(petsc_test PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(petsc_test parmgmc)
