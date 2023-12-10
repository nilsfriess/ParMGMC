include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)
cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

add_executable(petsc_test test.cc)

target_include_directories(petsc_test SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
target_compile_options(petsc_test PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(petsc_test parmgmc)
