#include <cstdlib>
#include <stdexcept>
#include <utility>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include "pargibbs/mpi_helper.hh"

namespace pargibbs {
static bool mpi_is_initialised = false;

mpi_helper::mpi_helper(int *argc, char ***argv) {
  MPI_Init(argc, argv);
  mpi_is_initialised = true;
}

mpi_helper::~mpi_helper() { MPI_Finalize(); }

std::pair<int, int> mpi_helper::get_size_rank() {
  assert_initalised();
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return {size, rank};
}

int mpi_helper::get_rank() {
  assert_initalised();
  return get_size_rank().second;
}
int mpi_helper::get_size() {
  assert_initalised();
  return get_size_rank().first;
}

int mpi_helper::debug_rank() {
  assert_initalised();

  if (const char *env_rank = std::getenv("PARGIBBS_DEBUG_RANK"))
    return atoi(env_rank);
  else
    return 0;
}

bool mpi_helper::is_debug_rank() {
  assert_initalised();
  return get_rank() == debug_rank();
}

void mpi_helper::assert_initalised() {
  if (!mpi_is_initialised)
    throw std::runtime_error(
        "Construct a pargibbs::mpi_helper object in main().");
}

} // namespace pargibbs