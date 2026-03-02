#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <petsc4py/petsc4py.h>
#include <petscmat.h>
#include <petscsystypes.h>

#define VERIFY_PETSC4PY_FROMPY(func) \
  if (!func) { \
    if (import_petsc4py() != 0) return false; \
  }

#define VERIFY_PETSC4PY_FROMCPP(func) \
  if (!func) { \
    if (import_petsc4py() != 0) return {}; \
  }

// Macro for casting between PETSc and petsc4py objects
namespace pybind11::detail
{
template <>
struct type_caster<_p_Mat> {
public:
  PYBIND11_TYPE_CASTER(Mat, const_name("mat"));

  bool load(handle src, bool)
  {
    VERIFY_PETSC4PY_FROMPY(PyPetscMat_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyPetscMat_Type) != 0) {
      value = PyPetscMat_Get(src.ptr());
      return true;
    } else return false;
  }

  static handle cast(Mat src, return_value_policy policy, handle)
  {
    VERIFY_PETSC4PY_FROMCPP(PyPetscMat_New);
    if (policy == return_value_policy::take_ownership) {
      PyObject *obj = PyPetscMat_New(src);
      PetscCallAbort(MPI_COMM_WORLD, PetscObjectDereference((PetscObject)src));
      return {obj};
    } else if (policy == return_value_policy::automatic_reference or policy == return_value_policy::reference) {
      PyObject *obj = PyPetscMat_New(src);
      return {obj};
    } else {
      return {};
    }
  }

  operator Mat() { return value; }
};

template <>
struct type_caster<_p_Vec> {
public:
  PYBIND11_TYPE_CASTER(Vec, const_name("vec"));

  bool load(handle src, bool)
  {
    VERIFY_PETSC4PY_FROMPY(PyPetscVec_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyPetscVec_Type) != 0) {
      value = PyPetscVec_Get(src.ptr());
      return true;
    } else return false;
  }

  static handle cast(Vec src, return_value_policy policy, handle)
  {
    VERIFY_PETSC4PY_FROMCPP(PyPetscVec_New);
    if (policy == return_value_policy::take_ownership) {
      PyObject *obj = PyPetscVec_New(src);
      PetscCallAbort(MPI_COMM_WORLD, PetscObjectDereference((PetscObject)src));
      return {obj};
    } else if (policy == return_value_policy::automatic_reference or policy == return_value_policy::reference) {
      PyObject *obj = PyPetscVec_New(src);
      return {obj};
    } else {
      return {};
    }
  }

  operator Vec() { return value; }
};

template <>
struct type_caster<_p_DM> {
public:
  PYBIND11_TYPE_CASTER(DM, const_name("dm"));

  bool load(handle src, bool)
  {
    VERIFY_PETSC4PY_FROMPY(PyPetscDM_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyPetscDM_Type) != 0) {
      value = PyPetscDM_Get(src.ptr());
      return true;
    } else return false;
  }

  static handle cast(DM src, return_value_policy policy, handle)
  {
    VERIFY_PETSC4PY_FROMCPP(PyPetscDM_New);
    if (policy == return_value_policy::take_ownership) {
      PyObject *obj = PyPetscDM_New(src);
      PetscCallAbort(MPI_COMM_WORLD, PetscObjectDereference((PetscObject)src));
      return {obj};
    } else if (policy == return_value_policy::automatic_reference or policy == return_value_policy::reference) {
      PyObject *obj = PyPetscDM_New(src);
      return {obj};
    } else {
      return {};
    }
  }

  operator DM() { return value; }
};

template <>
struct type_caster<_p_PC> {
public:
  PYBIND11_TYPE_CASTER(PC, const_name("pc"));

  bool load(handle src, bool)
  {
    VERIFY_PETSC4PY_FROMPY(PyPetscPC_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyPetscPC_Type) != 0) {
      value = PyPetscPC_Get(src.ptr());
      return true;
    } else return false;
  }

  static handle cast(PC src, return_value_policy policy, handle)
  {
    VERIFY_PETSC4PY_FROMCPP(PyPetscPC_New);
    if (policy == return_value_policy::take_ownership) {
      PyObject *obj = PyPetscPC_New(src);
      PetscCallAbort(MPI_COMM_WORLD, PetscObjectDereference((PetscObject)src));
      return {obj};
    } else if (policy == return_value_policy::automatic_reference or policy == return_value_policy::reference) {
      PyObject *obj = PyPetscPC_New(src);
      return {obj};
    } else {
      return {};
    }
  }

  operator PC() { return value; }
};

template <>
struct type_caster<_p_PetscRandom> {
public:
  PYBIND11_TYPE_CASTER(PetscRandom, const_name("PetscRandom"));

  bool load(handle src, bool)
  {
    VERIFY_PETSC4PY_FROMPY(PyPetscRandom_Get);
    if (PyObject_TypeCheck(src.ptr(), &PyPetscRandom_Type) != 0) {
      value = PyPetscRandom_Get(src.ptr());
      return true;
    } else return false;
  }

  static handle cast(PetscRandom src, return_value_policy policy, handle)
  {
    VERIFY_PETSC4PY_FROMCPP(PyPetscRandom_New);
    if (policy == return_value_policy::take_ownership) {
      PyObject *obj = PyPetscRandom_New(src);
      PetscCallAbort(MPI_COMM_WORLD, PetscObjectDereference((PetscObject)src));
      return {obj};
    } else if (policy == return_value_policy::automatic_reference or policy == return_value_policy::reference) {
      PyObject *obj = PyPetscRandom_New(src);
      return {obj};
    } else {
      return {};
    }
  }

  operator PetscRandom() { return value; }
};

} // namespace pybind11::detail
