import lit.formats
import os

config.name = 'ParMGMC'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.c']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root)

# lit sanitizes the environment; forward the variables Open MPI needs.
# Without HOME, opal_init aborts with "Unable to get the user home directory".
for var in ['HOME', 'TMPDIR', 'OMP_NUM_THREADS', 'OMPI_ALLOW_RUN_AS_ROOT', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM', 'OMPI_MCA_rmaps_base_oversubscribe']:
    if var in os.environ:
        config.environment[var] = os.environ[var]

config.substitutions.append(('%cc', config.parmgmc_cc))
config.substitutions.append(('%flags', config.parmgmc_comp))
config.substitutions.append(('%mpirun', config.parmgmc_mpirun))

try:
    NP = lit_config.params['NP']
except KeyError as e:
    NP = 1
config.substitutions.append(('%NP', NP))

try:
    opts = lit_config.params['opts']
except KeyError as e:
    opts = ""

config.substitutions.append(('%opts', opts))
