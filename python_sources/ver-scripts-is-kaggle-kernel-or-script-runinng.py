import os
from pprint import pprint, pformat

print(pformat(os.environ))
print('-- List -- ')
script_keys = sorted(list(os.environ.keys()))

print(pformat(script_keys))
print(len(script_keys))

kernel_keys = ['CLICOLOR',
 'GIT_PAGER',
 'HOME',
 'HOSTNAME',
 'JPY_PARENT_PID',
 'JUPYTER_CONFIG_DIR',
 'KAGGLE_DATASET_PATH',
 'KAGGLE_DATA_PROXY_PROJECT',
 'KAGGLE_DATA_PROXY_TOKEN',
 'KAGGLE_DATA_PROXY_URL',
 'KAGGLE_GYM_DATASET_PATH',
 'KAGGLE_KERNEL_INTEGRATIONS',
 'KAGGLE_KERNEL_RUN_TYPE',
 'KAGGLE_URL_BASE',
 'KAGGLE_USER_SECRETS_TOKEN',
 'KAGGLE_WORKING_DIR',
 'LANG',
 'LC_ALL',
 'LD_LIBRARY_PATH',
 'MKL_THREADING_LAYER',
 'MPLBACKEND',
 'PAGER',
 'PATH',
 'PROJ_LIB',
 'PWD',
 'PYTHONPATH',
 'PYTHONUSERBASE',
 'SHLVL',
 'TERM',
 'TESSERACT_PATH',
 '_']

script_only = set(script_keys) - set(kernel_keys)
kernel_only = set(kernel_keys) - set(script_keys)
script_and_kernel = set(kernel_keys) & set(script_keys)

print(script_only)
print(kernel_only)
print(script_and_kernel)

def is_script_running():
    kernel_only_env_keys = {'TERM', 'PAGER', 'GIT_PAGER', 'JPY_PARENT_PID', 'CLICOLOR'}
    kernel_keys = set(os.environ.keys())
    is_kernel = kernel_only_env_keys.issubset(kernel_keys)
    return not is_kernel

print(is_script_running())