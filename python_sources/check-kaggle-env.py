import os
from pathlib import Path

import pandas as pd

KAGGLE_ENV_KEYS = {'KAGGLE_KERNEL_INTEGRATIONS', 'KAGGLE_DATA_PROXY_TOKEN',
                   'MPLBACKEND', 'KAGGLE_GYM_DATASET_PATH',
                   'KAGGLE_DATA_PROXY_URL', 'KAGGLE_DATA_PROXY_PROJECT',
                   'TESSERACT_PATH', 'HOSTNAME', 'PYTHONPATH',
                   'KAGGLE_KERNEL_RUN_TYPE', 'JUPYTER_CONFIG_DIR',
                   'PATH', 'LD_LIBRARY_PATH', 'KAGGLE_DATASET_PATH',
                   'MKL_THREADING_LAYER', 'PYTHONUSERBASE', 'LANG', 'PROJ_LIB',
                   'KAGGLE_WORKING_DIR', 'KAGGLE_URL_BASE', 'HOME', 'LC_ALL',
                   'KAGGLE_USER_SECRETS_TOKEN'}


def is_running_kaggle_kernel():
    current_running_kernel_keys = set(os.environ.keys())
    is_kaggle_kernel = KAGGLE_ENV_KEYS.issubset(current_running_kernel_keys)
    return is_kaggle_kernel


is_kaggle = is_running_kaggle_kernel()

if is_kaggle:
    root_path = Path("../input/titanic")
else:
    root_path = Path("resources/titanic")


def main():
    submit_path = root_path / "gender_submission.csv"
    submit = pd.read_csv(submit_path)
    if is_kaggle:
        submit.to_csv("submission.csv", index=False)
    else:
        submit.to_csv(root_path / "submission.csv", index=False)


if __name__ == '__main__':
    main()
