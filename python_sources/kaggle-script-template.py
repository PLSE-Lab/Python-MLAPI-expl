import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {'easy_gold/__init__.py': 'H4sIAMo3fFwC/wMAAAAAAAAAAAA=', 'easy_gold/main.py': 'H4sIAMo3fFwC/0tJTVPITczM09C04lIAgoKizLwSDXWP1JycfIXy/KKcFEV1TS4ursw0hfj4vMTc1Ph4BVtbBfX4eJCu+Hh1iDaIEVwAfxLL9k4AAAA=', 'setup.py': 'H4sIAMo3fFwC/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiHBxgSkNLgUgyEvMTbVVT00sroxPz89JUdcBixYkJmcnpqcW20YjScXqcGlyAQA1QoAFWQAAAA=='}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python easy_gold/main.py')
