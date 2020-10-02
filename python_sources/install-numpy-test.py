import pip._internal as pip

pip.main(['show', 'numpy'])
pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np
print('> NumPy version: {}'.format(np.__version__))
