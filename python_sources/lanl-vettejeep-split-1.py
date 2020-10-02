#!/usr/bin/env python
# coding: utf-8

# This is a part of a series of copy or reproduction of below kernel  
# Ref: https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392  
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")


# In[ ]:


OUTPUT_DIR = ''
DATA_DIR = '../input/'

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500


# In[ ]:


df = pd.read_csv(os.path.join('../input/train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
max_start_index = len(df.index) - SIG_LEN
slice_len = int(max_start_index / 6)

# for i in range(NUM_THREADS):
for i in range(3):
    print('working', i)
    df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
    df0.to_csv('raw_data_%d.csv' % i, index=False)
    del df0

del df


# In[ ]:


# Verify
temp = pd.read_csv('raw_data_0.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(temp.shape)
display(temp.head(5))


# In[ ]:


# Verify
temp = pd.read_csv('raw_data_1.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(temp.shape)
display(temp.head(5))

