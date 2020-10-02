#!/usr/bin/env python
# coding: utf-8

# # ION Switching
# ### Identify the # of channels open at each time point
# #### Competition Intro
# > Many diseases, including cancer, are believed to have a *contributing factor in common.* <br>
# > **Ion channels** are pore-forming proteins present in animals and plants. They encode learning and memory, help fight infections, enable pain signals, and stimulate muscle contraction. If scientists  ion ccould better studyhannels, which may be possible with the aid of machine learning, it could have a far-reaching impact. <br>
# > When ion channels open, they** pass electric currents**. ... Scientists hope that technology could enable **rapid automatic detection of ion channel current events in raw data.** <br>
# > ... use **ion channel data** to better model **automatic identification** methods. If successful, youll be able to **detect individual ion channel events in noisy raw signals.** The data is simulated and injected with real world noise to emulate what scientists observe in laboratory experiments.
# 
# #### Evaluation
# - [macro F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# - In "macro" F1 a separate F1 score is calculated for each open_channels value and then averaged.
# - Submission: **For each time value** in the test set, you must **predict open_channels**. The files must have a header and should look like the following:
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Data Description
# - will be predicting the number of open_channels present, based on **electrophysiological signal data**
# - While the time series appears continuous, the data is **from discrete batches of 50 seconds long 10 kHz samples** (500,000 rows per batch). In other words, the *data from 0.0001 - 50.0000 is a different batch than 50.0001 - 100.0000*, and *thus discontinuous between 50.0000 and 50.0001*.
# - [Deep-Channel uses deep neural networks to detect single-molecule events from patch-clamp data](https://www.nature.com/articles/s42003-019-0729-3)

# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


print(train.head(2), '\n', test.head(2), '\n', submission.head(2))


# # EDA
# ### Reference: [EDA-Ion Switching by Peter](https://www.kaggle.com/pestipeti/eda-ion-switching)

# In[ ]:


import math
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

pd.set_option("display.precision", 8)

DIR_INPUT = '/kaggle/input/liverpool-ion-switching'


# In[ ]:


print(train.shape, test.shape, submission.shape)
print(train.columns,'\n', test.columns,'\n', submission.columns)


# #### Columns in Train set
# * time<br>
# total 5,000,000 rows(time points): 50 seconds long 10kHz samples (500,000 rows per batch). So there are 10 batches. Data is continuous in each batch but discontinuous between batches.
# * signal<br>
# electrophysiological signal data
# * open channels<br>
# 11 possible values (0~10)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




