#!/usr/bin/env python
# coding: utf-8

# # Feather format for super fast data loading
# 
# Original `panquet` format takes time to load data. Here I converted them and uploaded with `feather` format.<br/>
# It is about **30 times faster**.
# 
# You can see dataset here: [https://www.kaggle.com/corochann/bengaliaicv19feather](https://www.kaggle.com/corochann/bengaliaicv19feather)<br/>
# Please upvote both dataset and this kernel if you like it! :)
# 
# This kernel describes how to load this dataset.

# # How to add dataset
# 
# When you write kernel, click "+ Add Data" botton on right top.<br/>
# Then inside window pop-up, you can see "Search Datasets" text box on right top.<br/>
# You can type "bengaliai-cv19-feather" to find this dataset and press "Add" botton to add the data.

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/bengaliai-cv19')\n\n# Read in the data CSV files\ntrain = pd.read_csv(datadir/'train.csv')\ntest = pd.read_csv(datadir/'test.csv')\nsample_submission = pd.read_csv(datadir/'sample_submission.csv')\nclass_map = pd.read_csv(datadir/'class_map.csv')")


# To load `feather` format, we just need to change `read_parquet` to `read_feather`.
# 
# Original `parquet` format takes about 60 sec to load 1 data, while `feather` format takes about **2 sec to load 1 data!!!**

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_image_df0 = pd.read_parquet(datadir/'train_image_data_0.parquet')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "featherdir = Path('/kaggle/input/bengaliaicv19feather')\n\ntrain_image_df0 = pd.read_feather(featherdir/'train_image_data_0.feather')\ntrain_image_df1 = pd.read_feather(featherdir/'train_image_data_1.feather')\ntrain_image_df2 = pd.read_feather(featherdir/'train_image_data_2.feather')\ntrain_image_df3 = pd.read_feather(featherdir/'train_image_data_3.feather')")


# For test files, please be careful that this is **code competition** and **test data will change in the actual submission**. <br/>
# So I guess we need to load from original `parquet` format to load private test data when submission.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Please change this to `True` when actual submission\nsubmission = False\n\nif submission:\n    test_image_df0 = pd.read_parquet(datadir/'test_image_data_0.parquet')\n    test_image_df1 = pd.read_parquet(datadir/'test_image_data_1.parquet')\n    test_image_df2 = pd.read_parquet(datadir/'test_image_data_2.parquet')\n    test_image_df3 = pd.read_parquet(datadir/'test_image_data_3.parquet')\nelse:\n    test_image_df0 = pd.read_feather(featherdir/'test_image_data_0.feather')\n    test_image_df1 = pd.read_feather(featherdir/'test_image_data_1.feather')\n    test_image_df2 = pd.read_feather(featherdir/'test_image_data_2.feather')\n    test_image_df3 = pd.read_feather(featherdir/'test_image_data_3.feather')")


# In[ ]:


train_image_df0.head()

