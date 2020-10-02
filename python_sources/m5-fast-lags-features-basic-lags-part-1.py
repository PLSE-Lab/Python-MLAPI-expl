#!/usr/bin/env python
# coding: utf-8

# In a [previous notebook](https://www.kaggle.com/chrisrichardmiles/m5-fast-lags-features) I developed some very efficient functions that can can be used to develop features based on previous sales. The great benefit of these functions is that the features are generated in such a way that all features are aligned on the same index, which makes combing features very easy later on. I added these functions to my [helper file](https://www.kaggle.com/chrisrichardmiles/m5-helpers/edit/run/35754663). This notebook will be the first in a series of notebooks where we utilize these functions to create a huge amount of lags features, broken into many pickle files. When we are done exhausting our ideas for lags features, we will try to bring them all together using PCA. 
# 
# We choose to break them into many pickle files to prevent our notebooks from crashing when we are loading the data. 

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


# In[ ]:


from m5_helpers import * 
import m5_helpers as h # to look at docs and options


# In[ ]:


################### Read in data ########################
DATA_PATH = '/kaggle/input/m5-forecasting-accuracy/'
train_df, cal_df, prices_df, ss = read_data(DATA_PATH)


# In[ ]:


get_ipython().run_cell_magic('time', '', '################ Initial grid_df #################\n# We will make an intial grid. This way, when we\n# add features onto it, so that all our features \n# will share the same index. The rec is a \n# rectangle of values, simply sales data as \n# it is in train_df but as a numpy array for \n# fast computations. \ngrid_df, rec = make_grid_df(train_df)\noriginal_cols = grid_df.columns\n\n# rec.shape is num_items by num_days\nrec.shape')


# In[ ]:


################# Basic lags 1-15 ##################
start_time = time()
add_lags(grid_df, lags=range(1, 15))

print(f'{(time() - start_time):.2f} seconds to add lags 1_15')
print(f'Size of new features dataframe: {sizeof_fmt(grid_df.memory_usage().sum())}')

start_time = time()
################# Save new features ################
###### Saving space ######
# Since our features are aligned to the grid_df 
# index, we don't need to keep the id, d, and sales 
# columns with these features. We will save about 
# .6 Gigabytes by removing these columns. with a 
# final size of 1.7 GiB, we shouldn't have problems 
# reading the features in for modeling in other 
# notebooks. 
keep_cols = [col for col in list(grid_df) if col not in original_cols]
print(f'Size of new features dataframe without id, d, and sales columns: {sizeof_fmt(grid_df[keep_cols].memory_usage().sum())}')
grid_df[keep_cols].to_pickle('fe_lags_1_14.pkl')

print(f'{(time() - start_time):.2f} seconds to dump lags 1_15 to pickle')

################ Reset grid_df #####################
grid_df = grid_df[original_cols]


# In[ ]:


get_ipython().run_cell_magic('time', '', "################ Add more lags, 14 at a time #################\nfor i in [15, 29]: \n    add_lags(grid_df, lags=range(i, i + 14))\n    keep_cols = [col for col in list(grid_df) if col not in original_cols]\n    grid_df[keep_cols].to_pickle(f'fe_lags_{i}_{i + 13}.pkl')\n\n    ################ Reset grid_df #####################\n    grid_df = grid_df[original_cols]")


# In[ ]:


############### no more disk space #######################
# 4.7 GiB is about all we can fit on the output of a 
# notbook. From what I have seen, this amount of lags 


# In[ ]:


df1 = pd.read_pickle('fe_lags_1_14.pkl')
df1.info()


# In[ ]:


df15 = pd.read_pickle('fe_lags_15_28.pkl')
df15.info()


# In[ ]:


df29 = pd.read_pickle('fe_lags_29_42.pkl')
df29.info()

