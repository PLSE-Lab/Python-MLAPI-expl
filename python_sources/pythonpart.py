#!/usr/bin/env python
# coding: utf-8

# In[94]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.artist as fig

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[95]:


source = pd.read_csv('../input/dohodi3.csv',';', index_col=['Year'],parse_dates=['Year'],dayfirst=True)
source


# In[96]:


val=source.Value
val


# In[97]:


val.plot(figsize=(12,6))


# In[98]:


itog = val.describe()
val.hist()
itog


# In[99]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(val.values.squeeze(), lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(val, lags=20, ax=ax2)


# In[105]:


src_data_model = val
src_data_model
model = sm.tsa.ARIMA(src_data_model, order=(1,0,2)).fit(full_output=True, disp=0)
model


# In[106]:


#model.summary()


# In[ ]:




