#!/usr/bin/env python
# coding: utf-8

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


# Source data has below fields: (Time period: Jan-01 to June-22)
# 
# Gold price,dow futures, Nasdaq, Oil_Price_WTI (All in USD currency). 

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv('../input/gold-oil-nasdaq-dow-sp-covid/oil_gold_nasdaq_ytd.csv')
data.head()


# Plot values in line graph: 

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv('../input/gold-oil-nasdaq-dow-sp-covid/oil_gold_nasdaq_ytd.csv')
data.head()


fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
 data1 = data[data.columns[i+1]]
 ax.plot(data1, color='green', linewidth=1)
 ax.set_title(data.columns[i+1])
 ax.xaxis.set_ticks_position('none')
 ax.yaxis.set_ticks_position('none')
 ax.spines["top"].set_alpha(0)
 ax.tick_params(labelsize=6)
plt.tight_layout();


# Correlation analysis in heat map:  
# 
# Here is the significant correlation: 
# 
# **Oil Vs Gold price       : -0.84
# Oil Vs Dow future       : 0.35
# Nasdaq Vs Dow futures   : 0.73
# **
# 

# In[ ]:


import seaborn as sns
data.head()
data1=data[['Dow_futures','gold_price','nasdaq','Oil_price']]
corr=data1.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,annot=True, annot_kws={'size':10})
heat_map=plt.gcf()
heat_map.set_size_inches(10,6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

