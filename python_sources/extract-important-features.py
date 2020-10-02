#!/usr/bin/env python
# coding: utf-8

# Exploratory Analysis of Two Sigma 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")


# In[ ]:


len(df)
import kagglegym
# Create environment
env = kagglegym.make()
# Get first observation
observation = env.reset()
#Observations are the means by which our code "observes" the world. 
#The very first observation has a special property called "train" 
#which is a dataframe which we can use to train our model:
# Look at first few rows of the train dataframe
observation.train


# In[ ]:


df_fund=observation.train.filter(regex='fundamental',axis=1)
df_fund.fillna(df_fund.mean(),inplace=True)


# In[ ]:


df_fund.head()


# In[ ]:


#extract important fundamental features
df_fund['fund_y']=observation.train['y']
df_fund.corr()['fund_y']
a=df_fund.corr()['fund_y'].sort_values(ascending=False).head(5)
del a['fund_y']
imp_fund_features=a.index.tolist()
a


# In[ ]:


#extract important technical features
df_tech=observation.train.filter(regex='technical',axis=1)
df_tech['tech_y']=observation.train['y']
df_tech.corr()['tech_y']
b=df_tech.corr()['tech_y'].sort_values(ascending=False).head(5)
del b['tech_y']
imp_tech_features=b.index.tolist()
b


# In[ ]:


#extract important derived features
df_der=observation.train.filter(regex='derived',axis=1)
df_der['der_y']=observation.train['y']
df_der.corr()['der_y']
c=df_der.corr()['der_y'].sort_values(ascending=False).head(5)
del c['der_y']
c


# In[ ]:


import numpy as np
from sklearn import linear_model
regr = linear_model.LinearRegression()


# In[ ]:


#plot the important features

full_df = pd.read_hdf('../input/train.h5')
pd.options.mode.chained_assignment = None  # default='warn'
id = 1561 #val_set.id.sample().values[0]
print(id)
temp = full_df[full_df.id==id]
temp['feature'] = temp['fundamental_11']
temp['feature'] = temp['feature'] * 4
temp[['y', 'feature']].iloc[:100,:].plot(marker='.')


# In[ ]:


full_df = pd.read_hdf('../input/train.h5')
pd.options.mode.chained_assignment = None  # default='warn'
id = 1561 #val_set.id.sample().values[0]
print(id)
temp = full_df[full_df.id==id]
temp['feature'] = temp['technical_20']
temp['feature'] = temp['feature'] * 4
temp[['y', 'feature']].iloc[:100,:].plot(marker='.')


# In[ ]:


full_df = pd.read_hdf('../input/train.h5')
pd.options.mode.chained_assignment = None  # default='warn'
id = 1561 #val_set.id.sample().values[0]
print(id)
temp = full_df[full_df.id==id]
temp['feature'] = temp['derived_2']
temp['feature'] = temp['feature'] * 4
temp[['y', 'feature']].iloc[:100,:].plot(marker='.')


# In[ ]:




