#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport


# In[ ]:


get_ipython().system(' conda install hvplot -y')
import hvplot.pandas


# In[ ]:



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv',
                    index_col="Id")
train['Date'] = pd.to_datetime(train.Date).dt.date
display(train.head())


test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv',
                   index_col="ForecastId")
test['Date'] = pd.to_datetime(test.Date).dt.date
display(test.head())


# In[ ]:


print(f'Training dates range from {train.Date.min()} to {train.Date.max()} \n'
      f'Prediction dates range from {test.Date.min()} to {test.Date.max()}')


# Note - the dates above don't seem to make sense.

# In[ ]:


train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile


# In[ ]:


# Get all the geographies
list(train.groupby(['Country_Region', 'Province_State']).groups)


# In[ ]:


# Focus on Hong Kong
train_HK = train[train.Province_State == 'Hong Kong'].copy()
train_HK['NewCases'] = train_HK.ConfirmedCases-train_HK.ConfirmedCases.shift(1)
train_HK


# In[ ]:


total = train_HK.hvplot.line(x='Date', y='ConfirmedCases')
new = train_HK.hvplot.line(x='Date', y='NewCases')

(total+new).cols(1)


# In[ ]:


t = train_HK.hvplot.line(x='Date', y='ConfirmedCases')
n = train_HK.hvplot.line(x='Date', y='Fatalities')

(t+n).cols(1)


# In[ ]:


train_HK.hvplot.hist(y='NewCases')


# In[ ]:


train.loc[(train.Country_Region == 'China') & (train.Province_State != 'Hubei')].hvplot.line(x='Date', y='ConfirmedCases', by='Province_State')


# In[ ]:


train.loc[(train.Country_Region == 'China') & (train.Province_State != 'Hubei')].hvplot.violin(y='ConfirmedCases',
                                                                                    by='Province_State', rot=45)


# In[ ]:


train.loc[(train.Country_Region == 'US') & (~train.Province_State.str.contains('New', na=False))].hvplot.violin(y='ConfirmedCases',
                                                                                    by='Province_State', rot=45, width=700, height=300)


# In[ ]:


max_dead = train.groupby('Country_Region')['Fatalities'].max()
max_dead[max_dead>500].hvplot.bar(rot=65, width=600, height=350)


# In[ ]:




