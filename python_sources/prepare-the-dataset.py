#!/usr/bin/env python
# coding: utf-8

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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import seaborn as sns
df_gov_action = pd.read_excel('/kaggle/input/oxford-covid19-government-response-tracker/OxCGRT_Download_latest_data.xlsx')


# In[ ]:


df_gov_action.info()


# In[ ]:


df_gov_action.head()


# In[ ]:


df_gov_action['Date'].unique().size


# In[ ]:


df_gov_action['Date']=pd.to_datetime(df_gov_action['Date'], format='%Y%m%d', errors='ignore')


# In[ ]:


from datetime import datetime
df_gov_action['month_year']=[x.strftime('%Y%m') for x in df_gov_action['Date']]


# In[ ]:


df_gov_action['CountryName']=[ x.strip().upper()  for x in df_gov_action['CountryName'] ]


# In[ ]:


df_gov_action.CountryName.unique().size


# In[ ]:


df_gov_action.fillna(0,inplace=True)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')


# In[ ]:


df_train.shape


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train.Country_Region.unique().size


# In[ ]:


df_train['Date']=pd.to_datetime(df_train['Date'], format='%Y-%m-%d', errors='ignore')
df_test['Date']=pd.to_datetime(df_test['Date'], format='%Y-%m-%d', errors='ignore')


# In[ ]:


df_train['month_year']=[x.strftime('%Y%m') for x in df_train['Date']]
df_test['month_year']=[x.strftime('%Y%m') for x in df_test['Date']]


# In[ ]:


df_train['Country_Region']=[ x.strip().upper()  for x in df_train['Country_Region'] ]
df_test['Country_Region']=[ x.strip().upper()  for x in df_test['Country_Region'] ]


# In[ ]:


#df_train_extd=df_train.merge(df_gov_action,how='left',left_on=['Country_Region','month_year'],right_on=['CountryName','month_year'])
#df_test_extd=df_test.merge(df_gov_action,how='left',left_on=['Country_Region','month_year'],right_on=['CountryName','month_year'])

df_train_extd=df_train.merge(df_gov_action,how='left',left_on=['Country_Region','Date'],right_on=['CountryName','Date'])
df_test_extd=df_test.merge(df_gov_action,how='left',left_on=['Country_Region','Date'],right_on=['CountryName','Date'])


# In[ ]:


#date
df_train_extd.isna().sum()
df_test_extd.isna().sum()


# In[ ]:


#month
df_train_extd.isna().sum()
df_test_extd.isna().sum()


# In[ ]:


df_train_extd.Country_Region.unique().size


# In[ ]:


df_train_extd.dropna().Country_Region.unique()


# In[ ]:




