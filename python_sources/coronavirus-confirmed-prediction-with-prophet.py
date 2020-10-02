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


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv',parse_dates=['Last Update'])
df.head()


# In[ ]:


df


# In[ ]:


get_ipython().system('ls ../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# In[ ]:


df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# In[ ]:


df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# In[ ]:


confirmed = df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed']].sum().reset_index()
confirmed.columns=['ds','y']
confirmed['ds'] = confirmed['ds'].dt.date
confirmed


# In[ ]:


from fbprophet import Prophet
m = Prophet()
m.fit(confirmed)


# In[ ]:


future = m.make_future_dataframe(periods=30)
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


fig1 = m.plot(forecast)


# In[ ]:





# In[ ]:




