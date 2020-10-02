#!/usr/bin/env python
# coding: utf-8

# 

# 
# Based on BrendanGallegoBailey Data Science for Good City of LA Starter Kernel: 
# https://www.kaggle.com/bbailey/data-science-for-good-city-of-la-starter-kernel
# 
# Prophet link: 
# https://github.com/facebook/prophet/blob/master/notebooks/non-daily_data.ipynb
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


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"
data_list = []
for filename in os.listdir(bulletin_dir):
    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:
        for line in f.readlines():
            #Insert code to parse job bulletins
            if "Open Date:" in line:
                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()
        data_list.append([filename, job_bulletin_date])


# In[ ]:


df = pd.DataFrame(data_list)
df.columns = ["FILE_NAME", "OPEN_DATE"]
df["OPEN_DATE"] = df["OPEN_DATE"].astype('datetime64[ns]')
df.info()


# In[ ]:


data = df.groupby('OPEN_DATE').count()


# In[ ]:


from fbprophet import Prophet


# In[ ]:


data.index


# In[ ]:


data['FILE_NAME'].values.astype(int)


# In[ ]:


df = pd.DataFrame()
df['ds'] = data.index
df['y'] = data['FILE_NAME'].values.astype(int)
df = df.dropna()


# In[ ]:


m = Prophet(changepoint_prior_scale=0.01).fit(df)
future = m.make_future_dataframe(periods=300, freq='H')
fcst = m.predict(future)
fig = m.plot(fcst)


# In[ ]:


fig = m.plot_components(fcst)


# In[ ]:


m = Prophet(seasonality_mode='multiplicative').fit(df)
future = m.make_future_dataframe(periods=3652)
fcst = m.predict(future)
fig = m.plot(fcst)


# In[ ]:


m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(df)
fcst = m.predict(future)
fig = m.plot_components(fcst)

