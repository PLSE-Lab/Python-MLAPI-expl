#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('../input/covid19-in-usa/us_covid19_daily.csv')


# In[ ]:


data.head(12)


# In[ ]:


data.shape


# In[ ]:


yesterday = datetime.now() 
# datetime.now() starts with monday at 0 but for my analysis I will start at 1, in addition data given is one day behind present day 
yesterday.weekday() 


# In[ ]:


datacopyy= data.copy()
weekday = yesterday.weekday()  # add one since datime module starts with monday at zero and since data is being updated constantly
j= weekday #since first entry for date is a Tuesday (June 15 2020)
for i in range(datacopyy.shape[0]):   
    j= j%7 
    datacopyy.loc[[i], ['date']]=j
    j=j-1 # since data in decreasing order

datacopy = datacopyy.head(65)  #so data from the last 65 days since the dataset contains info from the last 143 days
datacopy.head(12)
    


# In[ ]:


#find the average number of people on ventilators per weekday:
# average number of people hospitalized increase per weekday ''
# average number of people that die more on days 'deathIncrease'


# In[ ]:



df_by_date = datacopy.groupby(['date']).mean()
df_by_date


# **NOTE**:
# 0 = Sunday, 1 = Monday, 2= Tuesday, 3=Wednesday, 4 = Thursday, 5=Friday, 6 = Saturday
# 

# In[ ]:


df_by_date.columns


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
### AVERAGE NUMBER OF PEOPLE IN ICU PER WEEKDAY
#plt.plot([0,1,2,3,4,5,6], df_by_date['onVentilatorCurrently'])
date = [0,1,2,3,4,5,6]
df1 = pd.DataFrame({'date': date,
                   'inIcuCurrently': df_by_date["inIcuCurrently"]})

plt.plot(date, df_by_date["inIcuCurrently"], 'o', color='black');df1.plot.bar(rot=0)


# There seems to be an increase number of people in ICU on Thursdays

# In[ ]:


##Number of People currently hospitalized per weekday
df2 = pd.DataFrame({'date': date,
                   'hospitalizedCurrently': df_by_date["hospitalizedCurrently"]})
plt.plot(date, df_by_date['hospitalizedCurrently'], 'o', color='black');df2.plot.bar(rot=0)


# In[ ]:


##Number of People died per weekday
df3 = pd.DataFrame({'date': date,
                   'deathIncrease': df_by_date["deathIncrease"]})
plt.plot(date, df_by_date['deathIncrease'], 'o', color='black');df3.plot.bar(rot=0)


# Wednesday and Thursdays have the most increase in covid related deaths

# In[ ]:


## average number of people on ventilator per weekday
df4 = pd.DataFrame({'date': date,
                   'onVentilatorCurrently': df_by_date["onVentilatorCurrently"]})
plt.plot(date, df_by_date['onVentilatorCurrently'], 'o', color='black');df4.plot.bar(rot=0)


#  Most people on Ventilators on Thursdays

# 
# 
# ## Increase in number of hospitalizations per weekday
# df5 = pd.DataFrame({'date': date,
#                    'hospitalizedIncrease': df_by_date['hospitalizedIncrease']})
# plt.plot(date, df_by_date['hospitalizedIncrease'], 'o', color='black');df5.plot.bar(rot=0)

# In[ ]:


## Increase in number of hospitalizations per weekday
df5 = pd.DataFrame({'date': date,
                   'hospitalizedIncrease': df_by_date['hospitalizedIncrease']})
plt.plot(date, df_by_date['hospitalizedIncrease'], 'o', color='black');df5.plot.bar(rot=0)


# Spike in hospitalizations on Tuesdays and Fridays

# IDEA: Large increase in hospitalizations on Fridays, reasonable to believe this is due to people being off work and going to check or get treated for covid related symptoms. What if we change the "weekend" this can flatten the distribution to make caring for patients more manageable and discourage people from meeting up in groups since people normally do that on the weekend

# 
