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


Air_Quality = pd.read_excel("/kaggle/input/airquality/AirQuality.xlsx")


# In[ ]:


Air_Quality.head()


# In[ ]:


print(type(Air_Quality))
print(Air_Quality.columns)


# In[ ]:


Air_Quality


# In[ ]:


print(Air_Quality.index)


# In[ ]:


print(Air_Quality['Country'])


# In[ ]:


print(Air_Quality['State'])


# In[ ]:


print(Air_Quality.State.unique())


# In[ ]:


print(Air_Quality.Pollutants.unique())


# In[ ]:


print(Air_Quality.head())
print()
print()
print(Air_Quality.describe())


# **GROUPING BY STATES**

# In[ ]:


group_of_States = Air_Quality.groupby('State')
group_of_States.head()


# In[ ]:


Air_Quality.head(5)


# In[ ]:


group_of_States.head([5])


# In[ ]:


states_pollution = Air_Quality.groupby(['State','Pollutants'])


# In[ ]:


states_pollution.head()


# In[ ]:


state_quality = Air_Quality.groupby(['State', 'Pollutants'])


# In[ ]:


state_quality.head(10)


# **MEAN OF POLLUTION IN STATES**

# In[ ]:


Mean_Pollution = group_of_States.mean()
Mean_Pollution


# In[ ]:


Mean_Pollution.sort_values(by=['Avg'])


# **TYPE**

# In[ ]:


type(Mean_Pollution)


# In[ ]:


type(Air_Quality)


# In[ ]:


type(group_of_States)


# In[ ]:


type(state_quality)


# **MAKE DATAFRAME**

# In[ ]:


group_of_States_df = pd.DataFrame(group_of_States)
Air_Quality_df = pd.DataFrame(Air_Quality)


# In[ ]:


group_of_States_df


# In[ ]:


Air_Quality_df


# **HISTOGRAM PLOT FOR AVERAGE OF AIR QUALITY **

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


plt.hist(Air_Quality_df.Avg, histtype='bar', rwidth=0.7)
plt.xlabel('AVERAGE')
plt.ylabel('COUNT')
plt.title('AVERAGE OF AIR QUALITY IN INDIA')
plt.show()


# **HISTOGRAM PLOT FOR CHECKING AIR QUALITY IN STATES**

# In[ ]:


plt.figure(figsize=(17,7), dpi=100)
sns.countplot(x='State',data=Air_Quality)
plt.xlabel('State')
plt.tight_layout()


# **HISTOGRAM OF MAXIMUM OF AIR QUALITY**

# In[ ]:


plt.hist(Air_Quality_df.Max, histtype ='bar', rwidth=0.7)
plt.xlabel("MAX")
plt.ylabel("COUNT")
plt.title("MAX OF AIR QUALITY OF INDIA")
plt.show()


# **HISTOGRAM OF MINIMUM OF AIR QUALITY**

# In[ ]:


plt.hist(Air_Quality_df.Min, histtype ='bar', rwidth=0.7)
plt.xlabel("Min")
plt.ylabel("COUNT")
plt.title("Min OF AIR QUALITY OF INDIA")
plt.show()


# **PLOTTING OF TYPES OF POLLUTANTS**

# In[ ]:


Air_Quality['Pollutants'].value_counts().plot()
plt.xlabel("Pollutants")
plt.ylabel("COUNT")
plt.title("Pollutants OF AIR QUALITY OF INDIA")
print(plt.show())
Air_Quality['Pollutants'].value_counts().plot('bar')
plt.xlabel("Pollutants")
plt.ylabel("COUNT")
plt.title("Pollutants OF AIR QUALITY OF INDIA")
print(plt.show())


# **PLOTTING OF COUNT EACH POLLUTANT IN  STATE**

# In[ ]:


import seaborn as sns
pollutant = list(Air_Quality['Pollutants'].unique())
for poll in pollutant:
    plt.figure(figsize=(18,8), dpi = 100)
    sns.countplot(Air_Quality[Air_Quality['Pollutants'] == poll]['State'], data = Air_Quality)
    plt.tight_layout()
    plt.title(poll)

