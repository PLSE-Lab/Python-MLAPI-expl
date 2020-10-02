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
import matplotlib as mlt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")


# In[ ]:


df.head()


# In[ ]:


## Data cleaning and manipulation


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


#The data have 210 rows and 14 columns


# In[ ]:


df.isna().sum()


# In[ ]:


## The data have missing values but except the world bank name column


# In[ ]:


df.tail()


# In[ ]:


df2 = df.drop(["Province_State"], axis = 1)


# In[ ]:


df2

# I have droped the Province_State column to have an accurate analysis has the column had 196 missing data out of 210


# In[ ]:


# check for duplications in the data 


# In[ ]:


df2.duplicated()


# In[ ]:


## The data has no duplication in it


# In[ ]:


df2.shape


# In[ ]:


# After i dropped the column the data shows that the data have 210 rows and 13 columns


# In[ ]:


df2.isnull().any()


# In[ ]:


df2.median()


# In[ ]:


# will be replacing every null value with the meadian value expect for Country_Region  column


# In[ ]:


df2["Country_Region"].fillna('no country_region', inplace = True)


# In[ ]:


df2


# In[ ]:


# For country_relion i have replaced all the null with no counry religion


# In[ ]:


df3 =df2.fillna(df2.median())


# In[ ]:


df3.isna().sum()


# In[ ]:


#They is no missing values in the data 


# **Checking for outliers**

# In[ ]:


for column in df4:
    plt.figure()
    df4.boxplot([column])


# In[ ]:


## we can see that they is outliers in the data


# In[ ]:


# df4 is the dataset with numeric values


# In[ ]:


df4.head()


# In[ ]:





# **DATA VISUALISATION**

# In[ ]:


# will use heatmap

correlations = df3.corr()
sns.heatmap(data=correlations, square=True, cmap="bwr")

plt.yticks(rotation=0)
plt.xticks(rotation =90)


# In[ ]:


## we can see that they is more behavior on health_exp_per_capita_usd 2016 and per_capit_exp_ppp_2016, The red colour shows maximum correlation and 
## the blue shows the minimum correlation


# In[ ]:


### Gonna focus on the health_exp_per_capita_usd 2016 and per_capit_exp_ppp_2016 to further understand there relationship

df3[['Health_exp_per_capita_USD_2016', 'per_capita_exp_PPP_2016']].groupby(['per_capita_exp_PPP_2016']).describe().unstack()


# In[ ]:


## we can see the count and max of both columns grouped by per capita exp ppp 2016


# In[ ]:


df3[['Health_exp_per_capita_USD_2016', 'per_capita_exp_PPP_2016']].corr()


# In[ ]:


# looking for relationship against per capita xep ppp 2016

fig,axs = plt.subplots(1,5, sharey = True)
df3.plot(kind = "scatter", x = "Health_exp_per_capita_USD_2016", y = "per_capita_exp_PPP_2016", ax = axs[0],figsize=(22,10))
df3.plot(kind = "scatter", x = "External_health_exp_pct_2016", y = "per_capita_exp_PPP_2016", ax = axs[1])
df3.plot(kind = "scatter", x = "Health_exp_public_pct_2016", y="per_capita_exp_PPP_2016", ax=axs[2])
df3.plot(kind="scatter", x ="Health_exp_out_of_pocket_pct_2016", y = "per_capita_exp_PPP_2016", ax=axs[3])
df3.plot(kind = 'scatter', x = 'Health_exp_per_capita_USD_2016', y = "per_capita_exp_PPP_2016", ax = axs[4])


# In[ ]:


# the relationship against Specialist_surgical_per_1000_2008

fig,axs = plt.subplots(1,2, sharey = True)
df3.plot(kind = "scatter", x = "Physicians_per_1000_2009-18", y = "Specialist_surgical_per_1000_2008-18", ax = axs[0],figsize=(22,10), color = 'g')
df3.plot(kind = "scatter", x = "Nurse_midwife_per_1000_2009-18", y = "Specialist_surgical_per_1000_2008-18", ax = axs[1],color = 'g')


# In[ ]:


# we can see the relationships of each column against the one column


# In[ ]:




