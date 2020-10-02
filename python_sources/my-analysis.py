#!/usr/bin/env python
# coding: utf-8

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


data.describe(include = "all")


# In[ ]:


data = pd.read_csv("../input/startup_funding.csv")
#date['Date'] = pd.to_datetime(data["Date"])
def temp(v):
    try:
        return pd.to_datetime(v.replace (".","/").replace("//","/"))
    except:
        print(v)

data["Date"]= data['Date'].apply(lambda v:temp(v))
data['month_year']= data["Date"].dt.strftime("%Y-%m")
data['amount'] = data["AmountInUSD"].str.replace(",","").astype(float)
print(data[["Date","month_year","amount"]].head())


# In[ ]:





# In[ ]:


data.info()


# In[ ]:


#data["month_year"].groupby.plot.bar(figsize = (18,6))
data.groupby(["month_year"]).size().plot.bar(figsize=(18,6))


# In[ ]:


data.groupby(["CityLocation"]).size().plot.bar(figsize=(18,6))


# In[ ]:


data.groupby(["IndustryVertical"]).size().plot.bar(figsize=(18,6))


# In[ ]:


x = data["IndustryVertical"].value_counts()/data.shape[0]*100
x.head(15).plot.bar(figsize=(18,6))


# In[ ]:


x = data["CityLocation"].value_counts()/data.shape[0]*100
x.head(15).plot.bar(figsize=(18,6))


# In[ ]:


x = data["InvestorsName"].value_counts()/data.shape[0]*100
x.head(15).plot.bar(figsize=(18,6))


# In[ ]:


x = data["SubVertical"].value_counts()/data.shape[0]*100
x.head(15).plot.bar(figsize=(18,6))


# In[ ]:



x = data["InvestmentType"].value_counts()/data.shape[0]*100
x.head(15).plot.bar(figsize=(18,6))


# In[ ]:


data.groupby(["IndustryVertical"])['amount'].mean().sort_values(ascending = False)
.head(15).plot.bar(figsize= (18,6))


# In[ ]:


data.groupby(["CityLocation"])['amount'].min().sort_values(ascending = False).head(15).plot.bar(figsize= (18,6))

