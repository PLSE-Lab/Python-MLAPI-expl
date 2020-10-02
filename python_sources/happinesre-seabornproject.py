#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2017.csv") # read Dataset (2017 happiness report)
data.info() # look info


# In[ ]:


data.columns # Data.columns is clean


# In[ ]:


data.head(10) # first 10 data


# In[ ]:


data.tail(10) # ending 10 data. There is no Turkey in it.


# In[ ]:


data.corr()  # correlation analy.


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18)) 
sns.heatmap(data.corr(),annot=True) 
plt.show()


# In[ ]:


t = 0
mist = 0

for i in data["Trust..Government.Corruption."]:
    if i >= 0.20:
        t +=1
    else:
        mist +=1
        
list_trust_government = {"Trust Goverment":t, "Mistrust Goverment":mist}
print(list_trust_government)
        
        


# In[ ]:


data.describe()


# In[ ]:


data.plot(kind="hist", subplots=True, grid=True, alpha=0.5,figsize=(18,18))
plt.show()


# In[ ]:


data[(data["Happiness.Score"]>5) & (data["Economy..GDP.per.Capita."]>1) & (data["Freedom"]>0.50)]


# In[ ]:


# Line Plot
data["Health..Life.Expectancy."].plot(kind = "line", color="b", label = "HealthLifeExp", alpha = 0.5, grid =True, linestyle=":", figsize=(18,18))
data["Economy..GDP.per.Capita."].plot(kind = "line", color="g", label="EcoGDPperCap", alpha=0.5, grid=True, linestyle="-.", figsize=(18,18))
plt.legend("HealtLife-EcoGDP")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Line Plot")
plt.show()



# In[ ]:


# scatter
data.plot(kind="scatter", x="Freedom", y="Trust..Government.Corruption.", alpha=0.5, color="red", figsize=(13,13))
plt.xlabel("Freedom")
plt.ylabel("Trust..Government.Corruption.")
plt.title("Freedom-TrustGoverment")


# In[ ]:


# Hist
data["Happiness.Score"].plot(kind="hist", bins =50, figsize=(12,12))
plt.show()

