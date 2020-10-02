#!/usr/bin/env python
# coding: utf-8

# # Impoting modules and load data files

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


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import seaborn as sns
import random
import math
import time
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# importing datasets
data_individual = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
                         #parse_dates=['Date'])
data_individual.shape


# In[ ]:


data_individual.head(2)


# In[ ]:


data_individual.columns


# In[ ]:


data_country=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


data_country.head(2)


# # Let's check the number of confirmed, recovered and dead cases by country

# In[ ]:


df_grp=data_country[["Country/Region", "Confirmed", "Deaths","Recovered"]].groupby(["Country/Region"]).sum().sort_values("Confirmed")
df_grp[(df_grp["Confirmed"]>10000) & (df_grp["Confirmed"]<500000)].plot(kind='bar', stacked=True)


# ## What about the recovery and death rates?

# In[ ]:


df_grp['Recovery_%']=(df_grp.Recovered/df_grp.Confirmed)*100
df_grp['Death_%']=(df_grp.Deaths/df_grp.Confirmed)*100
ax= plt.Figure(figsize=(10,7))
ax=df_grp[(df_grp["Confirmed"]>5000)][["Recovery_%","Death_%"]].plot(kind='bar', stacked=True)


# * Recovery rate is highest in Singapore apart from China
# * Death rate is high in Italy Spain
# * Death rate is low in Singapore and Malaysia

# In[ ]:





# ### What abou recovery to death Ratio?
# It seems Singapore is winning there and Israel is doing pretty well

# In[ ]:


ax= plt.Figure(figsize=(10,7))
df_grp['Recover_to_Death_ratio']=(df_grp.Recovered/df_grp.Deaths)
ax=df_grp[(df_grp["Confirmed"]>5000)][["Recover_to_Death_ratio"]].plot(kind='bar', stacked=True)


# In[ ]:


df_grp[df_grp.index=='Singapore']


# In[ ]:


df_grp[df_grp.index=='Mainland China']


# Singapore's Recovery to death ratio 709.5 where that of China is 14.75

# In[ ]:


df_high_success=data_individual[(data_individual["country"]=='Singapore') | (data_individual["country"]=='Mainland China') | (data_individual["country"]=='Israel')]


# ### Recovery Rates are highest in Singapore, and Israel apart from Mainland China

# # Find Common Symptoms Among Confirmed Cases
# 
# Here we take symptoms only if it appeared atleast in 5 people

# In[ ]:


def find_common_symptoms(df, syms=[], min_case=5):
    symptoms_uniq = {} 
    for sym in syms:
#         print(sym)
        symptoms = list(df[sym].dropna().str.split(",").values)
        for i in symptoms:
            for j in i:
                if j.lstrip(" ") not in symptoms_uniq:
#                   print(i,j)
                    symptoms_uniq[j.lstrip(" ")]=1
                else:
                    symptoms_uniq[j.lstrip(" ")]=symptoms_uniq[j.lstrip(" ")]+1
    syms_comm_confirmed={k: v for k, v in sorted(symptoms_uniq.items(), key=lambda item: item[1]) if v>min_case}
    return syms_comm_confirmed


# In[ ]:


syms_all=find_common_symptoms(data_individual, ['symptom'])
syms_all


# In[ ]:


plt.figure(figsize=(10,3))
plt.bar(x=syms_all.keys(), height=syms_all.values())
plt.xticks(rotation=90, fontsize=10)


# # Find symptoms common in people who died
# 

# In[ ]:


dead_data=data_individual[data_individual['death']!="0"]
syms_dead=find_common_symptoms(dead_data, ['symptom'], min_case=0)
print("Shape of data about deaths: ",dead_data.shape)


# # Only thses are significant symptoms in people who died
# Most commons are Fever,pneumonia and Cough

# In[ ]:


plt.figure(figsize=(10, 3))
plt.bar(x=syms_dead.keys(), height=syms_dead.values())
plt.xticks(rotation=45, fontsize=10)


# # Check the Effects of Age

# ### Minimum age among dead people is 36 and Mean 68

# In[ ]:


dead_data.age.describe()


# ### But does it change with Gender?

# In[ ]:


plt.figure(figsize=(6, 3))
plt.title('Gender')
data_individual.gender.value_counts().plot.bar();


# In[ ]:


pd.pivot_table(data=dead_data, index='gender', values=['age'], aggfunc=['mean','median'])


# In[ ]:


pd.pivot_table(data=data_individual, index='gender', values=['age'], aggfunc=['mean','median'])


# In[ ]:


f, ax = plt.subplots(1)
sns.kdeplot(dead_data[dead_data['gender']=='female'].age, label='Female Dead')
sns.kdeplot(dead_data[dead_data['gender']=='male'].age, label='Male Dead')

sns.kdeplot(data_individual[data_individual['gender']=='female'].age, label='Female Confirmed')
sns.kdeplot(data_individual[data_individual['gender']=='male'].age, label='Male Confirmed')


# ### Male population death starts with a little early age than their female counterparts

# In[ ]:


fig, ax = plt.subplots()
ax=sns.barplot(x='gender', y='age', data=dead_data)

