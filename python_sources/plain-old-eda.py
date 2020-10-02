#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[ ]:


data = pd.read_csv('/kaggle/input/Andhra_Health_Data.csv',engine='python')


# In[ ]:


data['SEX'] = data['SEX'].map({'Male':'M','Female':'F','Male(Child)':'M','Female(Child)':'F','FEMALE':'F','MALE':'M'})
data['SEX'].value_counts()


# In[ ]:


for cols in data.columns:
    print(cols,len(data[cols].unique()))
    print('-------------------------------------------------')


# In[ ]:


sns.distplot(data['AGE'])


# In[ ]:


sns.distplot(data['PREAUTH_AMT'])


# In[ ]:


sns.distplot(data['CLAIM_AMOUNT'])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
data['Mortality Y / N'].value_counts(normalize=True).plot.bar(title= 'Mortality')


# In[ ]:


data['Mortality Y / N'].value_counts()


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
data['CATEGORY_NAME'].value_counts(normalize=True).plot.bar(title='Surgery Category')


# In[ ]:


surg_cat = pd.crosstab(data['CATEGORY_NAME'],data['SEX'])
surgcat = surg_cat.sort_values(by=['M'],ascending=False)[:10]
surgcat.div(surgcat.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,10),color = ['b','y'])


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
data['SURGERY'].value_counts(normalize=True)[:20].plot.bar(title= 'Surgery')


# In[ ]:


surgs = pd.crosstab(data['SURGERY'],data['SEX'])
sort_surg = surgs.sort_values(by=['M'],ascending=False)[:10]
sort_surg.div(sort_surg.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,10),color = ['b','y'])


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
fatalities = pd.crosstab(data['SURGERY'],data['Mortality Y / N'])
fatalities.sort_values(by=['YES'],ascending=False)[:10]['YES'].plot.bar(title= 'Surgeries with Most Fatalities')


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
norm_fatalities = fatalities
norm_fatalities['YES_scaled'] = norm_fatalities['YES']/(norm_fatalities['YES']+norm_fatalities['NO'])*100
norm_fatalities.sort_values(by=['YES_scaled'],ascending=False)[:10]['YES_scaled'].plot.bar(title= 'Most fatal surgeries')


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
data['HOSP_DISTRICT'].value_counts(normalize=True).plot.bar(title= 'District')


# In[ ]:


dist_sex = pd.crosstab(data['HOSP_DISTRICT'],data['SEX'])
dist = dist_sex.sort_values(by=['M'],ascending=False)[:10]
dist.div(dist.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,10),color = ['b','y'])


# In[ ]:


fig = plt.gcf()
fig.set_size_inches(15,10)
data['HOSP_NAME'].value_counts(normalize=True)[:20].plot.bar(title= 'Hospitals')


# In[ ]:


surg_caste = pd.crosstab(data['CATEGORY_NAME'],data['CASTE_NAME'])
surg_caste['total'] = surg_caste.sum(axis=1)
surgcaste = surg_caste.sort_values(by=['total'],ascending=False)[:10]
surgcaste.drop(columns=['total'],inplace=True)
surgcaste.div(surgcaste.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,10),color = ['b','y','r','g','c','m'])


# In[ ]:




