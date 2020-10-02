#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


outbreak = pd.read_csv('outbreaks_kaggle.csv')


# In[ ]:


outbreak.info()


# In[ ]:


#sns.factorplot('Month',data=outbreak,kind='count')


# In[ ]:


outbreak.head()


# In[ ]:


#sns.factorplot('Year',data=outbreak,kind='count')


# In[ ]:


#sns.factorplot(x='Year',y='Fatalities',data=outbreak,kind='point')


# In[ ]:


out_year = outbreak.groupby(outbreak['Year'])

out_year.describe().T


# In[ ]:


ill_year = outbreak['Fatalities'].groupby(outbreak['Year'])

ill_year.sum()


# In[ ]:


#number of patients each year

illness_by_year = outbreak['Illnesses'].groupby(outbreak['Year'])

illness_by_year.sum().plot(kind='bar')


# In[ ]:





# In[ ]:


#count of death each year

death_by_year = outbreak['Fatalities'].groupby(outbreak['Year'])

death_by_year.sum().plot(kind='bar')


# In[ ]:





# In[ ]:


#count of patients adminning to hospital

hospitalize_by_year = outbreak['Hospitalizations'].groupby(outbreak['Year'])

hospitalize_by_year.sum().plot(kind='bar')


# # 1998

# In[ ]:


rep_1998 = outbreak[outbreak['Year']==1998]


# In[ ]:


rep_1998.head()


# In[ ]:





# In[ ]:


#number of species causing morbidity

plt.figure(figsize=(10,6))
rep_1998['Illnesses'].groupby(rep_1998['Species']).sum().plot(kind='bar')


# In[ ]:


#Number of patients in each state

plt.figure(figsize=(10,6))
rep_1998['Illnesses'].groupby(rep_1998['State']).sum().plot(kind='bar')


# In[ ]:


#Number of patients by month

plt.figure(figsize=(10,6))
rep_1998['Illnesses'].groupby(rep_1998['Month']).sum().plot(kind='bar')


# In[ ]:


#Number of death case by species

plt.figure(figsize=(10,6))
rep_1998['Fatalities'].groupby(rep_1998['Species']).sum().plot(kind='bar')


# In[ ]:





# In[ ]:


#rep_1998_cm[rep_1998_cm['Species']=='Amnesic shellfish poison']


# # 2004

# In[ ]:


rep_2004 = outbreak[outbreak['Year']==2004]


# In[ ]:


rep_2004.head()


# In[ ]:


#number of species causing morbidity

plt.figure(figsize=(10,6))
rep_2004['Illnesses'].groupby(rep_2004['Species']).sum().plot(kind='bar')


# In[ ]:


#Number of patients in each state

plt.figure(figsize=(10,6))
rep_2004['Illnesses'].groupby(rep_2004['State']).sum().plot(kind='bar')


# In[ ]:


#Number of patients by month

plt.figure(figsize=(10,6))
rep_2004['Illnesses'].groupby(rep_2004['Month']).sum().plot(kind='bar')


# In[ ]:


#Number of death case by species

plt.figure(figsize=(10,6))
rep_2004['Fatalities'].groupby(rep_2004['Species']).sum().plot(kind='bar')


# #  2011

# In[ ]:


rep_2011 = outbreak[outbreak['Year']==2011]

rep_2011.head()


# In[ ]:


#number of species causing morbidity

plt.figure(figsize=(10,6))
rep_2011['Illnesses'].groupby(rep_2011['Species']).sum().plot(kind='bar')


# In[ ]:


#Number of patients in each state

plt.figure(figsize=(10,6))
rep_2011['Illnesses'].groupby(rep_2011['State']).sum().plot(kind='bar')


# In[ ]:


#Number of patients by month

plt.figure(figsize=(10,6))
rep_2011['Illnesses'].groupby(rep_2011['Month']).sum().plot(kind='bar')


# In[ ]:


#Number of death case by species

plt.figure(figsize=(10,6))
rep_2011['Fatalities'].groupby(rep_2011['Species']).sum().plot(kind='bar')


# #  2015 the most recent one

# In[ ]:


rep_2015 = outbreak[outbreak['Year']==2015]

rep_2015.head()


# In[ ]:


#number of species causing morbidity

plt.figure(figsize=(10,6))
rep_2015['Illnesses'].groupby(rep_2015['Species']).sum().plot(kind='bar')


# In[ ]:


#Number of patients in each state

plt.figure(figsize=(10,6))
rep_2015['Illnesses'].groupby(rep_2015['State']).sum().plot(kind='bar')


# In[ ]:


#Number of patients by month

plt.figure(figsize=(10,6))
rep_2011['Illnesses'].groupby(rep_2011['Month']).sum().plot(kind='bar')


# In[ ]:


#Number of death case by species

plt.figure(figsize=(10,6))
rep_2011['Fatalities'].groupby(rep_2011['Species']).sum().plot(kind='bar')


# In[ ]:




