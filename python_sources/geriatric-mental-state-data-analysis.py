#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns 
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/Geriatrics.csv')
#shape of the dateset (column,rows)
df.shape


# In[ ]:


#description of dataset
# DM - diabetes , HTN - hypertension,
df.describe()


# In[ ]:


#first 5 items
#0 = female 1 = male
df.head(60)


# In[ ]:


#
def plot(group,column,plot):
    ax=plt.figure(figsize=(7,7))
    df.groupby(group)[column].sum().sort_values().plot(plot)
    
plot('Health','Gender','barh')


# In[ ]:


#Total patients count with normal/disease
explode = (0.2,0)  
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.pie(df['Health'].value_counts(), explode=explode,labels=['Normal','Disease'], autopct='%1.0f%%',
         startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()


# In[ ]:


#Total patients count with normal/disease
explode = (0.2,0)  
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['0','1'], autopct='%1.0f%%',
         startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Gender'],hue=df['Age'])


# In[ ]:



fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Personal_income'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Education'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Family_Type'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Occupation'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['substance_abuse'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Personal_income'],hue=df['Age'])


# In[ ]:


#duabetes
fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['DM'],hue=df['Age'])


# In[ ]:


#hypertension
fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['HTN'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['hearing_problem'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['vision_problem'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['mobility_problem'],hue=df['Age'])


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['sleep_problem'],hue=df['Age'])


# **OBSERVATIONS FROM THE DATA by Sanjay BN and Rakshith DC**
# 1. 57% of the Geriatrics are healthy.
# 2. 53% of the Geriatrics are male and 47% are female.
# 3. 6 males are of same aged and 5 females are of same aged in the geriatrics list
# 4. Males have personal income more when compared to female geriatrics
# 5. Education mark 4 and 6 has less patients while education marked 1 and 2 has most number of patients.
# 6. Family Type 0 has more number of patients.
# 7. More number of patients dont have any occupation.
# 8. More number of patients dont do substance abuse.
# 9. More number of patients dont have personal income
# 10. More number of patients have diabetes.
# 11. More number of patients dont have hypertension
# 12. More number of patients have hearing problem.
# 13. Vision problem seems to be equally present and not present for the patients.
# 14. More number of patients are mobile and can move freely.
# 15. More number of patients have sleep problems.
