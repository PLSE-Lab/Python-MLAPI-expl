#!/usr/bin/env python
# coding: utf-8

# # About the competition
# This competition is arranged by Open Source Imaging Consortium (OSIC) - a non-profit organization.
# 
# Pulmonary Fibrosis is an incurable lung disease. It occurs when lung tissue becomes damaged and scarred. This affects proper functioning of lungs and infact breathing.
# 
# Expectation from the competiton is to predict patient's severity of decline in the lung function based on data provided - CT scan of patient's lungs & allied details like gender, smoking status, FVC. We need to determine lung function based on the output from spirometer, which measures volume of air inhaled and exhaled. The challenge is to use machine learning techniques to make prediction.
# 
# If the prediction outcome is successful, it will benefit patients and their families to better understand any decline in lung function in advance and try for better cure or improved health condition.

# # 1. Importing the packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly as plty
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.io as pio
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '../input/osic-pulmonary-fibrosis-progression/'


# In[ ]:


df_train = pd.read_csv(f'{path}train.csv')
df_test = pd.read_csv(f'{path}test.csv')


# # 2. Training Data

# 2.1 Metadata Information

# In[ ]:


df_train.info()


# In[ ]:


df_train.describe(include='all').T


# In[ ]:


df_train.head()


# In[ ]:


df_tmp = df_train.groupby(['Patient', 'Sex'])['SmokingStatus'].unique().reset_index()


# In[ ]:


df_tmp


# In[ ]:


df_tmp['SmokingStatus'] = df_tmp['SmokingStatus'].str[0]
df_tmp['Sex'] = df_tmp['Sex'].str[0]


# In[ ]:


df_tmp['SmokingStatus'].value_counts()


# In[ ]:


df_tmp['Sex'].value_counts()


# In[ ]:


fig, ax = plt.subplots(1,2, figsize = (20,6), sharex=True)
sns.countplot(x='SmokingStatus',data=df_tmp,ax=ax[0])
sns.countplot(x='SmokingStatus',hue='Sex', data=df_tmp,ax=ax[1])
ax[0].title.set_text('Smoking Status')
ax[1].title.set_text('Smoking Status Vs Sex')
plt.show()


# # What do we have in training dataset (metadata info excluding CT Scan)
# * we have 1549 data with no missing values.
# * 176 unique patient data is made available  along with data related to their age, gender, smoking status, FVC, weeks
# * Age of patients is between 49 and 88. Average age of the patient within the dataset is 67
# * We have 139 Male and 37 female patients
# * We have 118 Ex-Smoker, 49 Never-Smoked and 9 people who are smoking currently (active)

# In[ ]:





# In[ ]:





# # 3. Test Data

# In[ ]:


df_test.info()


# In[ ]:


df_test


# We just have 5 patient data available in test set

# # Exploration of Data will continue

# In[ ]:




