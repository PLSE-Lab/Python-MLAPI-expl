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
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_patient=pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')


# In[ ]:


df_patient.head()


# In[ ]:


df_patient.columns


# In[ ]:


df_patient['disease'].value_counts()


# In[ ]:


df_patient['sex'].isnull().sum()


# In[ ]:


df_patient['sex'].value_counts()


# In[ ]:


df_patient["sex"].value_counts().plot.pie(explode=[0.01,0.01],autopct='%.1f%%')


# In[ ]:


df_patient['country'].value_counts()


# In[ ]:


sns.countplot("country",data=df_patient)


# In[ ]:


df_patient=df_patient.drop(['patient_id'],axis=1)


# In[ ]:


df_patient.columns


# In[ ]:


df_patient["infection_case"].value_counts()


# In[ ]:


df_patient["infection_case"].value_counts().plot.pie(autopct='%.1f%%')


# In[ ]:


sns.countplot("infection_case",data=df_patient)


# In[ ]:


df_patient['infection_order'].value_counts()


# In[ ]:


df_patient.columns


# In[ ]:


df_patient['birth_year']


# In[ ]:


df_patient['age']


# In[ ]:


df_patient["age"].value_counts().plot.pie(autopct='%.1f%%')


# In[ ]:


df_patient.columns #I'm just lazy to scroll up.


# In[ ]:


df_patient=df_patient.drop(['global_num'],axis=1)


# In[ ]:


#df_patient=df_patient.drop(['birth_year'],axis=1)
df_patient=df_patient.drop(['contact_number'],axis=1)


# In[ ]:


df_patient['symptom_onset_date'].isnull().sum()


# In[ ]:


df_patient.info()


# We just have 176 values for sympto_onset_date. Hence dropping it off

# In[ ]:


df_patient=df_patient.drop(['symptom_onset_date'],axis=1)


# In[ ]:


df_patient.columns


# In[ ]:


df_patient['state'].value_counts

