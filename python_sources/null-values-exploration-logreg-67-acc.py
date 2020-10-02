#!/usr/bin/env python
# coding: utf-8

# # Let's find some null values?

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')


# In[ ]:


df.columns = [x.lower().strip().replace(' ','_') for x in df.columns]
df.columns


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.isna(),cbar=False)


# In[ ]:


# Proportion of the dataset that has null values

def find_null_values(threshold_of_null=0):
    
    null_columns = np.where(df.isna().sum() > threshold_of_null)[0]
    prop = (len(null_columns) / df.shape[1])*100    

    print(f'{round(prop,2)}% ({len(null_columns)}) of columns in the dataset have at least {threshold_of_null} null values.')


# In[ ]:


find_null_values(df.shape[0]*0.8)


# ## Well! There are 88 columns (of 111) with at least 4515 null values. That's going to be hard.

# In[ ]:


def find_null_values_rows(threshold_of_null=0):
    
    null_rows = np.where(df.isna().sum(axis=1) > threshold_of_null)[0]
    prop = (len(null_rows) / df.shape[0])*100
    
    print(f'{round(prop,2)}% ({len(null_rows)}) of rows in the dataset have at least {threshold_of_null} null values.')


# In[ ]:


find_null_values_rows(df.shape[1]*0.8)


# ## Well! There are 4094 rows (of 5644) with at least 88 null values (out of 111). That's going to be really, really hard.

# In[ ]:


df.groupby('sars-cov-2_exam_result').count().iloc[:,list(set(np.where(df.groupby('sars-cov-2_exam_result').count() > 500)[1]))]


# In[ ]:


## Let's find where the we have at least 500 not null values.

df.groupby('sars-cov-2_exam_result').count().iloc[:,list(set(np.where(df.groupby('sars-cov-2_exam_result').count() > 500)[1]))].columns


# In[ ]:


## Let's use this columns

df.groupby('sars-cov-2_exam_result').count().loc[:,['hematocrit','hemoglobin','platelets','bordetella_pertussis','influenza_b,_rapid_test','influenza_a,_rapid_test','red_blood_cells','basophils','lymphocytes','leukocytes']]


# In[ ]:


df_model = df[['sars-cov-2_exam_result','hematocrit','hemoglobin','platelets','bordetella_pertussis','influenza_b,_rapid_test','influenza_a,_rapid_test','red_blood_cells','basophils','lymphocytes','leukocytes']]
df_model.notnull().sum()


# In[ ]:


df = df_model.dropna(how='any')


# In[ ]:


## Checking correlations

df_corr = df.assign(target = lambda x: x['sars-cov-2_exam_result'].map({'negative':0,'positive':1}))

plt.figure(figsize=(15,8))
sns.heatmap(df_corr.corr(method='spearman'),vmin= -1, vmax= 1,annot=True)


# I'll try to use this dataset ...

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score

df.head()


# In[ ]:


cat_features = ['bordetella_pertussis','influenza_b,_rapid_test','influenza_a,_rapid_test']
num_features = ['hematocrit','hemoglobin','platelets','red_blood_cells','basophils','lymphocytes','leukocytes']

mkcol = make_column_transformer((StandardScaler(),num_features),(OneHotEncoder(),cat_features),remainder='drop')
x = mkcol.fit_transform(df)
y = df.loc[:,'sars-cov-2_exam_result']


# In[ ]:


logreg = LogisticRegression(class_weight='balanced')
logreg.fit(x,y)


# In[ ]:


cross_val_score(logreg,x,y,cv=9).mean()


# In[ ]:




