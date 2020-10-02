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
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


patient = pd.read_csv('../input/indian_liver_patient.csv')
patient.head()


# In[ ]:


patient.describe()


# In[ ]:


patient.info()


# In[ ]:


# Showing different age groups in dataset
sns.distplot(patient['Age'],bins=40,kde=False)


# In[ ]:


# Showing stats of patients having liver disease
sns.countplot(x='Dataset',data=patient,palette='bwr')


# In[ ]:


sns.heatmap(patient.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


plt.figure(figsize=(8,8))
sns.boxplot(y='Albumin_and_Globulin_Ratio',x='Dataset',data=patient,palette='winter')


# In[ ]:


def impute_Nulls(cols):
    Albumin_Ratio = cols[0]
    Dataset = cols[1]
    
    if pd.isnull(Albumin_Ratio):

        if Dataset == 1:
            return 1

        elif Dataset == 2:
            return 1.2
    else:
        return Albumin_Ratio
patient['Albumin_and_Globulin_Ratio'] = patient[['Albumin_and_Globulin_Ratio','Dataset']].apply(impute_Nulls,axis=1)


# In[ ]:


Gender = pd.get_dummies(patient['Gender'],drop_first=True)
patient.drop('Gender',axis=1,inplace=True)
patient = pd.concat([patient,Gender],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(patient.drop('Dataset',axis=1), 
                                                    patient['Dataset'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:




