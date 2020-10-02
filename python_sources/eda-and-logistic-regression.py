#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression for fake job identifying

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


import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('notebook')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[ ]:


df = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')


# In[ ]:


df.head()


# ## EDA

# *Output of a number of visualisations to explore possible relations between the variables*

# In[ ]:


df.info()


# In[ ]:


df.describe()


# *Visualise the completness of the dataset.*

# In[ ]:


plt.figure(figsize=(24, 6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# *As  we can see there are a number of columns that have a high number missing values. Such as the salary range  and department.   hunch lets see if the there any relationship between the employment tpye and the ad being fraudulent*

# In[ ]:


df['employment_type'].value_counts()


# In[ ]:


sns.countplot(df['employment_type'], hue = df['fraudulent'])


# *As there are only five catergoris it might be worth obtaining dummy variables for these.
# Investigate the make up of the salary range*

# In[ ]:


df['salary_range'].head(15)


# *As we can see there is a upper band and an lower band. This means we could split the field to obtain two new variables.*

# In[ ]:


sns.countplot(df['has_company_logo'], hue = df['fraudulent'])


# In[ ]:


sns.countplot(df['telecommuting'], hue = df['fraudulent'])


# ## Creation of extra variables

# *First the creation of dummy variables for employment type*

# In[ ]:


type_job = pd.get_dummies(df['employment_type'], drop_first = True)


# > *Split of Salery Range and creation of new variables to include in the model*

# In[ ]:


df['salary_min'] = df['salary_range'][df['salary_range'].notnull()].apply(lambda x :x.split('-')[0])
df['salary_max'] = df['salary_range'][df['salary_range'].notnull()].apply(lambda x :x.split('-')[-1])
df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna("0")
df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna("0")


# *Combine the new type dummy variables and the dataframe*

# In[ ]:


df = pd.concat([df, type_job], axis = 1)


# ## Application of logistic regression

# In[ ]:


X= df[['telecommuting', 'has_company_logo', 'has_questions', 'Full-time', 'Other',
       'Part-time', 'Temporary', 'salary_min', 'salary_max']]
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(accuracy_score(y_test,predictions))


# *As we can see there is thus far no predictive power in the model. I believe it is due to the fact that the number of positives is very low and it skews training of the model to predict only negtives.*

# ## Next Steps

# *It might be worth to investigate wether some NLP on the title column will yield improvement in the model.*
