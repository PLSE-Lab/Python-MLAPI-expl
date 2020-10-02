#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/simple-decision-tree/salaries.csv')
df.head()


# In[ ]:


input = df.drop('salary_more_then_100k', axis = 'columns')
input.head()


# In[ ]:


target = df['salary_more_then_100k']
target.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[ ]:


input['company_n'] = le_company.fit_transform(input['company'])
input['job_n'] = le_job.fit_transform(input['job'])
input['degree_n'] = le_degree.fit_transform(input['degree'])

input


# In[ ]:


input.drop(['company','job','degree'], axis = 'columns', inplace = True)
input


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[ ]:


model.fit(input, target)


# In[ ]:


model.score(input,target)


# In[ ]:


model.predict([[2,1,0]])


# In[ ]:


model.predict([[2,1,1]])


# In[ ]:




