#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT THE RELEVANT LIBRARIES


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# LOAD THE DATA


# In[ ]:


raw_data = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')
raw_data


# In[ ]:


raw_data.columns


# In[ ]:


# DECLARE THE DEPENDENT AND INDEPENDENT VARIABLES 


# In[ ]:


x1 = raw_data[['satisfaction_level', 'average_montly_hours','promotion_last_5years','salary']]
y = raw_data['left']


# In[ ]:


# FIND MISSING VALUES AND EXPLORE THE DATA TOKNOW THE RELEVANT FEATURES


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


pd.crosstab( raw_data.salary, raw_data.left ).plot( kind='bar' )


# In[ ]:


pd.crosstab( raw_data.Department, raw_data.left ).plot( kind='bar' )


# In[ ]:


pd.crosstab( raw_data.promotion_last_5years, raw_data.left ).plot( kind='bar' )


# In[ ]:


pd.crosstab( raw_data.average_montly_hours, raw_data.left ).plot( kind='bar' )


# In[ ]:


# CONVERT SALARY FROM TEXT TO NUMBERS WITH DUMMIES


# In[ ]:


x_salary_dummies = pd.get_dummies(x1['salary'])


# In[ ]:


x_with_dummies = pd.concat([x1,x_salary_dummies], axis =1)


# In[ ]:


x = x_with_dummies.drop('salary', axis=1)
y = raw_data['left']
x


# In[ ]:


# SPLIT INTO TRAIN AND TEST DATA 


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)


# In[ ]:


# CREATE THE MODEL


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


reg_log=LogisticRegression()
reg_log.fit(x,y)


# In[ ]:


# CHECK THE ACCURACY  OF THE MODEL


# In[ ]:


reg_log.score(x,y)

