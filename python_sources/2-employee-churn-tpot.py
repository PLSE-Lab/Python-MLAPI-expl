#!/usr/bin/env python
# coding: utf-8

# # Part 2 : What would be the the best traditional modeling approach ?
# 
# After reading this article(https://towardsdatascience.com/building-an-employee-churn-model-in-python-to-develop-a-strategic-retention-plan-57d5bd882c2d), i was using the same dataset to verify potential added values using more advanced approach. 
# 
# ## Overview:
# Dataset contains both continuous and categorical featues, this is supported by TPOT.
# I use simple autoencoder to replace all categorical feature by new related columns.
# Then, i simply follow regular canevas described TPOT documentation.
# 
# ## Related work:
# Part 1 : Can we identify most important features using LOFO ?
# 
# Part 2 : What would be the the best traditional modeling approach ?
# 
# Part 3 : Can we predict employee churn uing deep learning ?

# In[ ]:


import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['OMP_NUM_THREADS'] = '4'
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


dtypes={
    'Age':                         'int64',
    'Attrition':                   'category',
    'BusinessTravel':              'category',
    'DailyRate':                   'int64',
    'Department':                  'category',
    'DistanceFromHome':            'int64',
    'Education':                   'int64',
    'EducationField':              'category',
    'EmployeeCount':               'int64',
    'EmployeeNumber':              'int64',
    'EnvironmentSatisfaction':     'int64',
    'Gender':                      'category',
    'HourlyRate':                  'int64',
    'JobInvolvement':              'int64',
    'JobLevel':                    'int64',
    'JobRole':                     'category',
    'JobSatisfaction':             'int64',
    'MaritalStatus':               'category',
    'MonthlyIncome':               'int64',
    'MonthlyRate':                 'int64',
    'NumCompaniesWorked':          'int64',
    'Over18':                      'category',
    'OverTime':                    'category',
    'PercentSalaryHike':           'int64',
    'PerformanceRating':           'int64',
    'RelationshipSatisfaction':    'int64',
    'StandardHours':               'int64',
    'StockOptionLevel':            'int64',
    'TotalWorkingYears':           'int64',
    'TrainingTimesLastYear':       'int64',
    'WorkLifeBalance':             'int64',
    'YearsAtCompany':              'int64',
    'YearsInCurrentRole':          'int64',
    'YearsSinceLastPromotion':     'int64',
    'YearsWithCurrManager':        'int64',}


# In[ ]:


# source : https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/
df = pd.read_excel('../input/WA_Fn-UseC_-HR-Employee-Attrition.xlsx', sheet_name=0,dtype=dtypes)


# In[ ]:


cont=[]
cat=[]
for key, value in dtypes.items():
    if key!='Attrition':
        if value == "int64":
            cont.append(key)
        else:
            cat.append(key)


# In[ ]:


df.head(5)


# In[ ]:


df = pd.get_dummies(df, columns=cat)


# In[ ]:


df['Attrition']=df.Attrition.eq('Yes').mul(1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[cont], df['Attrition'], test_size=0.2, random_state=42)
train = pd.concat([X_train, y_train], 1)
test = pd.concat([X_test, y_test], 1)


# In[ ]:


pipeline_optimizer = TPOTClassifier()


# In[ ]:


pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5, n_jobs=-1,random_state=42, verbosity=2, early_stop=5)


# In[ ]:


pipeline_optimizer.fit(X_train, y_train)


# In[ ]:


print(pipeline_optimizer.score(X_test, y_test))


# In[ ]:




