#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')
data.head()


# #Now the first marking will be: What is/are the main factor/factors behind the employee retention / holding on the jobs?
# #Here the column value left ==1 means employee leaving the job and left==0 means employee holding the job

# In[ ]:


retention=data[data['left']==1]
retention.shape


# In[ ]:


retention.head()


# In[ ]:


holding=data[data['left']==0]
holding.shape


# In[ ]:


holding.head()


# In[ ]:


data.groupby('left').mean()


# **Reasons behind the employee retentions are:**
# 1.Satisfaction Level
# 2.Average Monthly Hours
# 3.Promotion Last 5 Years

# **Relation between Satisfaction level and Department**

# In[ ]:


pd.crosstab(data.satisfaction_level,data.Department).plot(kind='bar',figsize=(22, 16))


# In[ ]:


max_satisfaction=data['satisfaction_level'].max()
max_satisfaction


# In[ ]:


a=data.loc[data['satisfaction_level']==1.0,'Department'].unique()
a


# In[ ]:


data.Department.unique()


# In[ ]:


data.Department.nunique()


# In[ ]:


min_satisfaction=data['satisfaction_level'].min()


# In[ ]:


min_satisfaction


# In[ ]:


a=data.loc[data['satisfaction_level']==0.09,'Department'].unique()
a


# In[ ]:


a=data.loc[data['satisfaction_level']==0.09,'Department'].nunique()
a


# **Relation Between Highest Satisfaction level and Salary**

# In[ ]:


pd.crosstab(data.salary,data['satisfaction_level']==1.0).plot(kind='bar')


# **Relation between Job left and satisfaction **

# In[ ]:


pd.crosstab(data.salary,data.left).plot(kind='bar')


# In[ ]:


pd.crosstab(data.left,data.salary).plot(kind='bar')


# **From the above two charts we can define that : Employees having high salary are likely not to leave the company **

# In[ ]:


pd.crosstab(data.left,data.Department).plot(kind='bar', figsize=(22,16))


# In[ ]:


pd.crosstab(data.Department,data.left).plot(kind='bar', figsize=(22,16))


# In[ ]:


data.Department.value_counts()


# ***So far from the Data analysis we can easily conclude that the followings will be the dependent variables:***
# 1. Satisfaction_Level
# 2. Average Monthly Hours
# 3. Promotion Last 5 years
# 4. Salary 

# In[ ]:


new_data=data[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
new_data.head()


# In[ ]:


#Now here i will apply the LOGISTICS REGRESSION analysis where the employees will leave or hold the job


# In[ ]:


dummy=pd.get_dummies(data.salary)
dummy.head()


# In[ ]:


new_data_dummy=pd.concat([new_data,dummy],axis='columns')
new_data_dummy.head()


# In[ ]:


new_data_dummy.drop('salary',axis='columns',inplace=True)
new_data_dummy.head()


# In[ ]:


new_data_dummy.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x=new_data_dummy
y=data.left


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:


model.predict(x_test)


# In[ ]:


model.score(x_test,y_test)


# In[ ]:




