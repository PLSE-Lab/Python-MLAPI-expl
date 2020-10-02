#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
data=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# Any results you write to the current directory are saved as output.


# **Lets take a look at data.**

# In[ ]:


data.info()


# We don't need serial No. It will not affect your chance of selection so Let's remove it.

# In[ ]:


data=data.drop('Serial No.',axis=1)


# In[ ]:


data.head()


# Now We have to understand how our data is correlated. We gonna draw a pairplot with with all its features.
# 
# So Let's plot.

# In[ ]:


import seaborn as sns
sns.pairplot(data=data, kind='reg')


# From the above plot we can see that GRE Score,TOEFL Score, University Rating,SOP,LOR,CGPA,Research are good correlated with   your Chance of Admit.  
# So We will try first with all features.
# Lets look at columns.

# In[ ]:


data.columns


# In[ ]:


y=data['Chance of Admit ']


# In[ ]:


x=data.drop('Chance of Admit ',axis=1)
x.head()


# Lets split train and test data.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# As we have seen that all features are positively correlated. So lets take linear regression model and apply on data.

# In[ ]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# Let's train the data.

# In[ ]:


model.fit(x_train,y_train)


# Lets predict the chance of selection on  test data.

# In[ ]:


model.predict(x_test)


# In[ ]:


predict=model.predict(x_test)


# Lets see what are my real chance of selection.

# In[ ]:


y_test.head()


# As per my observation both values are nearly equal.
# 
# Lets calculate mean squared error in chance of selection.

# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predict)


# This value is  small. 
# 
# Lets calculate mean absolute error in chance of selection.

# In[ ]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predict)


# This means we will get 4% error in chance of selection.
# We have to minimize but for now lets consider it as tolerable.
# 
# Now I really want to calculate my chance of selection.
# 
# Lets consider below are my marks.

# In[ ]:


marks=['320','120','4','4.0','3.0','8.00','1']
marks=pd.DataFrame(marks).T


# In[ ]:


marks


# Lets predict my chance of selection.

# In[ ]:


model.predict(marks)


# In[ ]:


value=model.predict(marks)


# In[ ]:


print("My chance of selection:", value)


# Check your chance of selection.
# 
# **Let me know If you have any suggestion.
# 
# Please upvote if you like.**
