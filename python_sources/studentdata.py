#!/usr/bin/env python
# coding: utf-8

# # ML_SLR_ASSG

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns


# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


stud = pd.read_csv('/kaggle/input/Student_v2.csv')
stud.head()


# In[ ]:


stud.info()


# In[ ]:


stud.describe()


# In[ ]:


stud.isnull().sum()


#     there are no null values in given dataset

# In[ ]:


stud["Number of hours studied"].plot.hist()
plt.show()


# In[ ]:


stud["Number of classes present"].plot.hist()
plt.show()


# In[ ]:


stud["Number of hours studied"].plot.box()
plt.show()


# In[ ]:


stud["Number of classes present"].plot.box()
plt.show()


# In[ ]:


stud["GMAT score"].plot.box()
plt.show()


# In[ ]:


stud["UG CGPA"].plot.hist()
plt.show()


#     From above we can incurr that data is clean, there are no any "Null Values",there are no any "Outliers" too
#     Hence, we can go for further checks and validations

# In[ ]:


sns.pairplot(stud, markers="+", diag_kind="kde")
plt.show()


# In[ ]:


sns.pairplot(stud, x_vars=['UG CGPA', 'GMAT score', 'Number of hours studied'], y_vars='PG_CGA', markers="+", size=4)
plt.show()


# ## Correlation between Variables

# In[ ]:


studcorr=stud.corr()
studcorr


# In[ ]:


sns.heatmap(studcorr,annot=True,cmap='seismic')
plt.show()


#     From the above heatmap,pairplot the variable "PG_CGA" is Highly correlated with "UG CGPA", so let's perform a "Simple Linear Regression" using "UG CGPA" as our feature variable.

# ## Simple Linear Regression

# The Equation for **SLR** is:
# 
# $y = c + m \times UGCGPA $
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 

# In[ ]:


x = stud[['UG CGPA']]
y = stud['PG_CGA']


# In[ ]:


x.head()


# In[ ]:


y.head()


# **TRAIN-TEST-SPLIT**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=100)


# In[ ]:


x_train.shape,x_test.shape


# **Now Implementing SLR using Sklearn**

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


mod=LinearRegression()


# In[ ]:


mod.fit(x_train,y_train)


# In[ ]:


mod.intercept_,mod.coef_


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


y_train_pred=mod.predict(x_train)


# In[ ]:


r2_score(y_train,y_train_pred)


#     Here, we observed an R-Square value of 22% apprx and lets test in on TEST DATA

# In[ ]:


y_test_pred=mod.predict(x_test)


# In[ ]:


r2_score(y_test,y_test_pred)


# ### MULTIPLE LINEAR REGRESSION

# In[ ]:


stud.corr()


# In[ ]:


x = stud[['UG CGPA','GMAT score','Number of friends','Number of classes present']]


# In[ ]:


y = stud['PG_CGA']


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


lr.coef_


# In[ ]:


lr.intercept_


# In[ ]:


prd1 = lr.predict(x_train)


# In[ ]:


r2_score(y_train,prd1)


# ### *Here, by  using appropriate variables that are required created a MLR model with R-Square value approx to 59%*

# ## SCALING(MIN_MAX Scaling)

# In[ ]:


x_train.describe()


# In[ ]:


num_vars = ['UG CGPA','GMAT score','Number of friends','Number of classes present']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


x_train[num_vars] = scaler.fit_transform(x_train[num_vars])


# In[ ]:


x_train.describe()


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr1=LinearRegression()


# In[ ]:


lr1.fit(x_train,y_train)


# In[ ]:


lr1.intercept_,lr1.coef_


# In[ ]:


prd2=lr1.predict(x_train)


# In[ ]:


r2_score(y_train,prd2)


# ### From the above after scaling also R-square value remains the same and we here used a very less 
# ### number of variables and appropriate variables to get the best model possible.

# In[ ]:




