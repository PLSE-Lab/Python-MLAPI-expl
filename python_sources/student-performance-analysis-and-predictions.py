#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


SP = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


SP.head()


# In[ ]:


SP.info()


# In[ ]:


SP.describe()


# In[ ]:


SP['gender'].value_counts()


# In[ ]:


SP['race/ethnicity'].value_counts()


# In[ ]:


SP['parental level of education'].value_counts()


# In[ ]:


SP['test preparation course'].value_counts()


# In[ ]:


SP['lunch'].value_counts()


# In[ ]:


sns.pairplot(SP,hue='gender',palette='cividis')


# In[ ]:


sns.countplot(x='gender',data = SP,palette='cividis')


# In[ ]:


sns.countplot(x='lunch',data=SP,palette='cividis')


# In[ ]:


sns.countplot(x='race/ethnicity',data=SP,palette='cividis')


# In[ ]:


sns.countplot(x='test preparation course',data=SP,palette='cividis')


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(x='parental level of education',data=SP,palette='viridis')


# In[ ]:


sns.boxplot(x='gender',y='writing score',data=SP,palette='viridis')


# In[ ]:


plt.figure(figsize=(10,6))
sns.swarmplot(x='test preparation course',y='writing score',data=SP,palette='viridis')


# In[ ]:


plt.figure(figsize=(10,5))
sns.stripplot(x='parental level of education',y='writing score',data=SP,palette='cividis')


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(SP['writing score'],kde =False,color='green')


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(SP['reading score'],kde =False,color='purple')


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(SP['math score'],kde =False,color='red')


# In[ ]:


sns.scatterplot(x='reading score',y='writing score',data=SP,hue='gender',palette='viridis')


# In[ ]:


sns.scatterplot(x='math score',y='writing score',data=SP,hue='gender',palette='viridis')


# In[ ]:


sns.heatmap(SP.corr(),annot=True)


# In[ ]:


mapGender = {'female':0,'male':1}
mapGroup = {'group C':3,'group D':4,'group B' :2,'group E':5,'group A':1}
mapLevel = {'some college':1,"associate's degree":2,"high school":3,'some high school':4,"bachelor's degree":5,"master's degree":6}
mapLunch = {"standard":0,"free/reduced":1}
mapPrepare = {'none':0,'completed':1}


# In[ ]:


SP.columns


# In[ ]:


SP['gender'] = SP['gender'].map(mapGender)
SP['race/ethnicity'] = SP['race/ethnicity'].map(mapGroup)
SP['parental level of education'] = SP['parental level of education'].map(mapLevel)
SP['lunch'] = SP['lunch'].map(mapLunch)
SP['test preparation course'] = SP['test preparation course'].map(mapPrepare)


# In[ ]:


SP.head()


# In[ ]:


sns.heatmap(SP.corr(),annot=True)


# ### LINEAR REGRESSION / PREDICT WRITING SCORE

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = SP.drop('writing score',axis = 1)
y = SP['writing score']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


print(lm.coef_) #this coeffecient relates to the values in X_train


# In[ ]:


coef = pd.DataFrame(lm.coef_,X.columns,columns = ['coef'])


# In[ ]:


coef


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


sns.scatterplot(y=y_test,x=predictions)
plt.title('Independent score vs Dependent score')
plt.xlabel('predicted score')


# In[ ]:


sns.distplot((y_test-predictions))


# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,predictions)


# In[ ]:


metrics.mean_squared_error(y_test,predictions)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# 
# 
# 
# ### LOGISTIC REGRESSION / PREDICT GENDER

# In[ ]:


X= SP.drop('gender',axis = 1)
y=SP['gender']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.33,random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


probability=logmodel.predict_proba(X_test)[:,1]
sns.regplot(probability,y_test,logistic=True)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[ ]:




