#!/usr/bin/env python
# coding: utf-8

# **Medical Costs analysis using Simple Linear Regression**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load your dataset

# In[ ]:


df = pd.read_csv("../input/insurance.csv")


# Firstly,  let's see basic information about our data

# In[ ]:


df.head()


# In[ ]:


df['region'].value_counts()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ext part is to graphically represent our numerical data. We will plot pairplot graph form seaborn library. 

# In[ ]:


sns.pairplot(df)


# The variable of interest is "charges". We want to predict what would be medical costs for specific individual, based on other given information. Let's see distribution of data for "charges".
# 
# From the figure bellow we can conclude that the variable "charges" do not possess normal distribution of data, but it has mixture distribution. That could be a problem for further assumptions.

# In[ ]:


sns.distplot(df["charges"],fit=norm)
fig = plt.figure()
res = stats.probplot(df["charges"], plot=plt)


# We want to improve normality of our data distribution so we will use simple log transformation.

# In[ ]:


df["charges"]=np.log(df['charges'])

sns.distplot(df["charges"],fit=norm)
fig = plt.figure()
res = stats.probplot(df["charges"], plot=plt)


# As we can see from the figures above, the log transformation has improved distribution of data and normality, which will help us to achieve better performances of our predictions.

# Next figure is a heatmap. It represents mutual correlation of numerical categories from our dataset. Interesting fact -  Children category (Number of children covered by health insurance) has the lowest correlation with "charges". Personally, I thought it would be vice versa.

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


df.columns


# We have three non numerical categories: sex, smoker and region. We want to use them too. So the next step is to convert these variables to binary values by using "One hot encoding" method.

# In[ ]:


sex_dummy = pd.get_dummies(df['sex'])
smoker_dummy = pd.get_dummies(df['smoker'])
region_dummy = pd.get_dummies(df['region'])

df = pd.concat([df,sex_dummy,smoker_dummy,region_dummy], axis=1)

df.rename(columns={'no': 'non-smoker', 'yes': 'nicotian'}, inplace=True)


# We can see last 8 columns at dataframe below (which represent converted categories)

# In[ ]:


df.head(10)


# In[ ]:


We have successfully transformed our categorical variables, so we can remove these original categories from our Dataframe.


# In[ ]:


df = df.drop(['sex','smoker','region'], axis=1)


# We have prepared our data for further processing. Finally, we can import, initialize and use the Linear Regression model.

# In[ ]:


df.head(10)


# In[ ]:


from sklearn.model_selection import train_test_split


# Eleven dataframe categories will be used as inputs X for the model. And we want to fit our model according to "charges" category - output Y of the model

# In[ ]:


X = df[['age', 'female','male','non-smoker','nicotian','northeast','northwest','southeast','southwest','bmi', 'children']]
y = df['charges']


# We will use train_test_split function to divide our data to training and testing data.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# Procedure for importing and fitting the model.

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# We will create a new dataframe to present estimated coeffcieints of our model. First one is the intercept, and other coefficients are in correlation with specific categories. 

# Interesting fact number 2: Quit smoking! We can observe that smoker_label category has the highest influence on increasing medical costs. It has stronger influence than all other analyzed parameters TOGETHER.

# In[ ]:


print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)


# Final part is to use fitted model for predicting new values (based on prepared X_test array)

# In[ ]:


predictions = lm.predict(X_test)
print("Predicted medical costs values:", predictions)


# Graphical comparison of expected values (y_test) and predicted values (predictions)

# In[ ]:


plt.scatter(y_test, predictions)


# Also, let's see error distribution graph of our predictions. Very close to normally distributed data.

# In[ ]:


sns.distplot((y_test-predictions), bins=50)


# Finally, let's print MAE and MSE erorrs for entire test data.

# In[ ]:


from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




