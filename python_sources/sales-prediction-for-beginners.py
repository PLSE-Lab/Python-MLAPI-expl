#!/usr/bin/env python
# coding: utf-8

# **Problem Statement**
# 
#     A company spends some money for advertising into three different channels such as Television, Radio and Newspaper for increasing the sales. So we going to build a machine learning model which will predict the sales based on the amount which is company spend for each platforms.

# In[ ]:


#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/advertising.csv/Advertising.csv',index_col = 'Unnamed: 0')


# In[ ]:


df.head()


# In[ ]:


#checking the length of data
len(df)


# In[ ]:


#checking the duplicates
df.duplicated().any()


# In[ ]:


#checking null values
df.isnull().sum()


# There is no duplicates to drop. As well as there is no NULL values to treat.

# In[ ]:


#Basic statistical report
df.describe()


# From the report, each columns has the length of 200. so there is no missing values. And all the columns almost follows a normal distribution.

# In[ ]:


import seaborn as sns
sns.pairplot(df, x_vars = ['TV','radio','newspaper'], y_vars='sales',size=7, kind='reg')


# From the above pair plot, we came known that the television is highly positively correlated with sales and the radio also positively correlated with sales but not as much as television. Finally the newspaper weakly correlated with sales. For predicting the sales , TV and radio is important than newspaper.

# In[ ]:


X = df[["TV","radio","newspaper"]] 
y = df.sales


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap='RdBu_r')
plt.show()


# From the above heatmap, radio and newspaper are suffered with multi collinearity. so we have to drop either radio or newspaper. I'm going to use OLS to confirm which one we going to drop from dataset.

# In[ ]:


#OLS - oridnary least square
import statsmodels.api as sm
model = sm.OLS(y, X).fit()
model.summary()


# From the OLS model, I found the p-value for radio and newspaper. both are less than 0.05, so we could take both for create a model. since the radio and newspaper suffered with multi collinearity, I choose newspaper to drop because it has a higher p-value than the radio.

# In[ ]:


X = df[["TV","radio"]] 
y = df.sales


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)


# In[ ]:


ypred = model.predict(xtest)


# In[ ]:


result = pd.DataFrame()
result['xtest - tv'] = xtest['TV'].copy()
result['xtest - radio'] = xtest['radio'].copy()
result['ytest'] = ytest.copy()
result['ypred'] = ypred.copy()
result.head()


# In[ ]:


from sklearn import metrics
print('MAE : ', metrics.mean_absolute_error(ytest,ypred))
print('MSE : ', metrics.mean_squared_error(ytest,ypred))
print('RMSE : ', np.sqrt(metrics.mean_squared_error(ytest,ypred)))
print('R-Squared : ', (metrics.r2_score(ytest,ypred))*100)

