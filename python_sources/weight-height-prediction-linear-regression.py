#!/usr/bin/env python
# coding: utf-8

# # Weight-Height Prediction using Linear Regression
# 
# simple linear regression model to predict the height of person for given weight

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


weight_height_dataset = pd.read_csv('../input/weight-height/weight-height.csv')
weight_height_dataset.head()


# In[ ]:


weight_height_dataset.info()


# In[ ]:


weight_height_dataset.describe()


# In[ ]:


weight_height_dataset.duplicated().sum()


# In[ ]:


weight_height_dataset.isnull().sum()


# # Univariate analysis

# In[ ]:


sns.boxplot(weight_height_dataset.Weight)
plt.show()


# In[ ]:


sns.boxplot(weight_height_dataset.Height)
plt.show()


# ## IQR method to remove outliers
# 
# outliers affect the regression line

# In[ ]:


q1 = weight_height_dataset['Weight'].quantile(0.25)
q3 = weight_height_dataset['Weight'].quantile(0.75)
iqr = q3 - q1
ul = q3 + 1.5*iqr
ll = q1 - 1.5*iqr
weight_height_dataset = weight_height_dataset[(weight_height_dataset.Weight >= ll) & (weight_height_dataset.Weight <= ul)]


# In[ ]:


q1 = weight_height_dataset['Height'].quantile(0.25)
q3 = weight_height_dataset['Height'].quantile(0.75)
iqr = q3 - q1
ul = q3 + 1.5*iqr
ll = q1 - 1.5*iqr
weight_height_dataset = weight_height_dataset[(weight_height_dataset.Height >= ll) & (weight_height_dataset.Height <= ul)]


# # Bivariate analysis

# In[ ]:


sns.scatterplot(weight_height_dataset.Weight,weight_height_dataset.Height,color='g')
plt.show()


# ### Split the dataset into train and test
# 70:30 ratio

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x = pd.DataFrame(weight_height_dataset['Weight'])
y = pd.DataFrame(weight_height_dataset['Height'])


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=123)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)


# # Apply Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()
lr.fit(xtrain,ytrain)
yPredict = lr.predict(xtest)


# In[ ]:


print(lr.coef_)
print(lr.intercept_)


# ### Equation of line : y = 0.11x + 48.5

# ## Check Rsquare and RMSE for accuracy

# In[ ]:





# In[ ]:


r2_score(ytest,yPredict)


# In[ ]:


np.sqrt(mean_squared_error(ytest,yPredict))


# # Plotting the Regression Line

# In[ ]:


sns.scatterplot(xtrain.Weight,ytrain.Height)
plt.plot(xtrain.Weight,lr.predict(xtrain),c='r')
plt.show()


# In[ ]:


sns.scatterplot(xtest.Weight,ytest.Height,color='r')
plt.plot(xtest.Weight,yPredict,c='b')
plt.show()


# # Linear Regression Assumptions

# In[ ]:


residual = ytest - yPredict


# ### 1. No pattern in residual

# In[ ]:


sns.residplot(yPredict,residual)
plt.show()


# ### 2. Normal Distribution

# In[ ]:


import pylab
import scipy.stats as stats


# In[ ]:


stats.probplot(residual.Height,plot=pylab)
plt.show()

Shapio Wilk test of normality

h0(null hypothesis) : residual is normal distribution
h1(alternate hypothesis) : residual is not normal distribution
# In[ ]:


test,pvalue = stats.shapiro(residual)
print(pvalue)

pvalue > 0.05, so we fail to reject null hypothesis. Thus we conclude distribution is normal.

-- here alpha error = 0.05
# ### 3. Multicollinearity

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


vif = [variance_inflation_factor(weight_height_dataset.drop('Gender',axis=1).values,i) for i in range(weight_height_dataset.drop('Gender',axis=1).shape[1])]


# In[ ]:


pd.DataFrame({'vif':vif},index=['Weight','Height']).T


# ### 4. Heteroscadastic
# 
# if heteroscadastic, linear regression cannot be used. 
# 
# h0: residual is not heteroscadastic
# 
# h1: residual is heteroscadastic

# In[ ]:


from statsmodels.stats.api import het_goldfeldquandt


# In[ ]:


df = pd.DataFrame(weight_height_dataset['Height'])


# In[ ]:


residual2 = df - lr.predict(df)


# In[ ]:


ftest,pvalue,result = het_goldfeldquandt(residual2,weight_height_dataset.drop('Gender',axis=1))
print(pvalue)

 pvalue > 0.05,hence we fail to reject null hypothesis.ie it is not heteroscadatic.
# ### 5. Auto-correlation
# 
# The errors should not be auto correlated in nature as it will violate the assumptions of the linear regression model.
# 
# - Durbin Watson Test
# 
# 0 to 4
# 
# [0-2) - (+)ve coorelation
# 
# =2 - no correlation
# 
# (2-4] - (-)ve correlaion

# In[ ]:


from statsmodels.stats.stattools import durbin_watson


# In[ ]:


print(durbin_watson(residual))

approx 2, so no correlation between residuals.
# ### 6. Linearity
# 
# - Rainbow Test
# 
# h0: linear in nature
# 
# h1: not linear in nature

# In[ ]:


import statsmodels.api as sms


# In[ ]:


model = sms.OLS(y,x).fit()
model.summary()


# In[ ]:


test,pvalue = sms.stats.diagnostic.linear_rainbow(model)
pvalue

p > 0.05 ,hence we fail to reject null hypothesis. Hence data is linear in nature
# # Using One hot Encoding & Scaling to improve accuracy

# In[ ]:


weight_height_dataset[['Female','Male']] = pd.get_dummies(weight_height_dataset['Gender'])
weight_height_dataset.head()


# In[ ]:


weight_height_dataset.drop('Gender',axis=1,inplace=True)


# In[ ]:


weight_height_dataset.head()


# In[ ]:


temp = pd.DataFrame(StandardScaler().fit_transform(weight_height_dataset),columns=weight_height_dataset.columns)
temp.head()


# In[ ]:


x = temp.drop('Height',axis=1)
y = temp['Height']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=123)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)


# In[ ]:


lr = LinearRegression()
lr.fit(xtrain,ytrain)
yPredict = lr.predict(xtest)


# In[ ]:


r2_score(ytest,yPredict)


# In[ ]:


np.sqrt(mean_squared_error(ytest,yPredict))

we can see on using encoder & scaling our accuracy increased by 1% & also reducing the RMSE.