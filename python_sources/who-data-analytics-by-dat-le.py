#!/usr/bin/env python
# coding: utf-8

# The dataset contains some factors obtained from WHO. In the dataset, the descritive anaytics will be performed to give the insight of WHO data, then the predicitve model is built by the life expectancy and three other variables. The single linear models and complex models will be compared by The Sum Squared Error.

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sbn
from scipy import optimize


# In[13]:


who_data = pd.read_csv('../input/WHO.csv')


# There are many missing values in the dataset and the missing values is replaced by the mean of variables with the same region.

# In[14]:


#check which column has NaN value
who_data.isna().any()
#replicate the data
who_full = who_data
#For all columns which have NaN, I fill in NaN by the mean of that coulumns grouped by Regions
who_full['FertilityRate'].fillna(who_full.groupby('Region')['FertilityRate'].transform('mean'), inplace = True)
who_full['LiteracyRate'].fillna(who_full.groupby('Region')['LiteracyRate'].transform('mean'), inplace = True)
who_full['GNI'].fillna(who_full.groupby('Region')['GNI'].transform('mean'), inplace = True)
who_full['PrimarySchoolEnrollmentMale'].fillna(who_full.groupby('Region')['PrimarySchoolEnrollmentMale'].transform('mean'), inplace = True)
who_full['PrimarySchoolEnrollmentFemale'].fillna(who_full.groupby('Region')['PrimarySchoolEnrollmentFemale'].transform('mean'), inplace = True)
who_full['CellularSubscribers'].fillna(who_full.groupby('Region')['CellularSubscribers'].transform('mean'), inplace = True)


# The descriptive information of dataset is given as below:

# In[15]:


#Population of regions
sbn.boxplot(y='Region', x='LifeExpectancy',data=who_full,orient = 'h')
#Average GNI of each Region
gni = who_full.groupby('Region',as_index=False)['GNI'].mean()
sbn.barplot(data=gni,y='Region',x='GNI',label="Average GNI of Region")
#Relationship between GNI and Literacy Rate by Region
sbn.scatterplot(data=who_full,x='GNI',y='LiteracyRate',hue='Region')
#Relationship between GNI and Population by Region
sbn.scatterplot(data=who_full,x='GNI',y='Population',hue='Region')
#Histogram of Life Expectancy
sbn.distplot(who_full['LifeExpectancy'])


# Now, let's investigate the correlation between life expectacny and other variables.

# In[16]:


#Relationship between life expectency and GNI:
sbn.lmplot(x='LifeExpectancy',y='GNI', data=who_full)
#It cannot be modeled by linear regression


# In[17]:


#Relationship between life expectency and Fertility Rate:
sbn.lmplot(x='LifeExpectancy',y='FertilityRate', data=who_full)
#It can be modeled by linear regression


# In[18]:


#Relationship between life expectency and Literacy Rate:
sbn.lmplot(x='LifeExpectancy',y='LiteracyRate', data=who_full)
#It can be modeled by linear regression


# In[19]:


#Relationship between life expectency and Cellular Subscribers:
sbn.lmplot(x='LifeExpectancy',y='CellularSubscribers', data=who_full)
#It can be modeled by linear regression


# In[20]:


#Relationship between life expectency and Population:
sbn.lmplot(x='LifeExpectancy',y='Population', data=who_full)
#It should not be modeled by linear regression


# In[21]:


#Relationship between life expectency and Child Mortality:
sbn.lmplot(x='LifeExpectancy',y='ChildMortality', data=who_full)
#It can be modeled by linear regression


# Base on the above plots, i decided to build the linear regression models between life expectancy and Child Mortality, Fertility Rate and Literacy Rate

# Then, I define the linear regression models for single variables and complex models by all three variables then compare the efficiency of these models.

# Simple Linear models:

# In[22]:


#Define the general linear model:
def lin_model(x, a = 1, b = 0):
    return a + b * x
#define function to find best line model:
def best_lin_model(x, y):
    # Calculate standardiced variables
    x_dev = x - np.mean(x)
    y_dev = y - np.mean(y)
    
    # Complete least-squares formulae to find the optimal a0, a1
    b = np.sum(x_dev * y_dev) / np.sum( np.square(x_dev) )
    a = np.mean(y) - (b * np.mean(x))
    return [a, b]

#model of Chil Mortality
test = best_lin_model(who_full['ChildMortality'],who_full['LifeExpectancy'])
y_pred = lin_model(who_full['ChildMortality'], test[0], test[1])
print('Life Expectancy and Child Mortality: y = ',test[0],'x + ',test[1])

#model of Fertility Rate
test = best_lin_model(who_full['FertilityRate'],who_full['LifeExpectancy'])
y_pred = lin_model(who_full['FertilityRate'], test[0], test[1])
print('Life Expectancy and FertilityRate: y = ',test[0],'x + ',test[1])

#model of Literacy Rate
test = best_lin_model(who_full['LiteracyRate'],who_full['LifeExpectancy'])
y_pred = lin_model(who_full['LiteracyRate'], test[0], test[1])
print('Life Expectancy and LiteracyRate: y = ',test[0],'x + ',test[1])


#define sum of squared error (SSE) function
def sum_square_error_lg(x,y):
    parra = best_lin_model(x,y)
    y_pred = lin_model(x,parra[0], parra[1])
    error = y_pred - y
    sse = np.sum(np.square(error))
    return sse
#SSE for Child Mortality, Fertility Rate and literacy rate
sum_square_error_lg(who_full['ChildMortality'],who_full['LifeExpectancy'])
sum_square_error_lg(who_full['FertilityRate'],who_full['LifeExpectancy']) 
sum_square_error_lg(who_full['LiteracyRate'],who_full['LifeExpectancy'])


# Now, build the complex model

# In[23]:


#define the complex model error
def complex_model_error(a,b,c,d,X,y):
    y_predict = complex_model(X,a,b,c,d)
    residuals = y_predict - y
    rss=np.sum(np.square(residuals))
    return rss

#define the complex model with a tuple inputed variable X
def complex_model(X,a=1,b=0,c=0,d=0):
    var1,var2,var3 = X
    return a + b * var1 + c * var2 + d * var3

#X is tuple of variables = var1,var2,var3
X = (who_full['ChildMortality'],who_full['FertilityRate'],who_full['LiteracyRate'])

#use optimize curve fit to find the parameters
param_opt,param_cov= optimize.curve_fit(complex_model,X,who_full['LifeExpectancy'])
print(param_opt)

#Print the sum square error
print(complex_model_error(param_opt[0],param_opt[1],param_opt[2],param_opt[3],X,who_full['LifeExpectancy']))


# In simple model, the minimum SSE is on the linear regression model of Life Expectancy and Child Mortality , which is 2402.1758933
# In complex model, the Sum Squared Error is 2316.34197 which is lower than SSE of Child Mortality only.
# In conclusion, the Linear Model is improving with more inputed variables
