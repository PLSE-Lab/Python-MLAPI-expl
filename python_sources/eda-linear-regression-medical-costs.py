#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Variable Descriptions 
# 
# #### Independent variables/features:
# 
# * age: age of primary beneficiary
# 
# * sex: insurance contractor gender, female, male
# 
# * bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# * children: Number of children covered by health insurance / Number of dependents
# 
# * smoker: Smoking
# 
# * region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# #### Dependent Variable/target:
# 
# * charges: Individual medical costs billed by health insurance

# In[ ]:


# Importing libraries and magic functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# **Exploratory Data Analysis**

# In[ ]:


# read data
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

# first glimpse at Data
df.head()


#types
df.dtypes


# In[ ]:


# Statistics Summary - Numerical variables
df.describe()


# #### Average spending on medical treatment
# On average, 13270.4 $ are being spent on medical costs.
# 
# #### Average BMI across patients
# The mean BMI is approx. 30 which is very high --> overweight
# 
# #### Average age
# avg age = 39

# In[ ]:


# check for nan
df.isnull().sum()

# check for duplicates
duplicate_df = df[df.duplicated()]
duplicate_df


# In[ ]:


# Data Cleaning

# remove duplicates
df.drop(df.index[[581]], inplace=True)

# check if removed
duplicate_df2 = df[df.duplicated()]
duplicate_df2


# In[ ]:


# Checking for outlieres - Boxplot

sns.boxplot(df.bmi)


# In[ ]:


# Create dummies

df = pd.get_dummies(df, prefix=['sex', 'smoker', 'region'])
df.head()


# In[ ]:


# Correlation
df_corr = df.corr()

# visualize correlation using heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_corr,annot= True, ax=ax)


# In[ ]:


# dropping sex_female and smoker_no

df = df.drop(['sex_female','smoker_no'], axis=1)
df.columns


# In[ ]:


# Checking distribution and correlation - Pairplot

sns.pairplot(df)


# In[ ]:


# Distribution of smoker

sns.countplot(df.smoker_yes)
sns.despine(top=True, right=True, left=True, bottom=False) 
plt.title('Distribution smoker vs. non smoker')
plt.xlabel('Smoker no=0/yes=1')

# Value count
df.smoker_yes.value_counts()


# In[ ]:


# Distribution of charges

sns.distplot(df.charges)
sns.despine(top=True, right=True, left=False, bottom=False) 


# In[ ]:


# distribution of charges depending on wether smoker or not

f= plt.figure(figsize=(12,5))

mean_smoker=df[(df.smoker_yes == 1)]["charges"].mean()

ax=f.add_subplot(121)
sns.distplot(df[(df.smoker_yes == 1)]["charges"],color='r',ax=ax)
ax.axvline(mean_smoker, color='b', linestyle='--')
ax.set_title('Distribution of charges for smokers')

mean_non_smoker = df[(df.smoker_yes == 0)]["charges"].mean()

ax=f.add_subplot(122)
sns.distplot(df[(df.smoker_yes == 0)]['charges'],color='g',ax=ax)
ax.axvline(mean_non_smoker, color='b', linestyle='--')
ax.set_title('Distribution of charges for non-smokers')

sns.despine(top=True, right=True, left=False, bottom=False) 

plt.show()


# In[ ]:


# Distribution for charges depending on gender

g= plt.figure(figsize=(12,5))

ax=g.add_subplot(121)
sns.distplot(df[(df.sex_male == 1)]["charges"],color='dodgerblue',ax=ax)
ax.set_title('Distribution of charges for men')

ax=g.add_subplot(122)
sns.distplot(df[(df.sex_male == 0)]['charges'],color='orchid',ax=ax)
ax.set_title('Distribution of charges for women')

sns.despine(top=True, right=True, left=False, bottom=False) 


# In[ ]:


# Charges - BMI - Smoker
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker_yes')

sns.despine(top=True, right=True, left=True, bottom=False) 
plt.title("Charges - BMI - Smoker")


# ### Setting up the model

# In[ ]:


# Splitting Data 75% Train - 25% Test
from sklearn.model_selection import train_test_split

# clarify what is y and what is x label
y = df['charges']
x = df.drop(['charges'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 67)


# In[ ]:


# Linear Regression
lr = LinearRegression()

# fitting
lr_model= lr.fit(x_train, y_train)

# get intercept & coefficients
lr.coef_
lr.intercept_

# make predictions
y_train_pred = lr_model.predict(x_train)
y_test_pred = lr_model.predict(x_test)

# score results
print("The model's accuracy score before polynomial feature adjustment is:",round(lr_model.score(x_test,y_test),2))


# In[ ]:


# Polynomial Features

from sklearn.preprocessing import PolynomialFeatures

X = df.drop(['charges'], axis = 1)
Y = df.charges

quad = PolynomialFeatures (degree = 3)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))


# In[ ]:


# Model Evaluation

#MAE
from sklearn.metrics import mean_absolute_error

mean_absolute_error(Y_train, Y_train_pred)

#MSE
from sklearn.metrics import mean_squared_error

mean_squared_error(Y_train, Y_train_pred)

#R squared
from sklearn.metrics import r2_score

print('R2 score train data:',r2_score(Y_train, Y_train_pred))

print('R2 score test data:', r2_score(Y_test, Y_test_pred))


# In[ ]:


plt.figure(figsize=(10,6))

plt.scatter(Y_train_pred,Y_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(Y_test_pred,Y_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()


# In[ ]:




