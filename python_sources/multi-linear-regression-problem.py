#!/usr/bin/env python
# coding: utf-8

# ## 50_Starups dataset

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# to handle datasets
# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import mean_squared_error

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[6]:


# load dataset
data = pd.read_csv("../input/50_Startups.csv")
data.head()


# 1. ### Types of variables 
# 
# Let's go ahead and find out what types of variables there are in this dataset

# In[7]:


# let's inspect the type of variables in pandas
data.dtypes


# There are a mixture of categorical and numerical variables. Numerical are those of type int and float. Categorical those of type object.

# In[8]:


# find categorical variables
categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))


# In[9]:


# find numerical variables
numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))


# In[10]:


# view of categorical variables
data[categorical].head()


# In[11]:


# view of numerical variables
data[numerical].head()


# *All numerical are continuous variables.*

# In[12]:


# let's visualise the values of the discrete variables
for var in ['State']:
    print(var, ' values: ', data[var].unique())


# #### Types of variables, summary:
# 
# - 1 categorical variables: State  values:  ['New York' 'California' 'Florida']
# - 4 numerical variables: All continuous

# ### Types of problems within the variables
# 
# #### Missing values

# In[13]:


# let's visualise the percentage of missing values
data.isnull().mean()


# NO missing values
# 
# #### Outliers

# In[14]:


numerical = [var for var in numerical if var not in['Profit']]
numerical


# In[15]:


# let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution
for var in numerical:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of houses')
    fig.set_xlabel(var)

    plt.show()


# None contain outliers. 

# #### Outlies in discrete variables
# 
# Let's calculate the percentage of passengers for each  of the values that can take the discrete variables in the titanic dataset. I will call outliers, those values that are present in less than 1% of the passengers. This is exactly the same as finding rare labels in categorical variables. Discrete variables, in essence can be pre-processed / engineered as if they were categorical. Keep this in mind.

# In[16]:


# outlies in discrete variables
for var in ['State']:
    print(data[var].value_counts() / np.float(len(data)))
    print()


# **State** does not contain outliers, as all its numbers are present in at least 30% of the passengers.

# ### Separate train and test set

# In[17]:


# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(data, data.Profit, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape


# ### Encode categorical variables 

# In[18]:


categorical


# In[19]:


X_train=pd.get_dummies(X_train,columns=categorical,drop_first=True)


# In[20]:


X_test=pd.get_dummies(X_test,columns=categorical,drop_first=True)


# In[21]:


X_train.head()


# In[22]:


#let's inspect the dataset
X_train.head()


# ### Feature scaling

# In[23]:


X_train.describe()


# In[24]:


# fit scaler
scaler = StandardScaler() # create an instance
scaler.fit(X_train) #  fit  the scaler to the train set for later use


# The scaler is now ready, we can use it in a machine learning algorithm when required. See below.
# 
# ### Machine Learning algorithm building
# 
# #### xgboost

# In[25]:


xgb_model = xgb.XGBRegressor()

eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

pred = xgb_model.predict(X_train)
print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = xgb_model.predict(X_test)
print('xgb test mse: {}'.format(mean_squared_error(y_test, pred)))


# #### Random Forests

# In[26]:


rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

pred = rf_model.predict(X_train)
print('rf train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = rf_model.predict(X_test)
print('rf test mse: {}'.format(mean_squared_error(y_test, pred)))


# #### Support vector machine

# In[27]:


SVR_model = SVR()
SVR_model.fit(scaler.transform(X_train), y_train)

pred = SVR_model.predict(scaler.transform(X_train))
print('SVR train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = SVR_model.predict(scaler.transform(X_test))
print('SVR test mse: {}'.format(mean_squared_error(y_test, pred)))


# #### Regularised linear regression

# In[28]:


lin_model = Lasso(random_state=2909)
lin_model.fit(scaler.transform(X_train), y_train)

pred = lin_model.predict(scaler.transform(X_train))
print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = lin_model.predict(scaler.transform(X_test))
print('linear test mse: {}'.format(mean_squared_error(y_test, pred)))


# ### Feature importance

# In[29]:


importance = pd.Series(rf_model.feature_importances_)
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))


# In[30]:


importance = pd.Series(xgb_model.feature_importances_)
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))


# In[31]:


importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))

