#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression
# 
# In this notebook, we'll build a linear regression model to predict `Sales` using an appropriate predictor variable.

# ## Step 1: Reading and Understanding the Data
# 
# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

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


advertising = pd.read_csv("/kaggle/input/Advertising.csv")


# In[ ]:


type(advertising)


# In[ ]:


advertising.head()


# In[ ]:


advertising.info()


# In[ ]:


advertising.describe()


# ### Visualising the Data

# In[ ]:


advertising.TV.plot.hist()
plt.show()


# In[ ]:


advertising.Newspaper.plot.box()
plt.show()


# In[ ]:


sns.pairplot(advertising, markers="+", diag_kind="kde")
plt.show()


# In[ ]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', markers="+", size=4)
plt.show()


# ### Correlations between variables

# In[ ]:


corrs = advertising.corr()
corrs


# In[ ]:


sns.heatmap(corrs, annot=True, cmap="Greens")
plt.show()


# As is visible from the pairplot and the heatmap, the variable `TV` seems to be most correlated with `Sales`. So let's go ahead and perform simple linear regression using `TV` as our feature variable.

# ---
# ## Step 3: Performing Simple Linear Regression
# 
# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 
# ---

# ### Generic Steps in model building
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[ ]:


X = advertising[['TV']]
y = advertising['Sales']


# In[ ]:


X.head()


# In[ ]:


y.head()


# #### Train-Test Split
# 
# You now need to split our variable into training and testing sets. You'll perform this by importing `train_test_split` from the `sklearn.model_selection` library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)


# In[ ]:


X_train.shape, X_test.shape


# ### Regression model using SciKit Learn

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


#Instantiating the linear regression model
mod = LinearRegression()


# In[ ]:


mod.fit(X_train, y_train)


# In[ ]:


mod.intercept_, mod.coef_


# From the parameters that we get, our linear regression equation becomes:
# 
# $ Sales = 6.948 + 0.054 \times TV $

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


y_train_pred = mod.predict(X_train)


# In[ ]:


r2_score(y_train, y_train_pred)


#  

#  

# ## Building a multiple linear regression model

# ### Train-test split and some  pre-processing

# In[ ]:


df_train, df_test = train_test_split(advertising, train_size = 0.7, random_state = 100)


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


y_train = df_train[['Sales']]
X_train = df_train[['TV', 'Radio']]
y_test = df_test[['Sales']]
X_test = df_test[['TV', 'Radio']]


# In[ ]:


#Instantiating the linear regression model
mod = LinearRegression()
mod.fit(X_train, y_train)


# In[ ]:


mod.intercept_, mod.coef_


# In[ ]:


y_train_pred = mod.predict(X_train)
r2_score(y_train, y_train_pred)


# ### Build a model with all 3 variables

# In[ ]:


y_train = df_train[['Sales']]
X_train = df_train[['TV', 'Radio',"Newspaper"]]
y_test = df_test[['Sales']]
X_test = df_test[['TV', 'Radio',"Newspaper"]]


# In[ ]:


#Instantiating the linear regression model
mod = LinearRegression()
mod.fit(X_train, y_train)


# In[ ]:


y_train_pred = mod.predict(X_train)
r2_score(y_train, y_train_pred)


# In[ ]:


mod.coef_


# In[ ]:


X_train.describe()


# #### Detecting multi-collinearity

# In[ ]:


y_train = df_train[['Sales']]
X_train = df_train[['TV', 'Radio', 'Newspaper']]


# In[ ]:


sns.heatmap(X_train.corr(),  cmap = "Reds", annot =True)


# In[ ]:


X_train.describe()


# #### Pre-processing the features

# In[ ]:


X_train.describe()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'scaler')


# #### Pre-processing the features

# In[ ]:


num_vars = ['TV', 'Radio']


# In[ ]:


X_train[num_vars] = scaler.fit_transform(X_train[num_vars])


# In[ ]:


scaler.data_max_


# In[ ]:


X_train.describe()


# ### Linear Regression using SciKit Learn

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train[["TV","Radio"]], y_train)


# In[ ]:


X_train.columns.values


# In[ ]:


lr.coef_


# In[ ]:


mod.coef_


# In[ ]:


lr.intercept_


# In[ ]:


y_train_pred = lr.predict(X_train[['TV', 'Radio']])
r2_score(y_train, y_train_pred)


# In[ ]:


y_train_pred = mod.predict(X_train)
r2_score(y_train, y_train_pred)


#  

# ### Calculating Variance Inflation Factor

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X_train.head()


# In[ ]:


variance_inflation_factor(X_train.values, 2)


# In[ ]:


[variance_inflation_factor(X_train.values, ind) for ind in range(3)]


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### What is a good feature selection approach?
# #### Stastically significant
#  - hypothesis testing
#  - Null hypothesis?  
#      -- There is no association between X and y
#      -- the beta coefficient is 0  
# 
# H0: coefficient is 0  
# H1: coefficient is non-zero

# #### We will use statsmodel for getting these value

# In[ ]:


import statsmodels.api as sm


# In[ ]:


X_train.head()


# Add intercept manually for statsmodel to work

# In[ ]:


X_train_sm = sm.add_constant(X_train)
X_train_sm.head()


# In[ ]:


lr = sm.OLS(y_train, X_train_sm).fit()


# In[ ]:


lr.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train_sm = sm.add_constant(X_train[["TV","Radio"]])


# In[ ]:


lr = sm.OLS(y_train, X_train_sm).fit()


# In[ ]:


lr.summary()


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[["TV","Radio"]].columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train[["TV","Radio"]].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Recursive Feature Elimination
#  - using sklearn

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


lr = LinearRegression()


# In[ ]:


rfe = RFE(lr, 2)


# In[ ]:


num_vars = ["TV","Radio","Newspaper"]


# In[ ]:


X_train[num_vars] = scaler.fit_transform(X_train[num_vars])


# In[ ]:


X_train.describe()


# In[ ]:


rfe.fit(X_train, y_train)


# In[ ]:


rfe.ranking_


# In[ ]:


rfe.support_


# In[ ]:


selected_feats=X_train.columns[rfe.support_]


# In[ ]:


#X_train[selected_feats] = scaler.fit_transform(X_train[selected_feats])


# In[ ]:


rfe.fit(X_train[selected_feats],y_train)


# In[ ]:


pred=rfe.predict(X_train[selected_feats])


# In[ ]:


r2_score(y_train,pred)


# Here, by using RFE(Recursive feature elimination) we got a 91% of accuracy (by using less features)
