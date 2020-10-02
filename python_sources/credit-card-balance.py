#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information
# The dataset contains information about 400 credit card holders, their credit balances, demographic features, etc. 

# ## Contents:
# There are **11** variables.
# 
# #### **Income:** Income of the customer.
# #### **Limit:** Credit limit provided to the customer.
# #### **Rating:** The customer's credit rating.
# #### **Cards:** The number of credit cards the customer has. 
# #### **Age:** Age of the customer.
# #### **Education:** Educational level of the customer.
# #### **Gender:** Sex of the customer.
# #### **Student:** If the customer is a student or not.
# #### **Married:** If the customer is married.
# #### **Ethnicity:** Ethnicity of the customer.
# #### **Balance:** Credit balance of the customer.

# ## Importing the required packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## Importing the dataset

# In[ ]:


import os
print(os.listdir("../input"))
Credit_df = pd.read_csv("../input/Credit.csv",index_col=0)


# ## Dimensions of the data

# In[ ]:


print(Credit_df.shape)


# The dataset has 400 observations and 11 variables.

# ## Glimpse of the data

# In[ ]:


Credit_df.head()


# ## Structure of the data and its data types

# In[ ]:


Credit_df.info()


# ## Descriptive Stats

# In[ ]:


Credit_df.describe()


# It is interesting to see that there are customers who have **0** credit balance. We would be exploring this aspect at a later stage of the analysis.

# ## Missing value check

# In[ ]:


Missing_val =Credit_df.isnull().sum()
print(Missing_val)


# There are no missing values in the data.

# ## Data Exploration

# In[ ]:


sns.distplot(Credit_df.Balance, color='teal')


# The response variable **Balance** is normally distributed.

# ## Exploration of the categorical variables

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15, 6))
f.subplots_adjust(hspace=.3, wspace=.25)
Credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')
Credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')
Credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')
Credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')


# The response variable is normally distributed for each of the categorical variables.

# ## Correlation check

# In[ ]:


sns.heatmap(Credit_df.corr(), cmap='BuGn')


# The variables Limit and Rating are highly correlated with Balance as well as they are correlated among themselves, hence one of them needs to be dropped.

# ## Variance Inflation Factor

# In[ ]:


Credit_df_Numeric=pd.DataFrame(Credit_df[['Income','Limit','Rating','Cards','Age','Education','Balance']])
X = Credit_df_Numeric.assign(const=0)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# The results of the VIF also confirms those of the correlation matrix. The variables Limit and Rating are correlated among themselves. The variables are dependent among themselves as Credit Limit depends on the customer's credit rating and hence we would drop the Limit for the modelling purpose.

# ## Mulivariate Linear Regression

# In[ ]:


mod_reg = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity',
               data = Credit_df).fit()
mod_reg.summary()


# ## Customers with "0" Credit Balance

# In[ ]:


len(Credit_df[Credit_df.Balance==0])


# There are 90 customers with 0 credit balance. These are most probably Inactive Customers. We will repeat our analysis with only the **Active Customer Base**.

# ## Active Customer Base

# In[ ]:


Credit_df_active= Credit_df[Credit_df.Balance >0]


# ## Data Exploration of the Active Customer Base

# In[ ]:


sns.distplot(Credit_df_active.Balance, color='teal')


# The data is normally distributed for the Active Customer Base as well.

# ## Categorical Variables of the Active Customer Base

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15, 6))
f.subplots_adjust(hspace=.3, wspace=.25)
Credit_df_active.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')
Credit_df_active.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')
Credit_df_active.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')
Credit_df_active.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')


# The response variable is normally distributed for the categorical variables of the Active Customer Base as well.

# ## Multivariate Linear Regression for the Active Customer Base

# In[ ]:


mod_active = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity',
               data = Credit_df_active).fit()
mod_active.summary()


# The model fits more accurately for the Active Customer Base.
