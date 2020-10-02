#!/usr/bin/env python
# coding: utf-8

# ## What is a Variable?
# A variable is any characteristic, number, or quantity that can be measured or counted.
# Most variables in a data set can be classified into one of two major types:
# 
# - **Numerical variables** 
# - **Categorical variables**
# 
# ========================================================================================================================
# ## Numerical variables
# The values of a numerical variable are numbers. They can be further classified into discrete and continuous variables.
# 
# ### Discrete numerical variable
# A variable which values are whole numbers (counts) is called discrete. For example, 
# 
# - Number of pets in the family
# - Number of children in the family
# 
# ### Continuous numerical variable
# A variable that may contain any value within some range is called continuous. For example, 
# 
# - Time spent surfing a website (3.4 seconds, 5.10 seconds, ...)
# - Total debt as percentage of total income in the last month (0.2, 0.001, 0, 0.75, ...)

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv('../input/lending-club-loan-data/loan.csv')
df.head()


# In[ ]:


use_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status', 'open_il_12m']
data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)   # set a seed for reproducibility
data.sample(5)


# In[ ]:


data.shape


# ## Continuous Variables
# The values of the variable vary across the entire range of the variable. This is characteristic of continuous variables.

# In[ ]:


data.loan_amnt.unique()


# In[ ]:


fig = data.int_rate.hist(bins=50)
fig.set_title("Interest Rate Graph")


# In[ ]:


fig = data.loan_amnt.hist(bins=50)
fig.set_title("Loan Amount Distributed")
fig.set_xlabel("Loan Amount")
fig.set_ylabel("Number of Loans")


# In[ ]:


fig = data.int_rate.hist(bins=50)
fig.set_title("Interest Rate Distribution")
fig.set_xlabel("Interest Rate")
fig.set_ylabel("Number of Loans")


# In[ ]:


fig = data.annual_inc.hist(bins=100)
# fig.set_xlim(0, 400000)
fig.set_title("Customer's Annual Income")
fig.set_xlabel('Annual Income')
fig.set_ylabel('Number of Customers')


# ## Discrete Variable
# Histograms of discrete variables have this typical broken shape, as not all the values within the variable range are present in the variable. 

# In[ ]:


data.open_acc.unique()


# In[ ]:


data.open_acc.dropna().unique()


# In[ ]:


fig = data.open_acc.hist(bins=100)
fig.set_xlim(0, 30)
fig.set_title('Number of open accounts')
fig.set_xlabel('Number of open accounts')
fig.set_ylabel('Number of Customers')


# In[ ]:


data.open_il_12m.unique()


# In[ ]:


fig = data.open_il_12m.hist(bins=50)
fig.set_title('Number of installment accounts opened in past 12 months')
fig.set_xlabel('Number of installment accounts opened in past 12 months')
fig.set_ylabel('Number of Borrowers')


# ## A variation of discrete variables: the binary variable
# Binary variables, are discrete variables, that can take only 2 values, therefore binary.

# In[ ]:


data.loan_status.unique()


# In[ ]:


data['defaulted'] = np.where(data.loan_status.isin(['Default']), 1, 0)


# In[ ]:


data.sample(5)


# In[ ]:


data['Charged off'] = np.where(data.loan_status.isin(['Charged off']), 1, 0)
data.sample(5)


# In[ ]:


fig = data.defaulted.hist()
fig.set_xlim(0, 2)
fig.set_title('Defaulted accounts')
fig.set_xlabel('Defaulted')
fig.set_ylabel('Number of Loans')


# ## Categorical variables
# 
# The values of a categorical variable are selected from a group of **categories**, also called **labels**. Examples are gender (male or female).
# Other examples of categorical variables include:
# - Mobile network provider (Jio, Vodafone, BSNL, ...)
# - Postcode
# 
# Categorical variables can be further categorised into ordinal and nominal variables.
# 
# ### Ordinal categorical variables
# Categorical variable in which categories can be meaningfully ordered are called ordinal. For example:
# - Student's grade in an exam (A, B, C or Fail).
# - Days of the week can be ordinal with Monday = 1 and Sunday = 7.
# - Educational level, with the categories Elementary school,  High school, College graduate and PhD ranked from 1 to 4. 
# 
# ### Nominal categorical variable
# There isn't an intrinsic order of the labels. For example, country of birth (Argentina, England, Germany) is nominal. Other examples of nominal variables include:
# - Postcode
# - Vehicle make (Citroen, Peugeot, ...)
# 
# There is nothing that indicates an intrinsic order of the labels, and in principle, they are all equal.

# In[ ]:


use_cols = ['id', 'purpose', 'loan_status', 'home_ownership']
data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)
data.sample(5)


# In[ ]:


data.home_ownership.unique()


# In[ ]:


fig = data['home_ownership'].value_counts().plot.bar()
fig.set_title('home_ownership')
fig.set_ylabel('Number of customers')


# In[ ]:


data['home_ownership'].value_counts()


# In[ ]:


data.purpose.unique()


# In[ ]:


fig = data['purpose'].value_counts().plot.bar()
fig.set_title('Loan Purpose')
fig.set_ylabel('Number of customers')


# In[ ]:


fig = data.purpose.value_counts().plot.line()


# In[ ]:


fig = data.purpose.value_counts().plot.area()


# ## Dates and Times
# Datetime variables can contain dates only, or time only, or date and time.
# - Date variables usually contain a huge number of individual categories, which will expand the feature space dramatically
# - Date variables allow us to capture much more information from the dataset if preprocessed in the right way

# In[ ]:


use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']
data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols)
data.sample(5)


# In[ ]:


data.dtypes


# In[ ]:


data['issue_date'] = pd.to_datetime(data.issue_d)
data['last_pymnt_date'] = pd.to_datetime(data.last_pymnt_d)
data[['issue_d', 'issue_date', 'last_pymnt_d', 'last_pymnt_date']].head()


# In[ ]:


data.dtypes


# In[ ]:


fig = data.groupby(['issue_date', 'grade'])['loan_amnt'].sum().unstack().plot(figsize=(14, 8))
fig.set_title('Distributed amount with time')
fig.set_ylabel('Distributed Amount $')


# In[ ]:


fig = data.groupby(['issue_date', 'grade'])['loan_amnt'].sum().plot(figsize=(14, 8))
fig.set_title('Distributed amount with time')
fig.set_ylabel('Distributed Amount $')

