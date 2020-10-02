#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)


# > ## Load Data

# In[ ]:


data = pd.read_csv('../input/loan.csv')
print(data.duplicated().sum())
data.shape


# In[ ]:


# see first rows
data.head()


# ### Loan Amount
# 
# ![](http://)This is the amount of money disbursed (lent) by Lending Club to each borrower.

# In[ ]:


fig = data.loan_amnt.hist(bins=50)
fig.set_title('Loan Requested Amount')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of loans')


# We can see that the higher bars correspond to loan sizes of 10000, 15000, 20000, and 35000.
# 
# This indicates that people most likely tend to ask for these loan amounts. Likely, these pre-determined loan amounts are offered as such in the Lending Club website.
# 
# Those loan values that are less frequent, could be requested by people who require a specific amount of money for a definite purpose.

# ## Lending Club growth: Increased loan amounts disbursed in time

# In[ ]:


# parse loan requested date as datetime, to make the below plots
data['issue_dt'] = pd.to_datetime(data.issue_d)
data['month']= data['issue_dt'].dt.month
data['year']= data['issue_dt'].dt.year
data[['issue_d', 'issue_dt', 'month', 'year']].head()


# In[ ]:


# plot total loan amount lent in time, segregated by the different risk markets (variable grade)
fig = data.groupby(['year', 'grade'])['loan_amnt'].sum().unstack().plot()
fig.set_title('Loan Requested Amount')
fig.set_ylabel('Loan Amount')
fig.set_xlabel('Time')


# Lending Club has lent more money from 2013 onwards, suggesting that the company is growing. They lend more money, therefore they have more customers on both the lending and borrowing sides.

# ## Bad debt: defaulted loans
# 
# Defaulted loans are those lent to borrowers that will not be able to pay the money back. Therefore the money is lost. This is the risk any finance company faces when providing credit items.

# In[ ]:


bad_indicators = ["Charged Off ",
                    "Default",
                    "Does not meet the credit policy. Status:Charged Off",
                    "In Grace Period", 
                    "Default Receiver", 
                    "Late (16-30 days)",
                    "Late (31-120 days)"]

# define a bad loan
data['bad_loan'] = 0
data.loc[data.loan_status.isin(bad_indicators), 'bad_loan'] = 1
data.bad_loan.mean()


# We see that the overall risk of Lending's Club loan book is 2.5 %. These means that 2.5% of the loans that they give will end up defaulted (unpaid).
# 
# We can go ahead and segregate the risk by risk markets (grade), see below.

# In[ ]:


dict_risk = data.groupby(['grade'])['bad_loan'].mean().sort_values().to_dict()
dict_risk


# In[ ]:


fig = data.groupby(['grade'])['bad_loan'].mean().sort_values().plot.bar()
fig.set_ylabel('Percentage of bad debt')


# We see that lower risk markets (F and G) have higher risk of detault, whereas borrowers placed in A and B grades have the lowest risk of defaulting their loans. 
# 
# Grade markets are usually estimated based on the credit history of the borrowers. Borrowers with good financial history are usually placed in markets A, and borrowers in a somewhat tighter financial situation, or for whom there is not enough financial history, are placed in markets G and F,  as the company and thus the investors are taking a greater risk when lending them money.

# In[ ]:


fig = data.groupby(['grade'])['int_rate'].mean().plot.bar()
fig.set_ylabel('Interest Rate')


# Customers placed in riskier markets are charged higher interest rates, as the company and the investors take a greater risk.

# In[ ]:


fig = data.groupby(['grade'])['loan_amnt'].sum().plot.bar()
fig.set_ylabel('Loan amount disbursed')


# Lending Club lends the majority of their loans to markets B and C. This makes financial sense, as these are borrowers typically in good financial situations, who will be able to repay their loans.

# ## Term of loans
# 
# Time over which the borrowers agree with Lending Club to repay the loan. Lending club offers 2 different term options, either 3 or 5 years.

# In[ ]:


fig = data.groupby(['grade', 'term'])['loan_amnt'].mean().unstack().plot.bar()
fig.set_ylabel('Mean loan amount disbursed')


# We can see that loans which terms are longer, are usually of bigger sizes. In other words, when the borrowers ask bigger sizes of loans, they usually agree to repay it over a longer period of time.

# In[ ]:


fig = data.groupby(['grade', 'term'])['bad_loan'].mean().unstack().plot.bar()
fig.set_ylabel('Percentage of bad debt')


# Here we see that usually, loans given over shorter periods of time are usually riskier than those given over longer periods of time (as the badt debt percentage is higher). This is not a coincidence. Usually finance companies choose not to lend money for long periods to riskier customers, as this does not help their financial situation. Rather the opposite by generating them more debt.

# In[ ]:




