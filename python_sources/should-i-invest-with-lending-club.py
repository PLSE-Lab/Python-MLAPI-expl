#!/usr/bin/env python
# coding: utf-8

# # Should I Invest with Lending Club?
# I am interested in investing.  My decision is to invest in Lending Club or not.  I have many different investment options and my goal is to maximize expected returns while minimizing risk.  Peer-2-Peer lending is new to me and came up in a recent discussion which prompted me to look at this.  
# 
# To support this analysis, I will follow this [CRIPS-DM](https://www.the-modeling-agency.com/crisp-dm.pdf) methodology[1].   I will work through this process and continue development to help learn about Lending Club Investments, data analysis with Python and writing clearly.  

# ##   Business Understanding
# My goal is to invest some amount of money and I want to identify potential returns from Lending Club.  For example, we will pretend this is $10,000.  To help make this decision, I want to figure out potential returns for Lending Club for different investment strategies and see if I can maximize return on investment.  
# 
# Details on lending club can be found on their website [2].  In brief summary, as an investor you are able to fund personal loans for a set return on investment.  Lending Club determines the interest rate you will receive on the money which is linked to the grade of the loan applicant.  The biggest risk here is the loan defaults in which case you will lose the investment.  This is my interpretation of the program as of 2017.05.29 and I recommend you read more on their website for most accurate details of the program.  
# 
# To help make my decision, I want to find a way to simulate what potential returns would be on loans over a 12 month period based on potential investment strategies through analysis of the data.  From the historical data, I am hoping to get a basic understanding of potential ROI and estimate potential methods for investing in Lending Club to maximize returns.  This will help make my decision on where to invest the hypothetical amount of money.

# ## Data Understanding
# There are three different data files included.  We will look at these below.  

# In[ ]:


# setup workspace
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# setting column width to help read data dict
pd.options.display.max_colwidth = 255


# In[ ]:


# look at input files
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# For this analysis, I will focus on the excel data dictionary and loan.csv file.  I assume the sqlite and csv file are the same and CSV files are easier for me to work with.  
# 
# I will start by bringing in the data dictionary and csv file.  With these, then I will look at the individual features in the data and try to down select to a few features to start modeling with.  

# In[ ]:


# read in data dict
df_dict = pd.read_excel('../input/LCDataDictionary.xlsx')

# look at top of dict
print('records: ' + str(len(df_dict)))
df_dict.head()


# In[ ]:


# read in data from CSV file
df = pd.read_csv('../input/loan.csv', low_memory=False)

# look up summary of df
print(df.info())

# look at top records 
df.head()


# There are 80 columns of data in the data dictionary.  Rather than look at these and try to make judgement, I will evaluate the loan data.  The next step will be to provide some high level understanding of the data,and then evaluate each feature to understand it more.  At the end of this analysis, I hope to be able to down select to an initial set of features which I can start modeling with.  
# 

# ### member_id
# With the member ID, we are interested in seeing if there are any members who have multiple loans.  If not, then we will drop.  If yes, then we will keep it for further analysis which could help identify repeat offenders for bad loans.  

# In[ ]:


df_dict[df_dict['LoanStatNew']=='member_id']


# In[ ]:


print(len(df))
print(len(df['member_id'].unique()))


# As these match, there are no repeat loans.  We will drop this  in data preparation stage.  

# ### loan_amnt 
# I assume this will be one of the key variables to look at and we want to see the following.  
# 
#  - What is the distribution of loans
#  - Does the loan_amnt match funded_amnt?  If so, we can remove one of these.  

# In[ ]:


df_dict[df_dict['LoanStatNew'].isin(['loan_amnt', 'funded_amnt', 'funded_amnt_inv'])]


# In[ ]:


df['loan_amnt'].describe()


# The loan amount has all data points and now we will look at the distribution.  

# In[ ]:


sns.distplot(df['loan_amnt']
             , kde = False
             , bins = 20)


# The distribution seems heavier in the lower ranges (<10k) but there is plenty of opportunity throughout to fund loans.  I should be good with my $10k to invest.  Now we will look at the next two variables ['funded_amnt' and 'funded_amnt_inv'] as I suspect these will be the same as loan_amnt. If so, we can drop them 

# In[ ]:


print('Mismatched funded_amnt: ' + str(len(df[df['loan_amnt'] != df['funded_amnt']])))
print('Mismatched funded_amnt: ' + str(len(df[df['loan_amnt'] != df['funded_amnt_inv']])))


# From these two, it looks like both have non-matched items.  I will keep them and do some analysis later to see if these mis-matches are for any reason and should be avoided.  Based on descriptions, I would suspect these would be similar.  Lets see a scatter plot on these values.  

# In[ ]:


df.plot(kind='scatter', x='loan_amnt', y='funded_amnt');


# In[ ]:


df.plot(kind='scatter', x='loan_amnt', y='funded_amnt_inv');


# In[ ]:


df.plot(kind='scatter', x='funded_amnt_inv', y='funded_amnt');


# My suspicion is that this will not be very useful, but we will keep it.  There may be opportunity to utilize the fact that the funded amount does not match the invested amount to identify when other investors are avoiding specific loans.  

# ## Data Preparation
# Based on the data understanding, we will drop the columns not needed.  

# In[ ]:


df.drop(['id', 'member_id']
        , axis = 1
        , inplace=True, )


# ## Modeling

# ## Evaluation

# ## References ##
# [1] -  Cross Industry Standard Process for Data Mining (CRISP-DM), https://www.the-modeling-agency.com/crisp-dm.pdf  
# [2] - Lending Club website, https://www.lendingclub.com/  
# [?] - Interesting article analyzing same data for blog, http://kldavenport.com/lending-club-data-analysis-revisted-with-python/
# [?] - Good example of showing missing data with python, https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize
