#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display
pd.set_option('display.max_columns', 100)


# In[19]:


train = pd.read_csv('../input/train.csv', index_col = 0)
train.head()


# In[20]:


train.shape


# In[21]:


train.info()


# #### Only one column employment type has missing entries
# 

# In[22]:


target = train['LOAN_DEFAULT']
target.value_counts().plot.bar()


# ***Imbalanced Target class***

# **convert columns having data in year and month in month only**

# In[23]:


new = []
for items in train['AVERAGE_ACCT_AGE'].values: 
    y = int(items[0]) * 12
    if items[6]== 'm':
        m = int(items[5])
    else:
        m = int(items[5:7])
    total = y + m
    new.append(total)

train.loc[:, 'AVERAGE_ACCT_AGE'] = new



new = []
for items in train['CREDIT_HISTORY_LENGTH'].values: 
    y = int(items[0]) * 12
    if items[6]== 'm':
        m = int(items[5])
    else:
        m = int(items[5:7])
    total = y + m
    new.append(total)

train.loc[:, 'CREDIT_HISTORY_LENGTH'] = new


# **Check columns credit_history_length and Average_ACCT_AGE**

# In[24]:


train.head()


# **Convert Data Types of data columns and employment type to category**

# In[25]:


train.columns


# In[26]:


train['DATE_OF_BIRTH'] = pd.to_datetime(train['DATE_OF_BIRTH'])
train['DISBURSAL_DATE'] = pd.to_datetime(train['DISBURSAL_DATE'])
train['EMPLOYMENT_TYPE'] = train['EMPLOYMENT_TYPE'].astype('category')


# In[27]:


train.info()


# #### Treat Missing Values of Employment_Type

# In[28]:


train['EMPLOYMENT_TYPE'] = train['EMPLOYMENT_TYPE'].fillna(method = 'bfill')


# In[29]:


train = train.replace({'PERFORM_CNS_SCORE_DESCRIPTION':{'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',
                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',
                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',
                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',
                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'High',
                                                       'M-Very High Risk':'High','Not Scored: More than 50 active Accounts found':'Not Scored',
                                                       'Not Scored: Only a Guarantor':'Not Scored','Not Scored: Not Enough Info available on the customer':'Not Scored',
                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not Scored','Not Scored: No Updates available in last 36 months':'Not Scored',
                                                       'Not Scored: Sufficient History Not Available':'Not Scored', 'No Bureau History Available':'Not Scored'
                                                       }})
                                                        


# In[30]:


train['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts()


# In[31]:


corr_mat = train.corr()
f,ax = plt.subplots(figsize = (12,12))
sns.heatmap(corr_mat, vmax = 0.8, square= True)


# In[32]:


k = 10
cols = corr_mat.nlargest(k, 'LOAN_DEFAULT')['LOAN_DEFAULT'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# #### Disbursed amount and asset cost highly correlated, also correlated to ltv
# #### Average account age and credit history length correlated
# #### Predictors SECX also highly correlated
# #### ltv and current_pin_codeid correlated
# #### PRI. OVERDUE.ACCTS and DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS highly correlated

# 

# In[33]:


corr_mat['LOAN_DEFAULT'].sort_values(ascending = False)


# #### No predictor has strong correlation with loan_default 

# #### Hypothesis: Disbursed amount, ltv, delinquent accounts in last six months, primary current balance, credit history length, cns_score should contribute more in the loan_default.
