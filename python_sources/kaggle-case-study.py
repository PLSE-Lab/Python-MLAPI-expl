#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df1=pd.read_csv('../input/bureau_balance.csv', chunksize=100)
for i in df1:
    df1=pd.DataFrame(i)
    break
print(df1)


#  ques-1 what is the month balance of previous credits according to the credit status?
#     Credit Status-C means closed, X means status unknown, 0 means no DPD, 1 means maximal 
# 

# In[ ]:


sns.swarmplot(x="STATUS", y="MONTHS_BALANCE", data=df1)


# In[ ]:


df2=pd.read_csv('../input/installments_payments.csv', chunksize=20)


# In[ ]:


for i in df2:
    df2=pd.DataFrame(i)
    break


# In[ ]:


print(df2)


# Finding the difference between the actual payment date and the supposed payment date(relative to application date of current loan)
# Finding the difference between the actual amont paid and the prescribed installment amount

# In[ ]:


df2['days_difference']=df2['DAYS_ENTRY_PAYMENT']-df2['DAYS_INSTALMENT']
df2['AMT_DIFFERENCE']=df2['AMT_PAYMENT']-df2['AMT_INSTALMENT']


# ques-2 According to the version of installment, what is the amount difference and days difference between the prescribed and actual installment amount and date?

# In[ ]:


sns.barplot(x="AMT_DIFFERENCE", y="days_difference", hue='NUM_INSTALMENT_VERSION',data=df2)
plt.tight_layout()


# In[ ]:


df3 = pd.read_csv('../input/bureau.csv')


# Ques3
# 
# For each application, find the past record of credits based on the credit status
# 

# In[ ]:


df3.groupby(['SK_ID_CURR','CREDIT_ACTIVE']).size()


# Ques4
# 
# To check the Apllications with more than 4 times the previous credit has been prolonged
# 
# Helps in segregating the applicants with high risk of credit been prolonged
# 

# In[ ]:


df3.loc[df3['CNT_CREDIT_PROLONG']> 4 ]


# In[ ]:


df4 = df3.loc[df3['CNT_CREDIT_PROLONG'] >=1 ]


sns.barplot(x="CNT_CREDIT_PROLONG", y="AMT_CREDIT_SUM_DEBT", data=df4)


# In[ ]:


application = pd.read_csv('../input/application_train.csv', encoding='iso-8859-1')


# Ques-5
# Correlation between No. of enquiries to credit bureau and Repayment Status
# 
# AMT_REQ_CREDIT_BUREAU_HOUR - Number of enquiries to Credit Bureau about the client one hour before application.
# 
# AMT_REQ_CREDIT_BUREAU_DAY - Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application).
# 
# AMT_REQ_CREDIT_BUREAU_WEEK - Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application).
# 
# AMT_REQ_CREDIT_BUREAU_MON - Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application).
# 
# AMT_REQ_CREDIT_BUREAU_QRT - Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application).
# 
# AMT_REQ_CREDIT_BUREAU_YEAR - Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application).
# 

# In[ ]:


# Selecting required columns
cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

application.groupby("TARGET")[cols].max().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("Maximum enquries made by defaulters and repayers")

application.groupby("TARGET")[cols].mean().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("average enquries made by defaulters and repayers")

application.groupby("TARGET")[cols].std().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("standard deviation in enquries made by defaulters and repayers")

plt.show() 

