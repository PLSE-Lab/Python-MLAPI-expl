#!/usr/bin/env python
# coding: utf-8

# **To understand the bureau dataset and gain insights about consumer behaviour.**

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


bureau_df = pd.read_csv("../input/bureau.csv")
bureau_df.head(15)


# In[ ]:


bureau_df = bureau_df.iloc[0:500000,0:]
print(bureau_df.shape)


# In[ ]:


print('Number of NAs in the dataset is:\n',bureau_df.isnull().sum())


# In[ ]:


df1 = bureau_df[["CREDIT_ACTIVE","DAYS_CREDIT_ENDDATE","DAYS_ENDDATE_FACT","CNT_CREDIT_PROLONG",
                          "AMT_CREDIT_SUM_LIMIT","AMT_CREDIT_SUM_OVERDUE","CREDIT_TYPE","DAYS_CREDIT_UPDATE"]]
df1.head(10)


# In[ ]:


print('Number of NAs in the dataset is: \n',df1.isnull().sum())


# In[ ]:


df1 = df1.dropna(thresh=7)


# In[ ]:


print('Number of NAs in the dataset is:\n',df1.isnull().sum())


# In[ ]:


df1.shape


# ----------

# **Q1. Are there any customers who have a credit overdue beyond their credit limit?**

# In[ ]:


sns.jointplot(x="AMT_CREDIT_SUM_LIMIT", y="AMT_CREDIT_SUM_OVERDUE", data=df1)
plt.show()


# **We can observe that there no customers whose credit overdue exceeds the credit limit.**

# --------------

# **Q2. How many times was credit prolonged? if yes, what type of credit was it?**

# In[ ]:


sns.stripplot(x="CNT_CREDIT_PROLONG", y="CREDIT_TYPE", data=df1)
plt.show()


# **We can observe that maximum amount of times a credit card type credit was prolonged.  
# Since the target is for home credit, this is not an issue for the current problem statement.**

# ---------------

# **Q3. Is there is any on-going credit while applying for a home credit? if yes, how many?**

# In[ ]:


df2 = df1[df1.DAYS_CREDIT_ENDDATE > 0]
sns.countplot(x="DAYS_CREDIT_ENDDATE", data=df2, palette="Blues")
plt.show()


# In[ ]:


df2= df2.groupby(['DAYS_CREDIT_ENDDATE'])
df2.size()


# **We can observe that there are many new applications by the company with a large number of days remaining for previous credit to be repaid.**

# ---------------------

# **Conclusions -  **  
# **1. Home credit type does not have any customers which have credit payment prolonged.  **  
# **2. Many new applications are provided without having previous credit repayment done.  **  
# **3. No customer exceeds their credit payment compared to their credit limit.**
