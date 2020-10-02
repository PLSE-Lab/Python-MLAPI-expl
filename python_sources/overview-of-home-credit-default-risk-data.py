#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[2]:


application_test=pd.read_csv("../input/application_test.csv")
application_train=pd.read_csv("../input/application_train.csv")
bureau=pd.read_csv("../input/bureau.csv")
bureau_balance=pd.read_csv("../input/bureau_balance.csv")
credit_card_balance=pd.read_csv("../input/credit_card_balance.csv")
installments_payments=pd.read_csv("../input/installments_payments.csv")
POS_CASH_balance=pd.read_csv("../input/POS_CASH_balance.csv")
previous_application=pd.read_csv("../input/previous_application.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")


# In[3]:


application_test.head()


# In[4]:


application_train.head()


# In[5]:


bureau.head()


# In[6]:


bureau_balance.head()


# In[7]:


credit_card_balance.head()


# In[8]:


installments_payments.head()


# In[9]:


POS_CASH_balance.head()


# In[10]:


previous_application.head()


# In[11]:


sample_submission.head()


# In[12]:


from collections import Counter
count=Counter(application_train["TARGET"])


# In[14]:


#Distribution of Target
import matplotlib.pyplot as plt
plt.pie([float(v) for v in count.values()], labels=[float(k) for k in count])


# In[34]:


# NAME_CONTRACT_TYPE wise distribution of TARGET
application_train.groupby(["TARGET",'NAME_CONTRACT_TYPE']).size()


# In[35]:


# gender wise distribution of TARGET
application_train.groupby(["TARGET",'CODE_GENDER']).size()


# In[36]:


# no of unique SK_ID_CURR in bureau
len(set(bureau['SK_ID_CURR']))


# In[37]:


# no of unique SK_ID_CURR in application train
len(set(application_train['SK_ID_CURR']))


# In[38]:


# no of unique SK_ID_CURR in application test
len(set(application_test['SK_ID_CURR']))


# In[40]:


# no of SK_ID_CURR which are present both in bureau and application train
len(set(bureau['SK_ID_CURR']).intersection(set(application_train['SK_ID_CURR'])))


# In[ ]:


# no of SK_ID_CURR which are present both in bureau and application test
len(set(bureau['SK_ID_CURR']).intersection(set(application_test['SK_ID_CURR'])))


# In[44]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(application_train.corr(), annot=True)


# In[48]:


#columns with NaN in application train and thier percentages
a=pd.isnull(application_train).mean()
a[a!=0]


# In[49]:


#columns with NaN in application test and thier percentages
a=pd.isnull(application_test).mean()
a[a!=0]


# In[51]:


#outlier in REGION_POPULATION_RELATIVE
sns.boxplot("REGION_POPULATION_RELATIVE", data=application_train, palette="PRGn")


# In[ ]:





# In[ ]:




