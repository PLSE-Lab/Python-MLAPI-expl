#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis - Home Credit

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# ## Reading all data into respective dataframes

# In[2]:


POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
application_train = pd.read_csv('../input/application_train.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
application_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')


# In[3]:


print('------------- DataSet Sizes ----------------')
print('POS_CASH_balance:', POS_CASH_balance.shape)
print('bureau_balance:', bureau_balance.shape)
print('application_train:', application_train.shape)
print('previous_application:', previous_application.shape)
print('installments_payments:', installments_payments.shape)
print('credit_card_balance:', credit_card_balance.shape)
print('application_test:', application_test.shape)
print('bureau:', bureau.shape)


# ## Getting Glimpse of data

# In[4]:


def data_glimpse(df):
    return pd.concat([df.head(3),df.tail(3)])


# ### POS_CASH_balance

# In[5]:


data_glimpse(POS_CASH_balance)


# ### bureau_balance

# In[6]:


data_glimpse(bureau_balance)


# ### application_train

# In[7]:


data_glimpse(application_train)


# ### previous_application

# In[8]:


data_glimpse(previous_application)


# ### installments_payments

# In[9]:


data_glimpse(installments_payments)


# ### credit_card_balance

# In[10]:


data_glimpse(credit_card_balance)


# ### bureau

# In[11]:


data_glimpse(bureau)


# ## Finding Missing Data

# In[12]:


def plot_miss(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['name', 'cnt']
    missing_df = missing_df[missing_df['cnt']>0]
    missing_df = missing_df.sort_values(by='cnt', ascending=False)
    ind = np.arange(missing_df.shape[0])
    width = 0.8
    fig, ax = plt.subplots(figsize=(12.5,17))
    rects = ax.barh(ind, missing_df.cnt.values, color='orange')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.name.values, rotation='horizontal')
    ax.set_xlabel("Missing values count")
    ax.set_title("Number of missing values")
    plt.show()


# In[13]:


def tbl_miss(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_df  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_df


# ### application_train

# In[14]:


plot_miss(application_train)


# ### previous_application

# In[15]:


plot_miss(previous_application)


# ### credit_card_balance

# In[16]:


plot_miss(credit_card_balance)


# ### bureau

# In[17]:


plot_miss(bureau)


# ### POS_CASH_balance

# In[18]:


tbl_miss(POS_CASH_balance)


# ### bureau_balance

# In[19]:


tbl_miss(bureau_balance)


# ### installments_payments

# In[20]:


tbl_miss(installments_payments)


# ## Data Distribution

# In[21]:


def plot_dist(col):
    for i in [(0,'g'),(1,'y')]:
        plt.figure(figsize=(12,5))
        plt.title("Distribution of "+ col +" wrt Target = "+str(i[0]))
        ax = sns.distplot(application_train[col][application_train.TARGET==i[0]].dropna(), color=i[1])


# ### AMT_CREDIT distribution wrt Target

# In[22]:


plot_dist('AMT_CREDIT')


# ### AMT_INCOME_TOTAL distribution wrt Target

# In[23]:


plot_dist('AMT_INCOME_TOTAL')


# ### AMT_GOODS_PRICE distribution wrt Target

# In[24]:


plot_dist('AMT_GOODS_PRICE')


# ### AMT_ANNUITY

# In[25]:


plot_dist('AMT_ANNUITY')


# ### DAYS_BIRTH

# In[26]:


plot_dist('DAYS_BIRTH')


# ### DAYS_EMPLOYED

# In[27]:


plot_dist('DAYS_EMPLOYED')


# ### DAYS_REGISTRATION

# In[28]:


plot_dist('DAYS_REGISTRATION')


# ### DAYS_ID_PUBLISH

# In[29]:


plot_dist('DAYS_ID_PUBLISH')


# ## Plotting Categorical Variables

# In[30]:


def plot_bar(col):
    df_tmp = application_train[col][application_train.TARGET==0].value_counts()
    df1 = pd.DataFrame({col: df_tmp.index,'Count TARGET=0': df_tmp.values})
    df_tmp = application_train[col][application_train.TARGET==1].value_counts()
    df2 = pd.DataFrame({col: df_tmp.index,'Count TARGET=1': df_tmp.values})
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,8))
    p1 = sns.barplot(ax=ax1, x = col, y="Count TARGET=0",data=df1)
    p1.set_xticklabels(p1.get_xticklabels(),rotation=90)
    p2 = sns.barplot(ax=ax2, x = col, y="Count TARGET=1",data=df2)
    p2.set_xticklabels(p2.get_xticklabels(),rotation=90)


# ### NAME_INCOME_TYPE

# In[31]:


plot_bar('NAME_INCOME_TYPE')


# ### OCCUPATION_TYPE

# In[32]:


plot_bar('OCCUPATION_TYPE')


# ### CNT_FAM_MEMBERS

# In[33]:


plot_bar('CNT_FAM_MEMBERS')


# ### CNT_CHILDREN

# In[34]:


plot_bar('CNT_CHILDREN')


# ### NAME_FAMILY_STATUS

# In[35]:


plot_bar('NAME_FAMILY_STATUS')


# ### FLAG_OWN_CAR

# In[36]:


plot_bar('FLAG_OWN_CAR')


# ### FLAG_OWN_REALTY

# In[37]:


plot_bar('FLAG_OWN_REALTY')


# ### CODE_GENDER

# In[38]:


plot_bar('CODE_GENDER')


# ### NAME_CONTRACT_TYPE

# In[39]:


plot_bar('NAME_CONTRACT_TYPE')


# ### ORGANIZATION_TYPE

# In[40]:


plot_bar('ORGANIZATION_TYPE')


# ### NAME_EDUCATION_TYPE

# In[41]:


plot_bar('NAME_EDUCATION_TYPE')


# ### NAME_HOUSING_TYPE

# In[42]:


plot_bar('NAME_HOUSING_TYPE')


# ### REG_REGION_NOT_LIVE_REGION

# In[43]:


plot_bar('REG_REGION_NOT_LIVE_REGION')


# ### REG_REGION_NOT_WORK_REGION

# In[44]:


plot_bar('REG_REGION_NOT_WORK_REGION')


# ### REG_CITY_NOT_LIVE_CITY

# In[45]:


plot_bar('REG_CITY_NOT_LIVE_CITY')


# ### REG_CITY_NOT_LIVE_CITY

# In[46]:


plot_bar('REG_CITY_NOT_LIVE_CITY')


# In[ ]:




