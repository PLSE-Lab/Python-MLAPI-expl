#!/usr/bin/env python
# coding: utf-8

# ## Brief Introduction
# ____
# The key objective of this notebook is to provide a simple guide to impute most variables in application_train dataset. After this process, I also used PCA to deal with only a few variables. Turns out that only four principal components are necessary to explain almost 100% of the data's variance. This notebook is intended to help kagglers use other algorithms that can't deal well with missing values. I tried to make the imputation as intuitive and correct as possible, but if someone disagree with the values I chose and/or  have a better value to impute, please feel free to comment and help me improve this notebook!

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

app_train = pd.read_csv('../input/application_train.csv')
# application_test= pd.read_csv('../input/application_test.csv')
# bureau = pd.read_csv('../input/bureau.csv')
# bureau_balance = pd.read_csv('../input/bureau_balance.csv')
# POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
# credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
# previous_application = pd.read_csv('../input/previous_application.csv')
# installments_payments = pd.read_csv('../input/installments_payments.csv')


# ## Dealing with missing values

# In[2]:


null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr.head()


# There are several variables with "MODE", "AVG" or "MEDI" on it. In most cases, the number of missing values is exactly for all statistics belonging to a given feature. For instance, all variables with name "COMMONAREA" share the same number of null values. I think it is fairly safe to assume that those missing values actually represent the lack of that characteristic, so all clients with COMMONAREA features equal to zero probably don't live in a place that has a common area. Therefore I'll assign each and everyone of them the value of zero. 

# In[3]:


for variable in null_values_apptr["variable"]:
    if (variable.endswith("MEDI")|variable.endswith("MODE")|variable.endswith("AVG")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)


# I'll do the same for all variables related to Credit Bureau. As they are "Number of enquiries to Credit Bureau about the client \*some predefined period of time\*  before application", I assume that null quantities related to those variables are actually 0, meaning that no enquiries were made for that client.

# In[4]:


for variable in null_values_apptr["variable"]:
    if (variable.startswith("AMT_REQ_CREDIT_BUREAU")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)


# The "Social Circle" variables are a kind of mystery to me, but as they are a "number of observations" according to the columns dictionary, I feel that 0 is an appropriate value to impute as well, because no observations of client's social circle were made.

# In[5]:


for variable in null_values_apptr["variable"]:
    if (variable.endswith("SOCIAL_CIRCLE")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)


# Let's check for the remaining variables with missing values

# In[6]:


#checking for remaining nulls:
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
#percentage of missing values on a given column
null_values_apptr["pct_missing"] = null_values_apptr.n_missing/len(app_train)
null_values_apptr


# There is a lot less variables we have to check one by one. 
# ## OWN_CAR_AGE
# ___

# In[7]:


sns.kdeplot(app_train.OWN_CAR_AGE);


# This variable is a bit tricky. Here there are two things that I can think about that null values may represent:
# - null values mean that the client doesn't own a car
# - null values mean that it is just missing information
# - a mix of the two
# 
# I'll assume the first option, as 65% of missing data doesn't seem to indicate a failure in collecting the data. Maybe a small percentage of those NaN are really missing information, but I'll assume it will be a really small percentage. For customers without a car, there doesn't seem to be a reasonable and logical value to impute the car's age. To solve this, let's create a variable "own_car" and assign 1 to anyone who has a car and 0 otherwise.

# In[7]:


app_train["OWN_CAR"] = 0
app_train.loc[app_train.OWN_CAR_AGE >= 0, "OWN_CAR"] = 1
app_train.loc[:,("OWN_CAR", "OWN_CAR_AGE")].head()

#dropping car age column
app_train = app_train.drop(columns=["OWN_CAR_AGE"])


# ### EXT SCORE variables
# ___
# These variables are "normalized scores from external data sources".  So although the clients with missing values probably weren't scored, each customer should have a "ideal score", at least in theory. So we can't give them score 0 right away. Rather, let's see the distribution of all EXT SCOREs and impute case by case.

# In[38]:


sns.kdeplot(app_train.EXT_SOURCE_1);
sns.kdeplot(app_train.EXT_SOURCE_2);
sns.kdeplot(app_train.EXT_SOURCE_3);


# Source 1 has too much missing values, so imputation could potentially bias this variable. I prefer not to use it at all unless the algorithm can deal with null values by itself, like LightGBM. Source 2 has a smaller number of nulls and maybe using a ML algorithm would be the best shot to impute it, as imputing it with either mean or median values create a huge deformation on the distribution. For this version, I'll just remove it and mabe come back at a later time to deal with it properly. 
# At last, Source 3 has just a small percentage of missing values and imputing it with the median value seems to be the best.

# In[8]:


app_train = app_train.drop(columns= ["EXT_SOURCE_1"])
app_train = app_train.drop(columns= ["EXT_SOURCE_3"])
app_train.loc[:,"EXT_SOURCE_2"] = app_train.loc[:,"EXT_SOURCE_2"].fillna(app_train.EXT_SOURCE_2.median())
# app_train.loc[:,"EXT_SOURCE_3"] = app_train.loc[:,"EXT_SOURCE_3"].fillna(app_train.EXT_SOURCE_3.mode()[0])
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr


# In[10]:


sns.kdeplot(app_train.EXT_SOURCE_2);
# sns.kdeplot(app_train.EXT_SOURCE_3);


# The distribution remains almost the same, so it seems the median was a good choice indeed.
# 
# ## OCCUPATION_TYPE
# _____
# This variable will be dealt a bit differently than the others. Inspired by [ISR's kernel](https://www.kaggle.com/isr1512/home-credit-expanded-detailed-data-analysis), I'll use the education level of each person with a missing value in Occupation.

# In[9]:


app_train.NAME_EDUCATION_TYPE.unique()
sns.heatmap(pd.crosstab(app_train.OCCUPATION_TYPE, app_train.NAME_EDUCATION_TYPE), cmap="Blues");


# In[11]:


for education in app_train.NAME_EDUCATION_TYPE.unique():
    mode_to_impute = app_train[app_train.NAME_EDUCATION_TYPE == education].OCCUPATION_TYPE.mode()[0]
    app_train.loc[app_train.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"] = app_train.loc[app_train.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"].fillna(mode_to_impute)


# In[12]:


null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr


# ## NAME_TYPE_SUITE
# ___

# In[13]:


app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.astype("category")
sns.countplot(y = app_train.NAME_TYPE_SUITE)


# By far, most people are unaccompanied, so we will use this value to impute NAME_TYPE_SUITE.

# In[14]:


app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna(app_train.NAME_TYPE_SUITE.mode()[0])

null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr


# ## CNT_FAM_MEMBERS

# In[15]:


sns.countplot(app_train.CNT_FAM_MEMBERS)


# There are clearly more clients with two people in the family. Therefore I will  use this value to replace the nulls.

# In[15]:


app_train.CNT_FAM_MEMBERS = app_train.CNT_FAM_MEMBERS.fillna(app_train.CNT_FAM_MEMBERS.mode()[0])


#    ### DAYS_LAST_PHONE_CHANGE

# In[24]:


sns.kdeplot(app_train.DAYS_LAST_PHONE_CHANGE)


# The distribution is somewhat skewed, so I'll make the imputation using the mode. 

# In[16]:


app_train.DAYS_LAST_PHONE_CHANGE = app_train.DAYS_LAST_PHONE_CHANGE.fillna(app_train.DAYS_LAST_PHONE_CHANGE.mode()[0])


# ## AMT_ANNUITY

# In[47]:


sns.kdeplot(app_train.AMT_ANNUITY)


# The median value seems to be the best imputation value on this case, as the distribution is gratly skewed towards the left part.

# In[17]:


app_train.AMT_ANNUITY = app_train.AMT_ANNUITY.fillna(app_train.AMT_ANNUITY.median())


# ### AMT_GOODS_PRICE

# In[49]:


sns.distplot(app_train.AMT_GOODS_PRICE[pd.notnull(app_train.AMT_GOODS_PRICE)])


# This one is a bit tricky. There are several peaks along the distribution. Let's impute using the mode and see if the distribution is still about the same.

# In[18]:


app_train.AMT_GOODS_PRICE = app_train.AMT_GOODS_PRICE.fillna(app_train.AMT_GOODS_PRICE.mode()[0])
sns.distplot(app_train.AMT_GOODS_PRICE)


# Seems to be alright, as there were just a few missing values compared to the total number. Now  we have finished our work dealing with the missing data.

# In[19]:


app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna(app_train.NAME_TYPE_SUITE.mode()[0])

null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr


# ## Principal Component Analysis
# ____
# Now that our data doesn't have missing values, I'm going to use a PCA to reduce the number of variables we have to deal with.

# In[20]:


######################################
############### PCA ##################
######################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = app_train.copy()
df = df.drop(columns = ["SK_ID_CURR", "TARGET"])
df = pd.get_dummies(df)

#scaling the components
ss = StandardScaler()
df = ss.fit_transform(df)

pca = PCA(random_state = 0)
df_decomposed = pca.fit(df)
df = pca.transform(df)


# In[21]:


cumulative_var = []
cumul_var = 0
for exvar in df_decomposed.explained_variance_ratio_:
    cumul_var = exvar + cumul_var
    cumulative_var.append(cumul_var)


# In[23]:


plt.plot(cumulative_var, label = "cumulative explained variance")
plt.plot(df_decomposed.explained_variance_ratio_, label = "individual explained variance")
plt.legend();


# Looking at this plot, we can see that if we want almost 100% of variance explained, we need about 170 PCs. We can further reduce this number if only around 80% of explained variance is wanted. In this case we would need "only" 130 PCs.
# 
# Now the next step is to train some models and see how they perform when compared to the LB scores. There is a lot of room for improvement by feature engineering and imputation using ML algorithms, so feel free to take this as a basic pipeline to make your own imputation/PCA analysis or just use it as it is. Hope y'all enjoy it :) 

# 
