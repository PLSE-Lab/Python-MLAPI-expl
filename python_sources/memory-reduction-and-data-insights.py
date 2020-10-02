#!/usr/bin/env python
# coding: utf-8

# Here is my first kernel on KKBox.
# 
# I will be playing with only three of the csv files (train.csv, members.csv and transactions.csv). the file user_logs_1.csv is too big for these kernels.
# 
# What are we going to see in this kernel?
# * How to reduce memory consumption of dataframes?
# * Some insights into the data at hand.
# 
# ##Memory Reduction
# 
# The memory consumed by each of the above mentioned files are too high. I will first reduce it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.csv')
df_members = pd.read_csv('../input/members.csv')
df_transactions = pd.read_csv('../input/transactions.csv')
df_sample = pd.read_csv('../input/sample_submission_zero.csv')

#df_user_logs_1 = pd.read_csv('../input/user_logs.csv', chunksize = 500)
#df = pd.concat(df_user_logs_1, ignore_index=True)

# Any results you write to the current directory are saved as output.


# # Memory Reduction
# 
# **members.csv**
# 
# First we will see how to reduce this file to a managable size. 
# 
# Let us see the memory consumption of this dataframe

# In[ ]:


#--- Displays memory consumed by each column ---
print(df_members.memory_usage())

#--- Displays memory consumed by entire dataframe ---
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# The dataframe consumes ~273MB. Our aim is to REDUCE this to a minimum possible.

# In[ ]:


#--- Check whether it has any missing values ----
print(df_members.isnull().values.any())


# In[ ]:


#--- check which columns have Nan values ---
columns_with_Nan = df_members.columns[df_members.isnull().any()].tolist()
print(columns_with_Nan)


# In[ ]:


#--- Check the datatypes of each of the columns in the dataframe ---
print(df_members.dtypes)


# In[ ]:


print (df_members.head())


# Memory consumption can be reduced for columns having values of type *integer* or *float*.
# 
# First we have go through each column and find the **maximum** and **minimum** value and choose the appropriate datatype. [See this page for more info](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html).
# 

# In[ ]:


print(np.max(df_members['city']))
print(np.min(df_members['city']))


# In[ ]:


df_members['city'] = df_members['city'].astype(np.int8)


# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We have already reduced the dataframe size from 273MB to ~240MB. 
# 
# Hold tight we have many more columns to go!!!!

# In[ ]:


print(np.max(df_members['bd']))
print(np.min(df_members['bd']))


# In[ ]:


df_members['bd'] = df_members['bd'].astype(np.int16)


# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# Now the memory has reduced from 240MB to ~210MB.

# In[ ]:


print(np.max(df_members['registered_via']))
print(np.min(df_members['registered_via']))


# In[ ]:


df_members['registered_via'] = df_members['registered_via'].astype(np.int8)


# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We have further reduced the consumption to 175MB from 210MB!!!
# 
# Now we have two date columns which are **NOT** of type datetime but normal integers. We will have to split them based on *year*, *month*, and *date*.

# In[ ]:


df_members['registration_init_year'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[:4]))
df_members['registration_init_month'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
df_members['registration_init_date'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[-2:]))


# In[ ]:


df_members['expiration_date_year'] = df_members['expiration_date'].apply(lambda x: int(str(x)[:4]))
df_members['expiration_date_month'] = df_members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
df_members['expiration_date_date'] = df_members['expiration_date'].apply(lambda x: int(str(x)[-2:]))


# In[ ]:


print(df_members.head())


# The newly created columns are of type **int64** by default.

# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# You can see a surge in  memory consumption to ~410MB!!!
# 
# In a similar manner we have to check every **maximum** and **minimum** value in each of the newly created columns and assign the appropriate datatype.

# In[ ]:


df_members['registration_init_year'] = df_members['registration_init_year'].astype(np.int16)
df_members['registration_init_month'] = df_members['registration_init_month'].astype(np.int8)
df_members['registration_init_date'] = df_members['registration_init_date'].astype(np.int8)

df_members['expiration_date_year'] = df_members['expiration_date_year'].astype(np.int16)
df_members['expiration_date_month'] = df_members['expiration_date_month'].astype(np.int8)
df_members['expiration_date_date'] = df_members['expiration_date_date'].astype(np.int8)


# In[ ]:


#--- Now drop the unwanted date columns ---
df_members = df_members.drop('registration_init_time', 1)
df_members = df_members.drop('expiration_date', 1)


# In[ ]:


mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# VOILA!!! we have reduced the dataframe to 137MB from ~273MB; which is 50% decrease in memory usage!!!!
# 
# Now we can follow suit for the remaining two columns.

# **train.csv**

# In[ ]:


print(df_train.head())


# In[ ]:


print(df_train.isnull().values.any())


# In[ ]:


print(df_train.dtypes)


# In[ ]:


mem = df_train.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[ ]:


df_train['is_churn'] = df_train['is_churn'].astype(np.int8)

mem = df_train.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# The train.csv file has been reduced from 15MB to 8MB.
# 
# Let us now check the transactions.csv file
# 
# **transactions.csv**

# In[ ]:


print(df_transactions.head())


# In[ ]:


mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# This is consuming a whooping 1.4GB!!!!

# In[ ]:


print(df_transactions.isnull().values.any())


# In[ ]:


print(df_transactions.dtypes)


# In[ ]:



df_transactions['payment_method_id'] = df_transactions['payment_method_id'].astype(np.int8)
df_transactions['payment_plan_days'] = df_transactions['payment_plan_days'].astype(np.int16)
df_transactions['plan_list_price'] = df_transactions['plan_list_price'].astype(np.int16)
df_transactions['actual_amount_paid'] = df_transactions['actual_amount_paid'].astype(np.int16)
df_transactions['is_auto_renew'] = df_transactions['is_auto_renew'].astype(np.int8)
df_transactions['is_cancel'] = df_transactions['is_cancel'].astype(np.int8)


# Here we have two date columns as well of type integer. We proceed as before.

# In[ ]:



df_transactions['transaction_date_year'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[:4]))
df_transactions['transaction_date_month'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[4:6]))
df_transactions['transaction_date_date'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[-2:]))

df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))
df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))


# Now we assign different datatypes as appropriate to the newly created columns.

# In[ ]:



df_transactions['transaction_date_year'] = df_transactions['transaction_date_year'].astype(np.int16)
df_transactions['transaction_date_month'] = df_transactions['transaction_date_month'].astype(np.int8)
df_transactions['transaction_date_date'] = df_transactions['transaction_date_date'].astype(np.int8)

df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date_year'].astype(np.int16)
df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date_month'].astype(np.int8)
df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date_date'].astype(np.int8)


# In[ ]:


#--- Now drop the unwanted date columns ---
df_transactions = df_transactions.drop('transaction_date', 1)
df_transactions = df_transactions.drop('membership_expire_date', 1)


# In[ ]:


print(df_transactions.head())


# In[ ]:


mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We have reduced the transcations.csv file from 1.4GB to ~513 MB!!!!!!!

# In[ ]:


print('DONE!!')


# Now we can perform feature engineering and then work out various models!!!!
# 
# I am figuring out a way to use the *user_logs.csv* file though. If anyone has a way please do share it!!

# # Data Analysis
# 
# Let us look into the data now. Below I am merging the three dataframes baesd on column **msno**.

# In[ ]:


#print(df_members.head())
#print(df_train.head())

df_train_members = pd.merge(df_train, df_members, on='msno', how='inner')
df_merged = pd.merge(df_train_members, df_transactions, on='msno', how='inner')
print(df_merged.head())


# ## is_churn
# First and foremost let us analyze the output variable **is_churn**.

# In[ ]:


df_train_members.hist(column='is_churn')


# Roughly around 5000 people have churned the  website over time. Our aim is to predict when the remaining people will churn.

# In[ ]:


#--- Check whether new dataframe has any missing values ----
print(df_train_members.isnull().values.any())


# In[ ]:


#--- check which columns have Nan values ---
columns_with_Nan = df_train_members.columns[df_train_members.isnull().any()].tolist()
print(columns_with_Nan)


# In[ ]:


df_train_members['gender'].isnull().sum()


# We will see how to fill them later. Let us see the other variables.
# 
# ** is_churn vs gender**

# In[ ]:


churn_vs_gender = pd.crosstab(df_train_members['gender'], df_train_members['is_churn'])

churn_vs_gender_rate = churn_vs_gender.div(churn_vs_gender.sum(1).astype(float), axis=0) # normalize the value
churn_vs_gender_rate.plot(kind='barh', , stacked=True)


# ** is_churn vs registered_via **

# In[ ]:


churn_registered_via = pd.crosstab(df_train_members['registered_via'], df_train_members['is_churn'])

churn_vs_registered_via_rate = churn_registered_via.div(churn_registered_via.sum(1).astype(float), axis=0) # normalize the value
churn_vs_registered_via_rate.plot(kind='barh', stacked=True)


# ** is_churn vs city **

# In[ ]:


churn_vs_city = pd.crosstab(df_train_members['city'], df_train_members['is_churn'])

churn_vs_city_rate = churn_vs_city.div(churn_vs_city.sum(1).astype(float),  axis=0) # normalize the value
churn_vs_city_rate.plot(kind='bar', stacked=True)


# **is_churn vs bd (age)**

# In[ ]:


#eliminating extreme outliers
df_train_members = df_train_members[df_train_members['bd'] >= 1]
df_train_members = df_train_members[df_train_members['bd'] <= 80]

import seaborn as sns
sns.violinplot(x=df_train_members["is_churn"], y=df_train_members["bd"], data=df_train_members)


# ## **Variable: 'city'**

# In[ ]:


print (df_train_members['city'].unique())


# In[ ]:


data = df_train_members.groupby('city').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='city', y='msno', data=data)


# We see a high number of viewers/subscribers from city 1.
# 
# ## **Variable: 'bd'**

# In[ ]:


print (df_train_members['bd'].nunique())


# In[ ]:


df_train_members.plot(x=df_train_members.index, y='bd')


# As stated in the data file, we can clearly see some outliers:
# * There is an occurrence of -3000 on the negatve side.
# * And several occurrences on the positive side.
# 
# Since this column represents **age** such occurences must be removed
# 
# Based on these variations there can be some change in the way a model predicts for the test data. Moreover, we do not know whether such cases are present in the test set.
# 
# ## **Variable: 'registered_via' **

# In[ ]:


print (df_train_members['registered_via'].unique())


# There are only 5 ways of registration. Let us plot them against the count.

# In[ ]:


data = df_train_members.groupby('registered_via').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registered_via', y='msno', data=data)


# Most of the users registered through methods **7** and **8**.
# 
# ## **Variable 'registration_init_year' **
# 
# When did the users register to this site?

# In[ ]:


print (df_train_members['registration_init_year'].unique())


# In[ ]:


data = df_train_members.groupby('registration_init_year').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_year', y='msno', data=data)


# We can see a spike in registrations since the year 2010!
# 
# We can perform the same for **registration_init_month** and **registration_init_date**.

# In[ ]:


data = df_train_members.groupby('registration_init_month').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_month', y='msno', data=data)


# In[ ]:


data = df_train_members.groupby('registration_init_date').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='registration_init_date', y='msno', data=data)


# We are unable to visualize any form of trend in the above two plots.
# 
# ## **Variable: 'payment_method_id'**

# In[ ]:


data = df_merged.groupby('payment_method_id').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='payment_method_id', y='msno', data=data)


# A majorityof the people have opted payment method (41)
# 
# ## **Variable: 'payment_plan_days'**
# 
# To see distribution across different payment plans

# In[ ]:


from matplotlib import pyplot
data = df_merged.groupby('payment_plan_days').aggregate({'msno':'count'}).reset_index()
a4_dims = (11, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='payment_plan_days', y='msno', data=data)


# Most subscribers have planned for a month
# 
# ## **Variable: 'plan_list_price'**
# 

# In[ ]:


data = df_merged.groupby('plan_list_price').aggregate({'msno':'count'}).reset_index()
a4_dims = (20, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='plan_list_price', y='msno', data=data)


# ## **Variable: 'actual_amount_paid'**

# In[ ]:


data = df_merged.groupby('actual_amount_paid').aggregate({'msno':'count'}).reset_index()
a4_dims = (20, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.barplot(x='actual_amount_paid', y='msno', data=data)


# Both the above plotted graphs have a strong resemblence. Maybe we can retain one of these features for modelling and training.
# 
# ## **Variable: 'is_cancel'**

# In[ ]:


data = df_merged.groupby('is_cancel').aggregate({'msno':'count'}).reset_index()
ax = sns.barplot(x='is_cancel', y='msno', data=data)


# In[ ]:


#print(df_merged.columns)


# # Correlations
# 
# Finding high correlations among columns 

# In[ ]:


corr_matrix = df_merged.corr()
f, ax = plt.subplots(figsize=(20, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#--- For positive high correlation ---
high_corr_var = np.where(corr_matrix > 0.8)
high_corr_var = [(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr = []
for i in range(0,len(high_corr_var)):
    high_corr.append(high_corr_var[i][0])
    high_corr.append(high_corr_var[i][1])
high_corr = list(set(high_corr))

#--- For negative high corrlation ---
high_neg_corr_var = np.where(corr_matrix < -0.8)
high_neg_corr_var = [(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_neg_corr_var) if x!=y and x<y]
high_neg_corr = []
for i in range(0,len(high_neg_corr_var)):
    high_corr.append(high_neg_corr_var[i][0])
    high_corr.append(high_neg_corr_var[i][1])
high_neg_corr = list(set(high_neg_corr))  

#--- Merge both these lists avoiding duplicates ---
high_corr_list = list(set(high_corr + high_neg_corr))


# In[ ]:


print(high_corr_list)


# There are still more data insights to come on this kernel
# 
# **TUNE IN AGAIN !! **
