#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(6,5)});
plt.figure(figsize=(6,5));

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
print(data.shape)
data.head()


# In[ ]:


# Checking inbalance
data.isFraud.value_counts()


# In[ ]:


p = sns.countplot(data=data, x='isFraud')
#p.set(ylim = (-10000, 500000))

plt.ylabel('Count')


# In[ ]:


data.isFraud.value_counts(normalize=True)*100


# In[ ]:


data[data.step > 718].isFraud.value_counts()


# Zero values are not available after 718 steps so we will remove these rows from the dataset

# In[ ]:


data = data[data.step <= 718]
print(data.shape)


# In[ ]:


# Checking inbalance
data.isFraud.value_counts()


# In[ ]:


data.type.value_counts()


# In[ ]:


# Fraud occurs only among 2 type of transactions
data.groupby('type')['isFraud'].sum()


# isFlaggedFraud is irrelevant

# In[ ]:


data.groupby('type')['isFlaggedFraud'].sum()


# In[ ]:


data[data.isFlaggedFraud == 1]


# In[ ]:


# Missing Values
data.isnull().values.any()


# In[ ]:


data[(data.step > 50) & (data.step < 90)].isFraud.value_counts()


# In[ ]:


data[data.isFraud == 1].shape


# In[ ]:


data[data.isFraud == 1].nameDest.value_counts()


# In[ ]:


data[data.nameDest == 'C1259079602']


# In[ ]:


data.describe()


# In[ ]:


data[data.amount > 1500000].shape


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='amount')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='amount')
p.set(ylim = (0, 4000000))


# In[ ]:


sns.boxplot(data=data[data.amount < 1500000], x='isFraud', y='amount')


# In[ ]:


data[data.isFraud == 0].amount.mean()


# In[ ]:


data[data.isFraud == 0].amount.describe()


# In[ ]:


data[data.isFraud == 1].amount.describe()


# In[ ]:


data[(data.amount < 1) & (data.isFraud == 0)]


# In[ ]:


data[(data.amount == 0) & (data.isFraud == 1)]


# In[ ]:


set(data[data.isFraud == 1].nameOrig).intersection(set(data[data.isFraud == 0].nameDest.unique()))


# In[ ]:


data[data.isFraud == 0].nameDest.sort_values()


# In[ ]:


data[data.isFraud == 1].nameOrig.sort_values()


# In[ ]:


data[data.nameOrig == 'C1510987794']


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='oldbalanceOrg')
p.set(ylim = (0, 4000000))
plt.ylabel('Opening Balance')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')
#p.set(ylim = (-10000, 500000))
p.set(ylim = (-100000, 4000000))
plt.ylabel('Origin Closing Balance')


# In[ ]:


data[data.isFraud == 1].oldbalanceOrg.describe()


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='oldbalanceDest')
p.set(ylim = (-10000, 1500000))
plt.ylabel('Destination Opening Balance')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='newbalanceDest')
p.set(ylim = (-100000, 1500000))
plt.ylabel('Destination Closing Balance')


# In[ ]:


data[data.isFraud == 1].amount.hist(bins=30)


# In[ ]:


data.head()


# In[ ]:


data['amount'] = np.log1p(data['amount'])
data['oldbalanceOrg'] = np.log1p(data['oldbalanceOrg'])
data['newbalanceOrig'] = np.log1p(data['newbalanceOrig'])
data['oldbalanceDest'] = np.log1p(data['oldbalanceDest'])
data['newbalanceDest'] = np.log1p(data['newbalanceDest'])
data.head()


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='amount')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='oldbalanceOrg')
plt.ylabel('Opening Balance')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='newbalanceOrig')
plt.ylabel('Origin Closing Balance')


# In[ ]:


data.head()


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='oldbalanceDest')
plt.ylabel('Destination Opening Balance')


# In[ ]:


p = sns.boxplot(data=data, x='isFraud', y='newbalanceDest')
plt.ylabel('Destination Closing Balance')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.step.value_counts()


# In[ ]:


data.tail()


# In[ ]:


data[(data.step > 700) & (data.step < 710)].isFraud.value_counts()


# In[ ]:


data[(data.step == 718)].isFraud.value_counts()


# In[ ]:


718/24


# In[ ]:




