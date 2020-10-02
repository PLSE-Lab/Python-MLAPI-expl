#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


train_df.columns.groupby(train_df.dtypes)


# Train and test sets have 9557 and 23856 records respectively, which means test set has almost 2.5 times more data. There are 143 columns in train set, which includes Target and ID, so there are 141 features available for training, from which 8 are float, 129 are int and 4 are object. 
# 
# Let's take a quick look at the columns with null values in both train and test sets:

# In[ ]:


def count_nulls(df):
    null_counter = df.isnull().sum(axis=0)
    null_counter = null_counter[null_counter > 0]
    null_percent = df.isnull().sum(axis=0) / df.shape[0] * 100
    null_percent = null_percent[null_percent > 0]
    null_df = pd.concat([null_counter,null_percent],axis=1)
    null_df.columns = ['count','percent']
    display(null_df)


# In[ ]:


count_nulls(train_df)


# In[ ]:


count_nulls(test_df)


# Looks like train and test sets have the same columns with missing values with almost the same percent! Let's exclude the three columns that have > 70% missing values:

# In[ ]:


exclude_cols = ['Id','Target','v2a1','v18q1','rez_esc']


# Let's take a look at the Target values:

# In[ ]:


np.unique(train_df['Target'])


# In[ ]:


train_df['Target'].value_counts()


# In[ ]:


plt.hist(train_df['Target'])
plt.show()


# As we expected, there are only four values in the target values where 1 means extreme poverty and 4 means non-vulnerable families, and 2 and 3 are in between. The majority (~60%) are non-vulnerable. 
# 
# Let's take a look at the columns with dtype='O' (object) to prepare them for training. These are columns that have mixed types.

# In[ ]:


print([x for x in train_df.columns if train_df[x].dtype=='O'])


# 'Id' is a unique identifier of each row, which we are going to exclude, and 'idhogar' is the unique identifier of each household. 

# In[ ]:


train_df.idhogar = train_df.idhogar.astype('category')
test_df.idhogar = test_df.idhogar.astype('category')


# In[ ]:


train_df.dependency.value_counts()


# We have from the Column descriptions:
# 
# dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# 
# It is pretty strange to have yes/no values for a rate that can be calculated. Looks like we can calculate this rate ourselves from the data using these columns:
# 
# hogar_nin, Number of children 0 to 19 in household
# 
# hogar_adul, Number of adults in household
# 
# hogar_mayor, # of individuals 65+ in the household
# 
# hogar_total, # of total individuals in the household

# In[ ]:


train_df['dependency_calculated'] = (train_df.hogar_nin + train_df.hogar_mayor)/(train_df.hogar_adul - train_df.hogar_mayor)


# In[ ]:


train_df[['dependency','dependency_calculated']]


# Looks like for some reason, 1 is replaced with 'yes', 0 with 'no' and 'inf' (when there is no adult between 19 and 64) with 8! Let's try these and see:

# In[ ]:


train_df.dependency.replace('no','0',inplace=True)
train_df.dependency.replace('yes','1',inplace=True)
train_df.dependency_calculated.replace(float('inf'),8,inplace=True)


# In[ ]:


all(np.isclose(train_df.dependency.astype('float'), train_df.dependency_calculated))


# Looks like it was a correct guess. This was fun, but probably not very important as far as training a model is concerned! :) 
# 
# Let's update the type and look at other object types.

# In[ ]:


test_df.dependency.replace('no','0',inplace=True)
test_df.dependency.replace('yes','1',inplace=True)
train_df.dependency = train_df.dependency.astype('float')
test_df.dependency = test_df.dependency.astype('float')
train_df.drop('dependency_calculated', axis=1, inplace=True)


# edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, **yes=1 and no=0**
# 
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, **yes=1 and no=0**

# In[ ]:


train_df.edjefe.value_counts()


# There are some numbers that show the years of education of male/female head of household and some yes/no values that need to be replace by 1 and 0 respectively. 

# In[ ]:


train_df.edjefe.replace('no','0',inplace=True)
train_df.edjefe.replace('yes','1',inplace=True)
train_df.edjefe = train_df.edjefe.astype('float')
test_df.edjefe.replace('no','0',inplace=True)
test_df.edjefe.replace('yes','1',inplace=True)
test_df.edjefe = test_df.edjefe.astype('float')

train_df.edjefa.replace('no','0',inplace=True)
train_df.edjefa.replace('yes','1',inplace=True)
train_df.edjefa = train_df.edjefa.astype('float')
test_df.edjefa.replace('no','0',inplace=True)
test_df.edjefa.replace('yes','1',inplace=True)
test_df.edjefa = test_df.edjefa.astype('float')


# In[ ]:


exclude_cols


# In[ ]:


use_cols = train_df.columns.difference(exclude_cols)
print(len(use_cols))
print(use_cols)


# In[ ]:


for x in use_cols.difference(['idhogar']):
    train_df[x] = train_df[x].astype('float')
    test_df[x] = test_df[x].astype('float')


# In[ ]:


train_df[use_cols].dtypes.value_counts()


# In[ ]:


test_df[use_cols].dtypes.value_counts()

