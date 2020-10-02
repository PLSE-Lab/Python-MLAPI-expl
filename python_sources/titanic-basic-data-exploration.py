#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print('train', train_df.shape)
print('test', test_df.shape)


# ## Total Passengers

# In[ ]:


print(train_df.shape[0] + test_df.shape[0])


# In[ ]:


train_df.info()


# In[ ]:


train_df.head()


# ## Data Dictionary
# - **PassengerId**
# - **Survived** - Survival	0 = No, 1 = Yes
# - **Pclass** - Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# - **Name** - Name
# - **Sex** -	Sex	
# - **Age** -	Age in years	
# - **SibSp** - # of siblings / spouses aboard the Titanic	
# - **Parch** - # of parents / children aboard the Titanic	
# - **Ticket** - Ticket number	
# - **Fare** - Passenger fare	
# - **Cabin** - Cabin number	
# - **Embarked** - Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# ## Passenger Counts per Pclass

# In[ ]:


print(train_df.groupby(['Pclass'])['Survived'].value_counts())


# ## Pclass & Sex => Survived

# In[ ]:


print(train_df.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize='True'))


# ## Train vs Test

# In[ ]:


describe_fields = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
print('\nTrain males:')
print(train_df[train_df.Sex == 'male'][describe_fields].describe())

print('\nTest males:')
print(test_df[test_df.Sex == 'male'][describe_fields].describe())

print('\nTrain females:')
print(train_df[train_df.Sex == 'female'][describe_fields].describe())

print('\nTest females:')
print(test_df[test_df.Sex == 'female'][describe_fields].describe())

