#!/usr/bin/env python
# coding: utf-8

# Good morning to everyone, I am new to this kind of things (data mining, machine learning, etc. ) and the first thing I did facing with this problem it was to analyse and to deal with missing values. 
# I developed a very simple technique, I decided to share it since I would like to improve it by putting together even your suggestion (I hope you will join). 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Loading training set
train = pd.read_csv('../input/train.csv')


# In[ ]:


# Loading testing set
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Print the first 5 samples of the training set
train.head(5)


# In[ ]:


# Print the first 5 samples of the testing set
test.head(5)


# In[ ]:


# Some information about training and testing set shape 
print ("Train shape: "+str(train.shape))
print ("Test shape: "+str(test.shape))
print ("Number of attributes: "+str(train.shape[1]))
print ("Number of observatons in the training set: "+str(train.shape[0]))
print ("Number of observatons in the testing set: "+str(test.shape[0]))


# In[ ]:


# Get the list of the columns with some missing values
attributes = train.columns.values
c = 0 
for attr in attributes:
    list_missing_values = train[attr][train[attr].isnull()]
    if(list_missing_values.empty==False):
        c += 1
        print (attr + " presents some missing values")

print ("\nIn the training set there are "+str(c)+" attributes (out of "+str(attributes.size)+") that presents at least one missing value")


# In[ ]:


# Value replacing
train['LotFrontage'].fillna(0,inplace=True)
train['Alley'].fillna('None',inplace=True)
train['FireplaceQu'].fillna('None',inplace=True)
train['PoolQC'].fillna('None',inplace=True)
train['Fence'].fillna('None',inplace=True)
train['MiscFeature'].fillna('None',inplace=True)
train['GarageCond'].fillna('No Garage',inplace=True)
train['GarageQual'].fillna('No Garage',inplace=True)
train['GarageFinish'].fillna('No Garage',inplace=True)
train['GarageYrBlt'].fillna('No Garage',inplace=True)
train['GarageType'].fillna('No Garage',inplace=True)
train['BsmtFinType1'].fillna('No Basement',inplace=True)
train['BsmtFinType2'].fillna('No Basement',inplace=True)
train['BsmtExposure'].fillna('No Basement',inplace=True)
train['BsmtQual'].fillna('No Basement',inplace=True)
train['BsmtCond'].fillna('No Basement',inplace=True)


# In[ ]:


# Row deletion: the following are the lines that present some missing values in attributes 'MasVnrType','MasVnrArea','Electrical' 
train = train[train.Id != 235]
train = train[train.Id != 530]
train = train[train.Id != 651]
train = train[train.Id != 937]
train = train[train.Id != 974]
train = train[train.Id != 978]
train = train[train.Id != 1244]
train = train[train.Id != 1279]
train = train[train.Id != 1380]


# In[ ]:


# Some information about the new training set
print ("New training set shape: "+str(train.shape))
print ("Number of attributes: "+str(train.shape[1]))
print ("Number of observatons in the new training set: "+str(train.shape[0]))


# In[ ]:


# Let's check if all the missing values have been correctly manipulated
attributes = train.columns.values
c = 0 
for attr in attributes:
    list_missing_values = train[attr][train[attr].isnull()]
    if(list_missing_values.empty==False):
        c += 1
        print (attr + " presents some missing values")

print ("\nIn the new training set there are "+str(c)+" attributes (out of "+str(attributes.size)+") that presents at least one missing value")


# In[ ]:


# Get the list of the columns with some missing values
attributes = test.columns.values
c = 0 
for attr in attributes:
    list_missing_values = test[attr][test[attr].isnull()]
    if(list_missing_values.empty==False):
        c += 1
        print (attr + " presents some missing values")

print ("\nThere are "+str(c)+" attributes (out of "+str(attributes.size)+") that presents at least one missing value")


# In[ ]:


# In the testing set I cannot delete any row
test['LotFrontage'].fillna(0,inplace=True)
test['Alley'].fillna('None',inplace=True)
test['FireplaceQu'].fillna('None',inplace=True)
test['PoolQC'].fillna('None',inplace=True)
test['Fence'].fillna('None',inplace=True)
test['MiscFeature'].fillna('None',inplace=True)
test['GarageCond'].fillna('No Garage',inplace=True)
test['GarageQual'].fillna('No Garage',inplace=True)
test['GarageFinish'].fillna('No Garage',inplace=True)
test['GarageYrBlt'].fillna('No Garage',inplace=True)
test['GarageType'].fillna('No Garage',inplace=True)
test['BsmtFinType1'].fillna('No Basement',inplace=True)
test['BsmtFinType2'].fillna('No Basement',inplace=True)
test['BsmtExposure'].fillna('No Basement',inplace=True)
test['BsmtExposure'].fillna('No Basement',inplace=True)
test['BsmtQual'].fillna('No Basement',inplace=True)
test['BsmtCond'].fillna('No Basement',inplace=True)
test['Utilities'].fillna('AllPub',inplace=True)
test['BsmtFinSF1'].fillna(0,inplace=True)
test['BsmtFinSF2'].fillna(0,inplace=True)
test['BsmtUnfSF'].fillna(0,inplace=True)
test['TotalBsmtSF'].fillna(0,inplace=True)
test['GarageCars'].fillna(0,inplace=True)
test['GarageArea'].fillna(0,inplace=True)


# In[ ]:


# Let's check if all the missing values have been correctly manipulated
attributes = test.columns.values
c = 0 
for attr in attributes:
    list_missing_values = test[attr][test[attr].isnull()]
    if(list_missing_values.empty==False):
        c += 1
        print (attr + " presents some missing values")

print ("\nIn the new testing set there are still "+str(c)+" attributes (out of "+str(attributes.size)+") that presents at least one missing value")


# In[ ]:


# The new training set and testing set are written to csv files
train.to_csv('train_no_missing_values.csv')
test.to_csv('test_no_missing_values.csv')


# Do you have any idea on how to deal with these remaining missing values that cannot be replaced with predefined values ? 
