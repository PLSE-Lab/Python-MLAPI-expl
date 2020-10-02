#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

import random
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os
print(os.listdir("../input"))

import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[ ]:


# *** Read Data ***
df = pd.read_csv('../input/train.csv')
dfTest  = pd.read_csv('../input/test.csv')
dfSub   = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


# *** Data View ***
print(df.head())
print(dfSub.head())


# In[ ]:


# *** First Data Impression ***
print('Train Data Info')
print(df.info())
print('\nTest Data Info \n')
print(dfTest.info())
print('\nSubmission Data Info \n')
print(dfSub.info())


# In[ ]:


# *** Check Missing Values ***
print('Missing Values in Training Data')
print(df.isnull().sum())
print('Missing Values in Test Data')
print(df.isnull().sum())


# No Missing Values in either train or test data

# In[ ]:


# *** Check Unique Values ***
print('Unique Values in Training Data')
print(df.nunique())
print('Unique Values in Test Data')
print(dfTest.nunique())


# - 85,003 molecules in train data and 45,772 molecules in test data
# - 8 types of coupling in both training and test data

# In[ ]:


# *** Explore Trainin Data ***
# scalar_coupling_constant
df['scalar_coupling_constant'].describe(percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[ ]:


# Plot the Distribution
sns.boxplot(df['scalar_coupling_constant'])


# - 50% of the coupling constant values are in between -0.2 to 2.2
# - Some values even crosses 200

# In[ ]:


# Check the dot plot to make the distribution even more clear
sns.stripplot(x = df['scalar_coupling_constant'])


# In[ ]:


sns.distplot(df['scalar_coupling_constant'])


# - With this probability distribution graph, the actual data nature has been revealed completely
# - Apart from the two very obvious peak, there is a very small third peak at the end of dostribution around 200

# In[ ]:


# Distribution of scalar_coupling_constant by molecule_name
df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index().head()


# In[ ]:


df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()['scalar_coupling_constant'].         describe(percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[ ]:


sns.distplot(df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()['scalar_coupling_constant'])


# - Almost All the mean vaues of coupling constant for a molecule are in between 10 to 30
# - From the distribution table also, it is very much clear

# In[ ]:


# Lets Explore the Atom Index Combination in Detail
(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).nunique()


# - There are 513 index combination
# - Let's check the distribution of count

# In[ ]:


(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().nlargest(10) 


# - It is evident that the top 10 combinations are originationg from 9 to 12
# - Now I am interested in lookin the bottom ten also

# In[ ]:


(df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().nsmallest(10) 


# - It looks very rare for the same index pairs to come together
# > - Lets verify this 

# In[ ]:


df[df['atom_index_0'].map(str) == df['atom_index_1'].map(str)].shape


# - We can conclude that same pair of indexes never come together
# - Let's check the complete distribution of index pair count

# In[ ]:


# *** Distribution of Index 0 and 1
df['atom_index_0'].value_counts()


# In[ ]:


df['atom_index_0'].value_counts(normalize = True)


# - Top 12 index 0 accounts for 90% of the data

# In[ ]:


df['atom_index_1'].value_counts()


# In[ ]:


df['atom_index_1'].value_counts(normalize = True)


# - Top 12 index 1 accounts for 80% of the data
# - Top indexes by count varies in index 0 and index 1

# In[ ]:


countDist = (df['atom_index_0'].map(str) + '--' + df['atom_index_1'].map(str)).value_counts().reset_index().              rename(columns = {0 : 'combinationCount'})
sns.distplot(countDist['combinationCount'])


# In[ ]:


df['molecule_name'].value_counts().nlargest(10)


# Looks Like the counts doesn't vary much. let's check:

# In[ ]:


df['molecule_name'].value_counts().reset_index()['molecule_name'].                          describe(percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


# In[ ]:


ax = sns.boxplot(x=df['molecule_name'].value_counts().reset_index()['molecule_name'])


# - The median count is somewhere between 50 and 60.
# - From the above table it is 54.
# - Hardly a little more than 1% of molecules have occured more than 100 

# In[ ]:


# *** Simple Linear Regression Model ***
df['randomNum'] = df['id'].map(lambda x : random.uniform(0, 1))
dfTrain = df.query('randomNum <= 0.7')
dfValid  = df.query('randomNum > 0.7')
del dfTrain['randomNum']
del dfValid['randomNum']
del df['randomNum'] 


# - We will use simple risk based features

# In[ ]:


molecule_name_risk = dfTrain.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()
molecule_name_risk.rename(columns = {'scalar_coupling_constant' : 'molecule_name_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, molecule_name_risk, on = 'molecule_name')

index_0_risk = dfTrain.groupby('atom_index_0')['scalar_coupling_constant'].mean().reset_index()
index_0_risk.rename(columns = {'scalar_coupling_constant' : 'index_0_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, index_0_risk, on = 'atom_index_0')

index_1_risk = dfTrain.groupby('atom_index_1')['scalar_coupling_constant'].mean().reset_index()
index_1_risk.rename(columns = {'scalar_coupling_constant' : 'index_1_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, index_1_risk, on = 'atom_index_1')

type_risk = dfTrain.groupby('type')['scalar_coupling_constant'].mean().reset_index()
type_risk.rename(columns = {'scalar_coupling_constant' : 'type_risk'}, inplace = True)

dfTrain = pd.merge(dfTrain, type_risk, on = 'type')


# In[ ]:


dfTrain.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[ ]:


dfValid = pd.merge(dfValid, molecule_name_risk, on = 'molecule_name')
dfValid = pd.merge(dfValid, index_0_risk, on = 'atom_index_0')
dfValid = pd.merge(dfValid, index_1_risk, on = 'atom_index_1')
dfValid = pd.merge(dfValid, type_risk, on = 'type')


# In[ ]:


dfValid.isnull().sum()


# In[ ]:


print(dfValid.head())


# In[ ]:


dfValid.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[ ]:


molecule_name_risk.head()


# In[ ]:


print(dfTrain.head())


# In[ ]:


lmReg = LinearRegression()
lmReg.fit(dfTrain.drop(['id', 'scalar_coupling_constant'], axis = 1), dfTrain['scalar_coupling_constant'])
dfValid['scalar_coupling_constant_pred'] = lmReg.predict(dfValid.drop(['id', 'scalar_coupling_constant'], axis = 1))


# In[ ]:


dfValid['predDiff'] = dfValid['scalar_coupling_constant'] - dfValid['scalar_coupling_constant_pred'] 


# In[ ]:


dfValid['predDiff'].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# In[ ]:


dfValid['scalar_coupling_constant'].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# In[ ]:


# The coefficients
print('Coefficients: \n', lmReg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(dfValid['scalar_coupling_constant'], dfValid['scalar_coupling_constant_pred'] ))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(dfValid['scalar_coupling_constant'], dfValid['scalar_coupling_constant_pred']))


# In[ ]:


# Plot outputs
sns.boxplot(x="variable", y="value", data=pd.melt(dfValid[['scalar_coupling_constant', 'scalar_coupling_constant_pred']]))


# In[ ]:


ax = sns.lineplot(data = dfValid[['scalar_coupling_constant', 'scalar_coupling_constant_pred']])


# In[ ]:


#dfTest = pd.merge(dfTest, molecule_name_risk, on = 'molecule_name')
dfTest = pd.merge(dfTest, index_0_risk, on = 'atom_index_0')
dfTest = pd.merge(dfTest, index_1_risk, on = 'atom_index_1')
dfTest = pd.merge(dfTest, type_risk, on = 'type')
dfTest['molecule_name_risk'] = 0


# In[ ]:


dfTest['scalar_coupling_constant_pred'] = lmReg.predict(dfTest.drop(['id', 'molecule_name', 'atom_index_0', 'atom_index_1',                                                                     'type'], axis = 1))


# In[ ]:


dfSub['scalar_coupling_constant'] = dfTest['scalar_coupling_constant_pred']


# In[ ]:


dfSub.to_csv('sample_submission.csv', index = False)


# - Submission Score : 2.09
# - Build Model on Complete Training Data

# In[ ]:


molecule_name_risk = df.groupby('molecule_name')['scalar_coupling_constant'].mean().reset_index()
molecule_name_risk.rename(columns = {'scalar_coupling_constant' : 'molecule_name_risk'}, inplace = True)

df = pd.merge(df, molecule_name_risk, on = 'molecule_name')

index_0_risk = df.groupby('atom_index_0')['scalar_coupling_constant'].mean().reset_index()
index_0_risk.rename(columns = {'scalar_coupling_constant' : 'index_0_risk'}, inplace = True)

df = pd.merge(df, index_0_risk, on = 'atom_index_0')

index_1_risk = df.groupby('atom_index_1')['scalar_coupling_constant'].mean().reset_index()
index_1_risk.rename(columns = {'scalar_coupling_constant' : 'index_1_risk'}, inplace = True)

df = pd.merge(df, index_1_risk, on = 'atom_index_1')

type_risk = df.groupby('type')['scalar_coupling_constant'].mean().reset_index()
type_risk.rename(columns = {'scalar_coupling_constant' : 'type_risk'}, inplace = True)

df = pd.merge(df, type_risk, on = 'type')

df.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


dfTest  = pd.read_csv('../input/test.csv')
dfTest = pd.merge(dfTest, index_0_risk, on = 'atom_index_0')
dfTest = pd.merge(dfTest, index_1_risk, on = 'atom_index_1')
dfTest = pd.merge(dfTest, type_risk, on = 'type')
dfTest['molecule_name_risk'] = 0


# In[ ]:


dfTest.head()


# In[ ]:


dfTest.drop(['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type'], axis = 1).head()


# In[ ]:


df.head()


# In[ ]:


lmReg = LinearRegression()
lmReg.fit(df.drop(['id', 'scalar_coupling_constant'], axis = 1), df['scalar_coupling_constant'])
dfTest['scalar_coupling_constant_pred'] = lmReg.predict(dfTest.drop(['id', 'molecule_name',                                                         'atom_index_0', 'atom_index_1', 'type'], axis = 1))


# In[ ]:


dfSub['scalar_coupling_constant'] = dfTest['scalar_coupling_constant_pred']
dfSub.to_csv('sample_submission.csv', index = False)


# In[ ]:


dfSub.head()


# In[ ]:




