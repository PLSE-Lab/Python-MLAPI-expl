#!/usr/bin/env python
# coding: utf-8

# # Naive Kernel for Magnetic Interaction Prediction
# This is a work-in-progress kernel.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import HuberRegressor
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Load data sets

# In[2]:


trainSet = pd.read_csv('../input/train.csv')
display(trainSet.head())


# In[3]:


testSet = pd.read_csv('../input/test.csv')
display(testSet.head())


# In[4]:


structures = pd.read_csv('../input/structures.csv')
display(structures.head())


# ### Atomic distance
# https://www.kaggle.com/inversion/atomic-distance-benchmark/

# In[5]:


# Map the atom structure data into train and test files

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)

testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)


# In[6]:


display(trainSet.head())
display(testSet.head())


# In[7]:


# https://www.kaggle.com/jazivxt/all-this-over-a-dog
# https://www.kaggle.com/artgor/molecular-properties-eda-and-models
train_p0 = trainSet[['x_0', 'y_0', 'z_0']].values
train_p1 = trainSet[['x_1', 'y_1', 'z_1']].values
test_p0 = testSet[['x_0', 'y_0', 'z_0']].values
test_p1 = testSet[['x_1', 'y_1', 'z_1']].values

trainSet['dist'] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet['dist'] = np.linalg.norm(test_p0 - test_p1, axis=1)

trainSet['dist_to_type_mean'] = trainSet['dist'] / trainSet.groupby('type')['dist'].transform('mean')
testSet['dist_to_type_mean'] = testSet['dist'] / testSet.groupby('type')['dist'].transform('mean')


# ### Atom types

# In[8]:


# All atom_0 are hydrogens
assert all(trainSet["atom_0"].astype('category').cat.categories == ['H'])
assert all(testSet["atom_0"].astype('category').cat.categories == ['H'])


# In[9]:


# atom_1 are carbon, hydrogen or nitrogen
print(trainSet["atom_1"].astype('category').cat.categories)
print(testSet["atom_1"].astype('category').cat.categories)


# In[10]:


# We use the interaction types, that already include the type of atoms involved
print(testSet["type"].astype('category').cat.categories)
print(trainSet["type"].astype('category').cat.categories)


# In[11]:


for i in trainSet["type"].astype('category').cat.categories.values:
    trainSet['type_'+str(i)] = (trainSet['type'] == i)
    testSet['type_'+str(i)] = (testSet['type'] == i)


# ## Huber regression
# Robust linear regression (tries to ignore outliers)

# In[12]:


model = HuberRegressor()


# In[13]:


# Features to include (regressors)
regressors = ['type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 'type_2JHN', 
                                       'type_3JHC', 'type_3JHH', 'dist', 'dist_to_type_mean']


# In[14]:


# Add bias, interaction term and quadratic and cubic terms
polyFeat = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)


# In[15]:


trainX = polyFeat.fit_transform(np.array(trainSet[regressors]))


# In[17]:


# Some features are uninformative:
# Interaction type features don't (statistically) interact as they are mutually exclusive
usefulFeatures = [i for i,x in enumerate(np.abs(np.sum(trainX, axis = 0))) if x > 0]
trainX = trainX[:,usefulFeatures]
trainX.shape


# In[18]:


# NB: no need to include type_3JHN as this is redundant: this is always true when all other types are false
fitDist = model.fit(trainX, 
                    trainSet['scalar_coupling_constant'])


# In[19]:


# Display factors to learn what is important for the prediction
fitDist.coef_


# ### Evaluate performance
# 

# In[22]:


# See https://www.kaggle.com/uberkinder/efficient-metric
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


# In[23]:


group_mean_log_mae(trainSet['scalar_coupling_constant'], 
                   model.predict(trainX), trainSet['type'])


# In[24]:


# Control: this should perform better than outputing the same overfitted value for all interactions
print(group_mean_log_mae(trainSet['scalar_coupling_constant'], trainSet['scalar_coupling_constant'].median(), trainSet['type']))
print(group_mean_log_mae(trainSet['scalar_coupling_constant'], 0.85, trainSet['type']))


# In[27]:


testX = polyFeat.transform(np.array(testSet[regressors]))[:,usefulFeatures]
resultSet = pd.DataFrame( { "id" : testSet['id'],
                            "scalar_coupling_constant" : model.predict(testX)} )


# ## Export results

# In[28]:


resultSet.to_csv("results.csv", index = False, header = True)


# In[29]:


# Check content of the output file
with open("results.csv", "r") as f:
    for i, line in enumerate(f):
        print(line)
        if i > 5:
            break

