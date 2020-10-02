# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# coding: utf-8

# In[17]:

get_ipython().magic('matplotlib inline')
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
from sklearn.preprocessing import Imputer


# In[63]:

dataset = pd.read_csv("../input/train.csv")
dataset.shape


# In[53]:

dataset.head()


# In[54]:

dataset.corr()
correlations = dataset.corr()
for x in correlations.columns:
    for y in correlations.columns:
        if x < y and correlations[x][y] > .8:
            print(x, y, correlations[x][y])
            #del dataset[x]
dataset.shape


# In[64]:

# FEATURE ENGINEERING!
train = dataset
train["HasAlley"] = 1 - pd.isnull(train["Alley"])
train["HasFireplace"] = 0 + train["Fireplaces"] > 0
train["HasBsmt"] = 1 - pd.isnull(train["BsmtQual"])
train["HasFence"] = 1 - pd.isnull(train["Fence"])
train["HasPool"] = 1 - pd.isnull(train["PoolQC"])

def quality(s):
    qual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, "Po": 1, 0: 0, "No": 3, "Mn": 3, "Av": 3}
    s = s.fillna(0)
    return [qual[e] for e in s]

train["ExterQual"] = quality(train["ExterQual"])
train["ExterCond"] = quality(train["ExterCond"])
train["BsmtQual"] = quality(train["BsmtQual"])
train["BsmtCond"] = quality(train["BsmtCond"])
train["BsmtExposure"] = quality(train["BsmtExposure"])
train["HeatingQC"] = quality(train["HeatingQC"])
train["KitchenQual"] = quality(train["KitchenQual"])
train["FireplaceQu"] = quality(train["FireplaceQu"])
train["GarageQual"] = quality(train["GarageQual"])
train["GarageCond"] = quality(train["GarageCond"])
train["PoolQC"] = quality(train["PoolQC"])

train["TotalBaths"]= train['FullBath'] + 0.5 * train['HalfBath']
# fills na values with means
train = train.fillna(train.groupby("Neighborhood").mean())
train = train.select_dtypes(include = ['float64', 'int64'])


# In[70]:

import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
# In order to train our classifier we need to split up the data into x and y
# X represents all the variables we are going to train with and y is the salePrice to predict
# We need to fill all na values with the mean of the column
# Before we can do this though we need to make sure that we only use numerical variables
# Therefore we should use the dummy function to replace all categorical with indicator 
# variables.
#we also need to drop one variable from each of the highly correlated pairs

X = train.drop(["YearRemodAdd", 'SalePrice', 'BsmtCond', 'BsmtExposure', 'TotalBsmtSF', 'FullBath', 'FireplaceQu', 'GarageArea', 'GarageCond', 'PoolQC', 'Id'], axis=1)
X = pd.get_dummies(X)
Y = train['SalePrice']
X = X.fillna(X.mean())

# We want to split up the data into a train and test set
# Therefore we split up X into a trainX and a testX and the same for Y
# 1) We will train the model on trainX and trainY
# 2) We will predict values using testX
# 3) We will compare these values against testY

local_trainX, local_testX = train_test_split(X, test_size=0.2, random_state=123)
local_trainY, local_testY = train_test_split(Y, test_size=0.2, random_state=123)

clf = sm.OLS(local_trainY, local_trainX)
result = clf.fit()


# In[71]:

result.summary()


# In[72]:

predictions = np.log(result.predict(local_testX) + 1)
local_testY = np.log(local_testY + 1)

# Mean squared error
error = np.sqrt(((predictions - local_testY) ** 2).mean())
error


# In[74]:

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
clf.fit(X, Y)


# In[119]:

import csv
def writecsv(predictions):
    n = 1461
    print("Id,SalePrice")
    init=[]
    init.append("Id")
    init.append("Saleprice")
    
    with open('persons.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(init)
        for p in predictions:
            RESULT =[]
            
            RESULT.append(n)
            RESULT.append(p)
            print(RESULT)
            filewriter.writerow(RESULT)

            n += 1
        csvfile


# In[121]:

test = pd.read_csv('../input/testh.csv')

# FEATURE ENGINEERING!
test["HasAlley"] = 1 - pd.isnull(test["Alley"])
test["HasFireplace"] = 0 + test["Fireplaces"] > 0
test["HasBsmt"] = 1 - pd.isnull(test["BsmtQual"])
test["HasFence"] = 1 - pd.isnull(test["Fence"])
test["HasPool"] = 1 - pd.isnull(test["PoolQC"])

def quality(s):
    qual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, "Po": 1, 0: 0, "No": 3, "Mn": 3, "Av": 3}
    s = s.fillna(0)
    return [qual[e] for e in s]

test["ExterQual"] = quality(test["ExterQual"])
test["ExterCond"] = quality(test["ExterCond"])
test["BsmtQual"] = quality(test["BsmtQual"])
test["BsmtCond"] = quality(test["BsmtCond"])
test["BsmtExposure"] = quality(test["BsmtExposure"])
test["HeatingQC"] = quality(test["HeatingQC"])
test["KitchenQual"] = quality(test["KitchenQual"])
test["FireplaceQu"] = quality(test["FireplaceQu"])
test["GarageQual"] = quality(test["GarageQual"])
test["GarageCond"] = quality(test["GarageCond"])
test["PoolQC"] = quality(test["PoolQC"])

test["TotalBaths"]= test['FullBath'] + 0.5 * test['HalfBath']

test = test.select_dtypes(include = ['float64', 'int64'])
test = test.drop(["YearRemodAdd", 'BsmtCond', 'BsmtExposure', 'TotalBsmtSF', 'FullBath', 'FireplaceQu', 'GarageArea', 'GarageCond', 'PoolQC', 'Id'], axis=1)
test = test.fillna(test.median())
preds = clf.predict(test)

writecsv(preds)
preds


# In[123]:

df = pd.read_csv("persons.csv")
df.shape


# In[ ]:



