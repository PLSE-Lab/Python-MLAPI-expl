#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.describe()
train


# In[ ]:


#manupilate Target
y_train = train["SalePrice"]

#Drop date
x_train = train.loc[:,["MSSubClass","OverallQual","OverallCond","LotArea","LotFrontage",
             "YearBuilt","YearRemodAdd","1stFlrSF","2ndFlrSF","GrLivArea",
             "FullBath","HalfBath","BedroomAbvGr","TotRmsAbvGrd",
             "Fireplaces","WoodDeckSF","OpenPorchSF","EnclosedPorch",
            "MoSold","YrSold","MiscVal"]]



null_cols = [col for col in x_train.columns if x_train[col].isnull().sum() > 0]


print(null_cols)
print(x_train.shape)
plt.hist((x_train["LotFrontage"]))


# In[ ]:


plt.hist(np.log(x_train["LotFrontage"]))


# In[ ]:


def fill(data,column):
    mean_lotFrontage = np.mean(np.log(data[column]))
    data[column] = data[column].fillna(mean_lotFrontage)
    return data
x_train = fill(x_train,"LotFrontage")
x_train = fill(x_train,"LotArea")

null_cols = [col for col in x_train.columns if x_train[col].isnull().sum() > 0]
print(null_cols)


# In[ ]:


enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
onehot_train = enc.fit_transform(x_train)


# In[ ]:


plt.hist(y_train)


# In[ ]:


pca = PCA(n_components=2)
sc = StandardScaler()

x_train_pca = pca.fit_transform(onehot_train)
data_train_std = sc.fit_transform(x_train_pca)

# display(data_train_std)
# print(data_train_std.shape)
pd.DataFrame(data_train_std)


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(data_train_std,y_train)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=1000,random_state=0,n_jobs=1)
regr.fit(data_train_std,y_train)


# In[ ]:


test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
id = test["Id"]
test = test.loc[:,["MSSubClass","OverallQual","OverallCond","LotArea","LotFrontage",
             "YearBuilt","YearRemodAdd","1stFlrSF","2ndFlrSF","GrLivArea",
             "FullBath","HalfBath","BedroomAbvGr","TotRmsAbvGrd",
             "Fireplaces","WoodDeckSF","OpenPorchSF","EnclosedPorch",
            "MoSold","YrSold","MiscVal"]]

test = fill(test,"LotFrontage")
test = fill(test,"LotArea")

pca = PCA(n_components=2)
sc = StandardScaler()

test = enc.fit_transform(test)
test = sc.fit_transform(test)
x_test_pca = pca.fit_transform(test)
test = pd.DataFrame(x_test_pca)


# In[ ]:





# In[ ]:


clf_predict = clf.predict(test)


# In[ ]:


regr_predict = regr.predict(test)


# plt.hist(clf.predict(test))
# 

# In[ ]:


plt.hist(regr.predict(test))


# In[ ]:


output = pd.DataFrame({'Id': id, 'SalePrice': regr_predict})

output.to_csv("submission.csv", index=False)


# In[ ]:




