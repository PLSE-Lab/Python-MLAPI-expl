#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


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


# **Data Ingestion**

# In[ ]:


ss = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")


# I want to know what are the main reasons of the variety of house price.

# * **EDA**
# * First,I check the type and missing value of the dataset.

# In[ ]:


train.info()


# In[ ]:


test.isnull().sum()


# I decide to drop the columns which contain too many missing value.

# In[ ]:


train = train.drop(train[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)
test = test.drop(test[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)
train


# I try to handle the column which contains little missing value.

# In[ ]:


train["LotFrontage"].value_counts().plot.bar()
plt.show()


# In[ ]:


train["LotFrontage"].describe()


# In[ ]:


train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace=True)


# In[ ]:


test["LotFrontage"].value_counts().plot.bar()
plt.show()


# In[ ]:


test["LotFrontage"].fillna(test["LotFrontage"].mean(),inplace=True)


# In[ ]:


train.info()


# Now I wanna see the breif correlation between features and SalePrice

# In[ ]:


plt.subplots(figsize=(20,20))
ax = plt.axes()
corr = train.corr()
sns.heatmap(corr)


# According to the heatmap,pick the features which have much bigger relationship and drop the features which have a really low correlation.

# In[ ]:


train1 = train[["SalePrice","OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]
train1.head()


# In[ ]:


test1 = test[["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]
test1.head()


# * Cause "MasVnrArea" and some features about Garage have bigger relationship than many other features, I wonder if the other features which are related to the MasVnr and Garage also have high correlation to the Saleprice. 
# * "OverallQual" has really big relationship with Salesprice, so I wonder if "quality" is a important feature such as "ExterQual"
# * In my opinion, a good feature should contain various value, and no value has a absolutly high proportion than other value.If a feature has contains a value which have a proportion of 90 percent, I will drop it.

# I add some possibly suitable feature in the new dataframe.

# In[ ]:


train1["MasVnrType"] = train["MasVnrType"]
test1["MasVnrType"] = train["MasVnrType"]


# In[ ]:


train1["ExterQual"] = train["ExterQual"]
test1["ExterQual"] = test["ExterQual"]


# In[ ]:


train1["Foundation"] = train["Foundation"]
test1["Foundation"] = test["Foundation"]


# In[ ]:


train1["GarageType"] = train["GarageType"]
test1["GarageType"] = test["GarageType"]


# In[ ]:


train1["GarageFinish"] = train["GarageFinish"]
test1["GarageFinish"] = test["GarageFinish"]


# Handle the missing value and change the type of the data.

# In[ ]:


train1.info()


# In[ ]:


train["MasVnrType"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["MasVnrType"] = train1["MasVnrType"].map({"None":1,"BrkFace":2,"Stone":3,"BrkCmn":4})
train1["MasVnrType"].fillna(method='ffill',inplace = True)
train1["MasVnrType"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["MasVnrArea"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["MasVnrArea"].describe()


# In[ ]:


train1["MasVnrArea"].fillna(method='ffill',inplace = True)


# In[ ]:


train1["GarageType"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["GarageType"] = train1["GarageType"].map({"Attchd":1,"Detchd":2,"BuiltIn":3,"Basment":4,"CarPort":5,"2Types":6})
train1["GarageType"].fillna(method='ffill',inplace = True)


# In[ ]:


train1["GarageType"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["GarageFinish"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["GarageFinish"] = train1["GarageFinish"].map({"Unf":1,"RFn":2,"Fin":3})
train1["GarageFinish"].fillna(method='ffill',inplace = True)
train1["GarageFinish"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["ExterQual"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["ExterQual"] = train1["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})
train1["ExterQual"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["Foundation"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1["Foundation"] = train1["Foundation"].map({"PConc":1,"CBlock":2,"BrkTil":3,"Slab":4,"Stone":5,"Wood":6})


# In[ ]:


train1["Foundation"].value_counts().plot.bar()
plt.show()


# In[ ]:


train1.info()


# In[ ]:


plt.subplots(figsize=(20,20))
ax = plt.axes()
corr = train1.corr()
sns.heatmap(corr)


# According to the heatmap,"ExterQual" is the only godd feature among the new features I just added.

# In[ ]:


traindata = train1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)
testdata = test1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)


# In[ ]:


traindata.head(10)


# Handle the testdata.

# In[ ]:


testdata.info()


# In[ ]:


testdata.head(10)


# In[ ]:


testdata["MasVnrArea"].fillna(method='ffill',inplace=True)


# In[ ]:


testdata.fillna(testdata.mean(),inplace=True)


# In[ ]:


testdata["ExterQual"].value_counts().plot.bar()
plt.show()


# In[ ]:


testdata["ExterQual"] = testdata["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})
testdata["ExterQual"].value_counts().plot.bar()
plt.show()


# In[ ]:


testdata.info()


# * **Modeling and Training**

# In[ ]:


x_train = traindata.iloc[:,1:]
y_train = traindata.iloc[:,0]
x_test = testdata
y_test = ss.iloc[:,1]


# In[ ]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0).fit(x_train, y_train)

#Returns the coefficient of determination R^2 of the prediction.
model.score(x_train, y_train)


# According to the score of the model,the features which are picked are nearly the main reasons of the saleprice of the house.

# **Some Questions**
# * I don't know why model.score(x_test,y_test) is a minus though the score of (x_train,y_train) is so high.
# * Why I can't cant the score(x_test,y_test) when I use RandomForestClassifier.

# In[ ]:


model.score(x_test, y_test)

