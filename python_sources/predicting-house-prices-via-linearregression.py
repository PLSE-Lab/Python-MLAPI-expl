#!/usr/bin/env python
# coding: utf-8

# To do linear prediction :-
# 
#  - Check the data
#  - Find the usefull data
#  - Clean the data both train and test
#  - fit to the model 
#  - Do prediction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load the data train and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#importing the liberary of linearregression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#library for ploting the graphs
import matplotlib.pyplot as plt


# In[ ]:


#finding the numerical values which is usefull
train.head()
train.describe()


# In[ ]:


#cleaning the data
train["LotFrontage"].isnull().sum()
train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())


# In[ ]:


#claeaning the data
train["MasVnrArea"].isnull().sum()
train["MasVnrArea"] = train["MasVnrArea"].fillna(train["LotFrontage"].median())
train["MasVnrArea"].isnull().sum()


# In[ ]:


#cleanning the data
train["SalePrice"].isnull().sum()


# In[ ]:


#making the useful data set
new_columns=["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"]
x_train = train[new_columns] 
y_train = train["SalePrice"]


# In[ ]:


train.columns


# In[ ]:


#checking the numeric data
train._get_numeric_data()


# In[ ]:


#calling the linear regression model
reg = linear_model.LinearRegression()


# In[ ]:



#fit the data to the model
reg.fit(x_train,y_train)


# In[ ]:


#checking the coffecient
reg.coef_


# In[ ]:





# In[ ]:


#now taking the test data
x_test = test[new_columns]
test.columns


# In[ ]:


#checking the numerical column
test.describe()


# In[ ]:


#cleaning the test data
test["LotFrontage"].isnull().sum()
test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].median())
test["LotFrontage"].isnull().sum()


# In[ ]:


#cleaning the test data
test["MasVnrArea"].isnull().sum()

test["MasVnrArea"] = test["MasVnrArea"].fillna(test["MasVnrArea"].median())
test["MasVnrArea"].isnull().sum()


# In[ ]:


#cleaning the test data
test["BsmtFinSF1"].isnull().sum()

test["BsmtFinSF1"] = test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].median())
test["BsmtFinSF1"].isnull().sum()


# In[ ]:


#cleaning the test data
test["GarageArea"].isnull().sum()
test["GarageArea"] = test["GarageArea"].fillna(test["GarageArea"].median())
test["GarageArea"].isnull().sum()


# In[ ]:


#cleaning the test data
test["YrSold"].isnull().sum()


# In[ ]:



#now making the datframe of cleaned columns
x_test = test[new_columns]


# In[ ]:


#doing the linear predictions
a=reg.predict(x_test)


# In[ ]:



#value of the arrary (Prediction)
a


# In[ ]:


#intercept value
reg.intercept_


# In[ ]:


rf = RandomForestRegressor(n_estimators=3500,criterion='mse',max_leaf_nodes=3000,max_features='auto',oob_score=True)


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


rf.score(x_train,y_train)


# In[ ]:


b=rf.predict(x_test)


# In[ ]:


id_col = test["Id"]
sale = ["SalePrice"]
newDf = pd.DataFrame({"SalePrice":sale} )
#submit = pd.DataFrame()
#submit = submit.append(id_col)
#submit = submit.append(a )


# In[ ]:


pd.to_numeric(submission["SalePrice"])
pd.to_numeric(submission["Id"])


# In[ ]:


submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": b
    })

submission.to_csv('Submission.csv',index=False)


# In[ ]:


y_out = pd.read_csv("../input/sample_submission.csv")
y_test_out = np.array(y_out["SalePrice"])


# In[ ]:


#mean square error
sc = np.mean((a-y_test_out)**2)
#variance
score = reg.score(x_test,y_test_out)


# In[ ]:


#printing the mean square and variance
print(sc)
print(score)


# In[ ]:




