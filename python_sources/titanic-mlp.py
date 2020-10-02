#!/usr/bin/env python
# coding: utf-8

# In[40]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


train.head()


# In[259]:


train[train["Cabin"].isin(["B58 B60"])]


# In[266]:


def cabin(t):
    if isinstance(t,str):
        return t[0]
    return "other"


# In[268]:


train["Cabin"] = train["Cabin"].apply(cabin)


# In[269]:


train["Cabin"].value_counts()


# In[270]:


feature = ["Pclass","Age","SibSp","Parch"]
label = "Survived"


# In[271]:


X_train = train[feature].copy()
Y = train[label]


# In[272]:


enc = preprocessing.OneHotEncoder()


# In[273]:


enc.fit(train["Sex"].values.reshape(-1,1))


# In[274]:


enc.transform(train["Sex"].values.reshape(-1,1)).toarray()


# In[275]:


ens = preprocessing.OneHotEncoder()


# In[276]:


ens.fit(train["Embarked"].fillna("S").values.reshape(-1,1))


# In[277]:


enb = preprocessing.OneHotEncoder()


# In[278]:


enb.fit(train["Cabin"].fillna("others").values.reshape(-1,1))


# In[279]:


enb.categories_[0]


# In[280]:



for cab in enb.categories_[0]:
    print(cab)
    X_train[cab] = 0


# In[281]:


enc.categories_


# In[282]:


X_train["male"] =0
X_train["female"]=0


# In[283]:


X_train[["male","female"]] = enc.transform(train["Sex"].values.reshape(-1,1)).toarray()


# In[284]:


X_train["S"] = 0
X_train["C"] = 0
X_train["Q"] = 0


# In[285]:


X_train[["S","C","Q"]]= ens.transform(train["Embarked"].fillna("S").values.reshape(-1,1)).toarray()


# In[286]:


X_train[enb.categories_[0]]=enb.transform(train["Cabin"].fillna("others").values.reshape(-1,1)).toarray()


# In[287]:


X_train.isnull().sum()


# In[288]:


X_train["Age"]=X_train["Age"].fillna(0)


# In[289]:


X_train.isnull().sum()


# In[290]:


Y.isnull().sum()


# In[312]:


mlp = MLPClassifier(hidden_layer_sizes=(25,),max_iter=500)


# In[313]:


mlp.fit(X_train,Y)


# In[314]:


mlp.score(X_train,Y)


# In[315]:


X_test = test[feature].copy()


# In[316]:


enc.transform(test["Sex"].values.reshape(-1,1)).toarray()


# In[317]:


ens.transform(test["Embarked"].fillna("S").values.reshape(-1,1)).toarray()


# In[318]:


test["Cabin"] = test["Cabin"].apply(cabin)


# In[319]:


enb.transform(test["Cabin"].fillna("others").values.reshape(-1,1)).toarray()


# In[300]:


X_test["male"] = 0
X_test["female"] = 0


# In[301]:


X_test[["male","female"]] = enc.transform(test["Sex"].values.reshape(-1,1)).toarray()


# In[302]:


X_test["Q"] = 0
X_test["S"] = 0
X_test["C"] = 0


# In[303]:


X_test[["Q","S","C"]] = ens.transform(test["Embarked"].fillna("S").values.reshape(-1,1)).toarray()


# In[304]:


for cab in enb.categories_[0]:
    print(cab)
    X_test[cab] = 0


# In[305]:


X_test.head()


# In[306]:


X_test.isnull().sum()


# In[307]:


X_test["Age"] = X_test["Age"].fillna(0)


# In[308]:


mlp.predict(X_test)


# In[309]:


test["Survived"] = mlp.predict(X_test)


# In[310]:


test.columns


# In[311]:


test[["PassengerId","Survived"]].to_csv("MLP titanic.csv",index=False)


# In[ ]:





# In[ ]:




