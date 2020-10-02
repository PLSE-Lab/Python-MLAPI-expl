#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.tree as tree
import sklearn.ensemble as ensem
from sklearn.model_selection import train_test_split


# In[ ]:


identity = pd.read_csv("../input/train_identity.csv",header=0)
transaction = pd.read_csv("../input/train_transaction.csv",header=0)


# In[ ]:


tempData = transaction[["TransactionAmt","ProductCD","card4","isFraud"]]


# I have just taken four columns 
# 
# * TransactionAmt
# * ProductCD
# * card4 
# * isFraud
# 
# I want to see the relationship of first three that is TransactionAmt, ProductCD, card4 with isFraud and also want to **fit** Decesion tree and see the performance of it

# In[ ]:


tempData.head()


# In[ ]:


#how many frauddata and  non fraud records 
tempData.isFraud.value_counts()


# ### From output above, it is sure that this data is imbalance. 

# In[ ]:


### Description of isFraud column 
tempData.groupby("isFraud").describe()


# In[ ]:


tempData.card4.isna().any()


# In[ ]:


tempData.card4.isna().sum()


# ### Description of TransactionAmt groupd on isFraud column 
# 
# * Mean value of TransactionAmt for catogry  1 is high
# * Median value of TransactionAmt for catogry  1 is less than catogry 0 
# * The TransactionAmt for catogry  1 is  more skewed on right side for category 1

# ### How many type of cards are used for transactions ?

# In[ ]:


from IPython.display import display, Markdown


# In[ ]:



sns.distplot(tempData.TransactionAmt)


# In[ ]:


sns.jointplot(x="isFraud", y="TransactionAmt", data=tempData);


# ### Stratified sampling 

# In[ ]:


data = tempData.groupby('isFraud').apply(lambda x: x.sample(n=20000))
data.reset_index(drop=True, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.ProductCD.value_counts()


# In[ ]:


data.card4.value_counts()


# In[ ]:


data.card4.value_counts()


# In[ ]:


data.replace({"ProductCD":{'C':0,'H':1,'R':2,'S':3,'W':4},
            "card4":{"american express":0,"discover":1,"mastercard":2,"visa":3}
           }, inplace=True)


# In[ ]:


data.card4.value_counts()


# In[ ]:


data.isna().any()


# In[ ]:


data.ProductCD.value_counts()


# In[ ]:


data.dropna(axis=0,inplace=True)


# In[ ]:


data.head()


# ### Deviding data into training and testing part

# In[ ]:


indData = data.loc[:,"TransactionAmt":"card4"]
depdData = data.loc[:,'isFraud']
indTrain, indTest, depTrain, depTest = train_test_split(indData, depdData, test_size=0.2, random_state=0)


# In[ ]:


mytree =  tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)


# In[ ]:


import sklearn.metrics as metric


# In[ ]:


mytree.fit(indTrain,depTrain)
predVal =  mytree.predict(indTest)
actVal = depTest.values
metric.confusion_matrix(actVal, predVal)


# In[ ]:


metric.accuracy_score(actVal, predVal)


# In[ ]:


rft =  ensem.RandomForestClassifier(criterion='entropy',max_depth=30,
                                   n_estimators=500,verbose=0)
rft.fit(indTrain,depTrain)
predVal =  rft.predict(indTest)
actVal = depTest.values
print(metric.confusion_matrix(actVal, predVal))
print(metric.accuracy_score(actVal, predVal))


# # Hope you have enjoyed this kernel. If enjoyed kindly upvote it 

# In[ ]:




