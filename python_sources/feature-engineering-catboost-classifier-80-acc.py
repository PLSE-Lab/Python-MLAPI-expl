#!/usr/bin/env python
# coding: utf-8

# ## Importing Required Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from catboost import Pool, CatBoostClassifier, cv
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading Data

# In[ ]:


traind=pd.read_csv("../input/titanic/train.csv")
testd=pd.read_csv("../input/titanic/test.csv")


# ## Analysis

# In[ ]:


traind.head()


# In[ ]:


testd.head()


# In[ ]:


traind.describe()


# In[ ]:


# Check the Missing Values
traind.isnull().sum()


# In[ ]:


testd.isnull().sum()


# # Data Preparation

# ## Delete the columns "Passenger Id" and "Ticket" because those wont contribute anything as features for classification. Along with them "Cabin" is also removed as too many missing values prevails in it.

# In[ ]:


del traind["PassengerId"]
del traind["Ticket"]
del traind["Cabin"]
del testd["PassengerId"]
del testd["Ticket"]
del testd["Cabin"]


# ## Dealing with NaN values in Age Column

# ## 177 values out of 891 in train and 86 of 418 in test data ages are missing. This percentage is not small but it's rather not too big either. The NaN values have been replaced with the median of the age values present in the column with respect to the gender of a particular person. This is one of the ways to handle missing data.

# In[ ]:


# Gropuing by age and sex and finding medians
trainmedians = traind.groupby('Sex')['Age'].median()
testmedians = testd.groupby('Sex')['Age'].median()


# In[ ]:


trainmedians


# In[ ]:


testmedians


# In[ ]:


traind = traind.set_index(['Sex'])
testd = testd.set_index(['Sex'])


# In[ ]:


traind


# In[ ]:


# Filling the missing age values with calculated medians
traind['Age'] =traind['Age'].fillna(trainmedians)
testd['Age'] =testd['Age'].fillna(testmedians)


# In[ ]:


# Resetting the index
traind = traind.reset_index()
testd=testd.reset_index()


# In[ ]:


# Only keep title from Name column either Mr, Mrs, Miss, Rev like dat

Title = []

for i in range(len(traind)):
    Title.append(traind["Name"][i].split(",")[1].split(".")[0].lstrip().rstrip())
traind["Name"] = Title


# In[ ]:


Title = []

for i in range(len(testd)):
    Title.append(testd["Name"][i].split(",")[1].split(".")[0].lstrip().rstrip())
testd["Name"] = Title


# In[ ]:


farmedian=testd["Fare"].median()

index = list(np.where(testd['Fare'].isna())[0])[0]

testd["Fare"][index]=farmedian


# In[ ]:


# As oonly two embarked values are missing I removed those two rows
traind=traind.dropna()


# In[ ]:


traind


# In[ ]:


traind.isnull().sum()


# In[ ]:


testd.isnull().sum()


# In[ ]:


labels=traind["Survived"]


# In[ ]:


del traind["Survived"]


# In[ ]:


traind


# In[ ]:


testd


# In[ ]:


traind.dtypes


# In[ ]:


cat_features_index = np.where(traind.dtypes != float)[0]


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(traind,labels,train_size=.85)


# In[ ]:


model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True)


# In[ ]:


model.fit(xtrain,ytrain,cat_features=cat_features_index,eval_set=(xtest,ytest))


# In[ ]:


pred = model.predict(testd)
pred = pred.astype(np.int)


# In[ ]:


testpass=pd.read_csv("../input/titanic/test.csv")
submission = pd.DataFrame({'PassengerId':testpass['PassengerId'],'Survived':pred})


# In[ ]:


submission.to_csv('catboost.csv',index=False)

