#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import math

import xgboost as xgb
from sklearn.model_selection import train_test_split
seed = 78
test_size = 0.3
from sklearn.metrics import accuracy_score

from numpy import loadtxt
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers

import os


# In[ ]:



#gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
testdf = pd.read_csv("../input/titanic/test.csv")
traindf = pd.read_csv("../input/titanic/train.csv")
traindf.head(3)


# In[ ]:


#assigning categories

def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] =  data['Age'].fillna(data['Age'].dropna().median())
    
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] =='female',  'Sex'] = 1
    
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data["Embarked"] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] =2
    
    
    
## call function:

clean_data(traindf)
clean_data(testdf)


# In[ ]:


## combine test and train as single to apply some function and applying the feature scaling
all_data=[traindf,testdf]


for dataset in all_data:
    
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    dataset['Fare_Range'] = pd.cut(dataset['Fare'], bins=[-10000,0,7.91,14.45,31,120,10000], labels=['very_low_fare','Low_fare','median_fare',
                                                                                      'Average_fare','high_fare','very_high_fare'])
    
    dataset['Age_Range'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])


# In[ ]:


# Define function to extract titles from passenger names
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


traindf.shape


# In[ ]:


testdf.shape


# In[ ]:


traindf.head(5)


# In[ ]:


print('check the nan value in test data')
print(testdf.isnull().sum())


# In[ ]:


#cabin has many null so remove
del testdf['Cabin']
del traindf['Cabin']


# In[ ]:


del testdf['Name']
del traindf['Name']
del testdf['Ticket']
del traindf['Ticket']


# In[ ]:


del testdf['PassengerId']
del traindf['PassengerId']


# In[ ]:


testdf.head()


# In[ ]:


traindf.head(5)


# In[ ]:


from fastai.tabular import *


# In[ ]:


dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize','Fare_Range','Age_Range','Title']
cont_names = ['Age', 'Fare']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


path = ""


# In[ ]:


data = (TabularList.from_df(traindf, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_rand_pct(0.2)
                           .label_from_df(cols=dep_var)
                           .databunch())


# In[ ]:


data.show_batch(rows=10)


# In[ ]:


learn = tabular_learner(data, layers=[200,150,130,100,50, 30,30,20], metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, 1e-2)


# In[ ]:


ans = []
for i in range (0,418):
    
    y = learn.predict(testdf.iloc[i])
    list(y)
    p = int(y[1])
    ans.append(p)


# In[ ]:



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = gender_submission['PassengerId']
submission_df['Survived'] = ans
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(418)


# In[ ]:




