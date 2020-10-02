#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load the required Libraries.

# Please make sure to upvote if you found it helpful and comment for any suggestion or doubt

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Read the train and test csv and display the head.

# In[ ]:


data = pd.read_csv('/kaggle/input/titanic/train.csv')
display(data.head())

trainlen = len(data)
print(trainlen)

test = pd.read_csv('/kaggle/input/titanic/test.csv')
display(test.head())

passenger_id = test.PassengerId


# Join train and test together for combined cleaning of features.

# In[ ]:


data_com = pd.concat([data, test])
data_com.info()


# ## Feature Engineering

# Create new column Title which have the title of all names

# In[ ]:


data_com['Title']= data_com.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]


# Replace rarely occuring titles with miss, mr ,etc 

# In[ ]:


data_com['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],inplace=True)


# Filling missing values of age with the help of titles.[](http://)

# In[ ]:


data_com.loc[(data_com.Age.isnull())&(data_com.Title=='Mr'),'Age']=33
data_com.loc[(data_com.Age.isnull())&(data_com.Title=='Mrs'),'Age']=36
data_com.loc[(data_com.Age.isnull())&(data_com.Title=='Master'),'Age']=5
data_com.loc[(data_com.Age.isnull())&(data_com.Title=='Miss'),'Age']=22
data_com.loc[(data_com.Age.isnull())&(data_com.Title=='Other'),'Age']=45


# Filling missing value of Embarked with S.

# In[ ]:


data_com.Embarked.fillna('S', inplace=True)
data_com.isnull().sum()


# Creating New column Cabin_code

# In[ ]:


x = list(data_com.Cabin.fillna('O'))
for i in range(len(x)):
    x[i] = x[i][0]
data_com['Cabin_code']= pd.Series(x)


# Encoding Age column 

# In[ ]:


data_com['Age_band']=0
data_com.loc[data_com['Age']<=16,'Age_band']=0
data_com.loc[(data_com['Age']>16)&(data_com['Age']<=32),'Age_band']=1
data_com.loc[(data_com['Age']>32)&(data_com['Age']<=48),'Age_band']=2
data_com.loc[(data_com['Age']>48)&(data_com['Age']<=64),'Age_band']=3
data_com.loc[data_com['Age']>64,'Age_band']=4
data_com.head(2)


# Filling Fare

# In[ ]:


data_com.Fare.fillna(10, inplace=True)


# Creating new Columns

# In[ ]:


data_com['Family_Size']=0
data_com['Family_Size']=data_com['Parch']+data_com['SibSp']#family size
data_com['Alone']=0
data_com.loc[data_com.Family_Size==0,'Alone']=1 #Alone


# Encoding Fare

# In[ ]:


data_com['Fare_Range']=pd.qcut(data_com['Fare'],4)


# In[ ]:


data_com['Fare_cat']=0
data_com.loc[data_com['Fare']<=7.91,'Fare_cat']=0
data_com.loc[(data_com['Fare']>7.91)&(data_com['Fare']<=14.454),'Fare_cat']=1
data_com.loc[(data_com['Fare']>14.454)&(data_com['Fare']<=31),'Fare_cat']=2
data_com.loc[(data_com['Fare']>31)&(data_com['Fare']<=513),'Fare_cat']


# Dropping Unnecessary Columns

# In[ ]:


data_com.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)


# Splitting Combined data into original train and test data 

# In[ ]:


train = data_com.iloc[:891]
test = data_com.iloc[891:]

y = train['Survived']
x = train.drop('Survived', axis=1)


# Splitting again using train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25)
test.drop('Survived', axis=1, inplace=True)


# ## Predictive Modelling Using CatBoost

# In[ ]:


from catboost import CatBoostClassifier
from catboost import Pool
from catboost import cv


# In[ ]:


seed = 1


# Creating list of categorical Features

# In[ ]:


cat_features = ['Embarked', 'Sex', 'Title', 'Cabin_code']


# Applying CatBoostClassifier

# In[ ]:


params = {'loss_function':'Logloss',
         'eval_metric':'AUC',
         'verbose': 200,
          'cat_features':cat_features,
         'random_seed': seed}

cbc1 = CatBoostClassifier(**params)

cbc1.fit(xtrain, ytrain, eval_set= (xtest, ytest), use_best_model=True, plot=True)


# #### Randomized Search for finding best parameters

# In[ ]:


grid = {
    'learning_rate': [0.05, 0.07, 0.09, 0.3],
    'depth': [5, 6, 7],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}


# Creating Pool object

# In[ ]:


train_pool = Pool(x, label=y, cat_features=cat_features)


# In[ ]:


model =CatBoostClassifier(
        loss_function='Logloss',   # RMSE with log1p-transformed labels is RMSLE
        early_stopping_rounds=100,
        has_time=True,
        iterations=5000
    )


# In[ ]:


model.randomized_search(grid, X=train_pool)


# In[ ]:


params= {'depth': 5,
  'l2_leaf_reg': 5,
  'learning_rate': 0.07,
  'grow_policy': 'Depthwise',
    'loss_function':'Logloss',
        'cat_features': cat_features}


# In[ ]:


model = CatBoostClassifier(**params)

model.fit(xtrain, ytrain, eval_set= (xtest, ytest), use_best_model=True, plot=True)


# In[ ]:


model.fit(x, y)


# ## Submission

# In[ ]:


results = model.predict(test)


# In[ ]:


submission = pd.DataFrame({'PassengerId':passenger_id,'Survived': results})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
display(submission)
filename = '/kaggle/working/Titanic Predictions catboost2.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:




