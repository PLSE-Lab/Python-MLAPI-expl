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


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_data.fillna(0,inplace=True)
test_data.fillna(0,inplace=True)


# In[ ]:


train_data = train_data.drop(columns=['Name','Cabin'])
train_data['TFamily'] = train_data['SibSp'] + train_data['Parch']
train_data = train_data.drop(columns=['SibSp', 'Parch'])


# In[ ]:


test_data = test_data.drop(columns=['Name','Cabin'])
test_data['TFamily'] = test_data['SibSp'] + test_data['Parch']
test_data = test_data.drop(columns=['SibSp', 'Parch'])


# In[ ]:


X = train_data.drop(columns=['Survived'])
Y = train_data['Survived']
#choose the features we want to train, just forget the float data
cate_features_index = np.where(X.dtypes != float)[0]


# In[ ]:


from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import train_test_split


# In[ ]:


#make the x for train and test (also called validation data) 
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.80,random_state=2)


# In[ ]:


clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=2)


# In[ ]:


#now just to make the model to fit the data
clf.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest), early_stopping_rounds=100,verbose=False)


# In[ ]:


prediction = clf.predict(test_data)


# In[ ]:


df_sub = pd.DataFrame()
df_sub['PassengerId'] = test_data.PassengerId
df_sub['Survived'] = prediction.astype(np.int)

df_sub.to_csv('submission_clf.csv', header=True,index=False)

