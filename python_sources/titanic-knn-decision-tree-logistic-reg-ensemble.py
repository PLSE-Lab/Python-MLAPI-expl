#!/usr/bin/env python
# coding: utf-8

# # Descision Tree

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/titanic/train.csv")    ## Train dataset

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')  ## Test dataset


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# In[ ]:


df.shape, df_test.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# #  Imputing the missing values

# For train

# In[ ]:


mean_val = df['Age'].mean()
mean_val


# In[ ]:


df['Age'] = df['Age'].fillna(value=mean_val)


# In[ ]:


mode_val = df['Embarked'].mode()
mode_val


# In[ ]:


df['Embarked'] = df['Embarked'].fillna(value='S')  


# In[ ]:


df['Cabin'] = df['Cabin'].fillna(value='NaN')


# In[ ]:


df.isnull().sum()


# For test

# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test['Age'].mean()


# In[ ]:


df_test['Age'] = df_test['Age'].fillna(value=(df_test['Age'].mean()))


# In[ ]:


df_test['Cabin'].mode()


# In[ ]:


df_test['Cabin'] = df_test['Cabin'].fillna(value=(df_test['Cabin'].mode()[0]))


# In[ ]:


df_test['Fare'] = df_test['Fare'].fillna(value=(df_test['Fare'].mean()))


# In[ ]:


df_test.isnull().sum()


# # Data Types

# In[ ]:


df = df.astype({'SibSp':'object','Parch':'object','Pclass':'object'})


# In[ ]:


df_test = df_test.astype({'SibSp':'object','Parch':'object','Pclass':'object'})


# # Convert categ. variables into binary valued 
For Train
# In[ ]:


df = pd.get_dummies(df.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1))


# In[ ]:


df.head()


# In[ ]:


df.shape

For Test
# In[ ]:


df_test.head()


# In[ ]:


df_test['Parch'].value_counts()  ## train Data does notcontain Parch_9
## But test_X data contains Parch_9 its shape when we create dummies will (418,25), but we want it to be (418,24)
## It is very much imp that the no. of columns train_X and test_X match
## Now when we split the data to be trained we get shape of train_X with 24 columns 
## And so as test_X will be having 24 columns when we put Parch_9 into Parch_0


# In[ ]:


df_test['Parch'].replace({9:0},inplace=True)
df_test['Parch'].value_counts()


# In[ ]:


df_test = df_test.astype({'Parch':'object'})


# In[ ]:


test_X = pd.get_dummies(df_test.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1))
test_X.head()


# In[ ]:


test_X.shape


# # Remember we always need to SCALE our data bcoz sometimes the parameters have different units (Distance Based Algo. eg. KNN)

# For Train

# In[ ]:


train_X = df.drop(['Survived'], axis=1)    ## X has the feature variables(Independent Var) which help in prediction of target
train_y = df['Survived']                  ## Y has the Dependent var

train_X.shape, train_y.shape


# In[ ]:


## MinMaxScaler scales down values in the range 0 to 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train_X)


# In[ ]:


## X_scaled is an array O/P, convert it into Panda DataFrame
## X earlier contained the Features therefore it shld only now hold the scaled features

train_X = pd.DataFrame(X_scaled, columns=train_X.columns)


# In[ ]:


train_X.head()   ## Range = 0 to 1


# For Test

# In[ ]:


testx_scaled = scaler.fit_transform(test_X)

test_X = pd.DataFrame(testx_scaled, columns=test_X.columns)
test_X.head()


# # Random Forest algo
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=100, max_depth=12)


# In[ ]:


rf.fit(train_X, train_y)


# In[ ]:


rf.score(train_X, train_y)


# In[ ]:


train_predict = rf.predict(train_X)


# In[ ]:


test_pred = rf.predict(test_X)


# In[ ]:


test_pred


# In[ ]:


output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': test_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




