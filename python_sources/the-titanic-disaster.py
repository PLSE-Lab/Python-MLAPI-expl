#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# ## Data Preprocessing

# In[ ]:


#input the data
train = pd.read_csv("../input/train.csv");
test = pd.read_csv('../input/test.csv');
test_new = pd.read_csv('../input/test.csv');


# ### Training Data

# In[ ]:


train.head()


# Lets check Info of training data to know data types and missing values of columns.

# **Information on the dataset**

# In[ ]:


train.info()


# Missing values are present in 
# 1. Age
# 2. Cabin
# 3. Embarked
# 
# Cabin has too many missing values i think we can't handle it so lets drop that coulmn, also we will drop passengerId as it is just a index from 1-891.

# #### Handle Missing values

# 1. 1st we will check missing values in Embarked.

# In[ ]:


train[train['Embarked'].isnull()]


# Most of the entries from 'Q(Queenstown)' look to have ticket no above 330000 so think it might they might not be from Q. So they should be from S(Southampton) or C(Cherbourg). My best guess is from ticket and fare they might be from C(Cherbourg).

# **Missing values handled.**

# In[ ]:


train['Embarked'].fillna('C',inplace=True);
train[train['Embarked'].isnull()]


# 2. Now lets check missing values in Age

# In[ ]:


train[train['Age'].isnull()]


# There is a relation between Age and Name (Mr,Miss,Mrs...) so based on that we can estimate age and fill missing values.

# In[ ]:


print('Mr is for people between')
print(train[train['Name'].str.contains("Mr\.")]['Age'].min())
print('and')
print(train[train['Name'].str.contains("Mr\.")]['Age'].max())
print('Mean is')
print(train[train['Name'].str.contains("Mr\.")]['Age'].mean())
print(' ')

print('Master is for people between')
print(train[train['Name'].str.contains("Master\.")]['Age'].min())
print('and')
print(train[train['Name'].str.contains("Master\.")]['Age'].max())
print('Mean is')
print(train[train['Name'].str.contains("Master\.")]['Age'].mean())
print(' ')

print('Mrs is for people between')
print(train[train['Name'].str.contains("Mrs\.")]['Age'].min())
print('and')
print(train[train['Name'].str.contains("Mrs\.")]['Age'].max())
print('Mean is')
print(train[train['Name'].str.contains("Mrs\.")]['Age'].mean())
print(' ')

print('Miss is for people between')
print(train[train['Name'].str.contains("Miss\.")]['Age'].min())
print('and')
print(train[train['Name'].str.contains("Miss\.")]['Age'].max())
print('Mean is')
print(train[train['Name'].str.contains("Miss\.")]['Age'].mean())
print(' ')


# So based on those upper and lower limits in the dataset let us fill missing values. For Mr if parch is present lest limit till 16 and for Miss if parch is present lest limit for 18. Also for Miss lest limit max for 35, for Master lets limit max for 10.

# **Missing values handled**

# In[ ]:



x = train[train['Age'].isnull()].index.tolist()
for i in x:
    name = train.iloc[i]['Name']
    if(str('Mr.') in name):
        if(train.iloc[i]['Parch']>0):
                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(11, 16)
        else:
                 train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(16, 50)
    elif(str('Master.') in name):
          train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(1, 10)
    elif(str('Mrs.') in name):
         train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(21, 63)
    elif(str('Miss.') in name):
        if(train.iloc[i]['Parch']>0):
                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(1, 18)
        else:
                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(18, 35)

train[train['Age'].isnull()]


# We missed out a Dr in the missing list lets handle it too
# 

# In[ ]:


train.loc[train['PassengerId']==767,'Age']=40
train[train['Age'].isnull()]


# lets again check the info om the dataset.

# In[ ]:


train.info()


# Except for cabin all the missing values are handled.

# Cabin has too many missing values so we drop it and PassengerId and name are also not useful fro predction so we drop them too. 

# In[ ]:


train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
train.info()


# ### Feature encoding

# Converting Ctegorical variables into numbers. 

# In[ ]:


train=pd.get_dummies(train,columns=['Sex','Embarked'])
train.drop(['Sex_female','Embarked_Q'],axis=1,inplace=True)
train.head()


# Lets look at relational heat map for feature selection.

# In[ ]:


plt.figure(figsize=(12,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# ### Normalization

# In[ ]:


train_x = train.iloc[:,1:]
train_y= train.iloc[:,0]

train_x = preprocessing.scale(train_x)
train_x


# ## Prediction

# In[ ]:


classifier=Sequential()
classifier.add(Dense(output_dim=12,init='uniform',activation='relu',input_dim=8))
classifier.add(Dropout(0.3))
classifier.add(Dense(output_dim=8,init='uniform',activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


classifier.fit(train_x,train_y,batch_size=10,epochs=300)


# **Preidcting test set**

# In[ ]:


test.head()
passengers = test['PassengerId'] 
print(test.info())


# In[ ]:


# handle missing age
y = test[test['Age'].isnull()].index.tolist()
for i in y:
    name = test.iloc[i]['Name']
    if(str('Mr.') in name):
        if(test.iloc[i]['Parch']>0):
                test.loc[i,'Age'] = random.randrange(11, 16)
        else:
                 test.loc[i,'Age'] = random.randrange(16, 50)
    elif(str('Master.') in name):
          test.loc[i,'Age'] = random.randrange(1, 10)
    elif(str('Mrs.') in name):
         test.loc[i,'Age'] = random.randrange(21, 63)
    elif(str('Miss.') in name):
        if(test.iloc[i]['Parch']>0):
                test.loc[i,'Age'] = random.randrange(1, 18)
        else:
                test.loc[i,'Age'] = random.randrange(18, 35)

test.loc[88,'Age']= random.randrange(18, 35)               
test[test['Age'].isnull()]


# In[ ]:


#handle missing fare
test.loc[152,'Fare']=7.7500
test[test['Fare'].isnull()]


# In[ ]:



test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
test.info()


# In[ ]:


test=pd.get_dummies(test,columns=['Sex','Embarked'])
test.drop(['Sex_female','Embarked_Q'],axis=1,inplace=True)

test = preprocessing.scale(test)
test


# In[ ]:


y_pred = classifier.predict(test) 
pred=[]
for i in range(0,y_pred.shape[0]):
    if(y_pred[i]>0.5):
        pred.append(1)
    else:
        pred.append(0)


# In[ ]:


test_new.head()


# In[ ]:


output = pd.DataFrame({'PassengerId': test_new.PassengerId,'Survived': pred})
output.to_csv('submission.csv', index=False)

