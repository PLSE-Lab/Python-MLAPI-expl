#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


datasets = [train,test]

for df in datasets:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)


# In[ ]:


for df in datasets:
    df['hasCabin'] = np.where(pd.isnull(df['Cabin']),0,1)
    df.loc[pd.isnull(df['Embarked']),'Embarked'] = 'None'
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    
train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

SEED = 1
np.random.seed(SEED)
le = dict()
le['Sex'] = LabelEncoder()
le['Sex'].fit(train.Sex)
le['Embarked'] = LabelEncoder()
le['Embarked'].fit(train.Embarked)
le['Title'] = LabelEncoder()
le['Title'].fit(pd.concat([train.Title, test.Title], axis=0))

for df in datasets:
    df['Sex'] = le['Sex'].transform(df['Sex'])
    df['Embarked'] = le['Embarked'].transform(df['Embarked'])
    df['Title'] = le['Title'].transform(df['Title'])
    
train.head()


# In[ ]:


for df in datasets:
    df.loc[pd.isnull(df['Age']), 'Age'] = df['Age'].mean()

for df in datasets:
    df.loc[:,'Age'] = np.round(df['Age'])


# In[ ]:


for df in datasets:
    df.loc[pd.isnull(df['Fare']),'Fare'] = df['Fare'].mean()


# In[ ]:


train.describe()


# In[ ]:



x_train0 = train.drop(['PassengerId','Survived'],axis=1)
y_train0 = train['Survived']

x_test0 = test.drop(['PassengerId'],axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train0)
x_test = sc.fit_transform(x_test0)

y_train = y_train0.values.astype('float32')


# Use keras and simple relu to training.

# In[ ]:


from keras import models
from keras import layers
from keras import optimizers


# In[ ]:


epochs_num = 100
batch_size = 20
input_dim = len(x_train[0])


# In[ ]:


def get_model(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(units = 7, kernel_initializer = 'lecun_uniform', activation = 'relu', input_dim = input_dim))
    model.add(layers.Dense(units = 5, kernel_initializer = 'lecun_uniform', activation = 'relu'))
    model.add(layers.Dense(units = 1, kernel_initializer = 'lecun_uniform', activation = 'sigmoid'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy', metrics=['accuracy']) 
    return model


# In[ ]:


model = get_model(input_dim)

history = model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_size, verbose=1)


# In[ ]:


predict = model.predict(x_test)


# In[ ]:


my_submission = pd.DataFrame({
	'PassengerId': test.PassengerId, 
	'Survived': pd.Series(predict.reshape((1,-1))[0]).round().astype(int)
})

my_submission.head()


# In[ ]:


my_submission.to_csv('submission.csv', index=False)

