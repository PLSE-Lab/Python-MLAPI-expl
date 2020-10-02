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


df = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout


# In[ ]:


df.columns


# In[ ]:


X = df[['Pclass','Sex', 'Age', 'SibSp','Parch','Embarked']]
y = df['Survived']


# In[ ]:


value = X.Age.mean()
X['Age'] = X.Age.fillna(value=value)
X['Embarked']= X.Embarked.fillna('S')


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# In[ ]:


X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()


# In[ ]:


X_train_np.shape


# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=(9)))
model.add(Dropout(0.5))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train_np, y_train_np, epochs=100, batch_size=5,validation_data=[X_test, y_test])


# In[ ]:


score = model.evaluate(X_test, y_test, batch_size=128)
print(score)


# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test = test[['Pclass','Sex', 'Age', 'SibSp','Parch','Embarked']]
value = df_test.mean()
df_test['Age'] = df_test.Age.fillna(value=value)
df_test['Embarked']= df_test.Embarked.fillna('S')
df_test= pd.get_dummies(df_test)


# In[ ]:


test_survived = model.predict_classes(df_test)
test_survived = test_survived.reshape(418,)


# In[ ]:


# test_survived = test_survived.reshape(418,)
print(test_survived)


# In[ ]:


submission = pd.DataFrame({
    'PassengerId' : test["PassengerId"].astype('int64'),
    'Survived' : test_survived
})

# submission.to_csv('SVMC_55th.csv', index=False)
submission.to_csv('titanic.csv', index=False)


# In[ ]:




