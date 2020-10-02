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


training_data=pd.read_csv('/kaggle/input/titanic/train.csv',index_col=0)


# In[ ]:


training_data.head()


# In[ ]:


len(training_data)-training_data.count()


# In[ ]:


training_data=training_data.drop(['Name','Cabin','Ticket'],axis=1)
training_data.head()


# In[ ]:


len(training_data)-training_data.count()


# In[ ]:


training_data['Embarked'].value_counts()


# In[ ]:


training_data['Embarked']=training_data['Embarked'].fillna('S')
training_data.head()


# In[ ]:


len(training_data)-training_data.count()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
age_guesser=KNeighborsClassifier(n_neighbors=3)


# In[ ]:


neighbors_data=training_data
neighbors_data.head()


# In[ ]:


has_age=neighbors_data[~neighbors_data.isnull().any(axis=1)]
has_age.head()


# In[ ]:


y_train,index=pd.factorize(has_age['Age'])
print(y_train)


# In[ ]:


has_age.head()


# In[ ]:


x_train=has_age.drop('Age',axis=1)
scale_mean=x_train['Fare'].mean()
scale_std=x_train['Fare'].std()
x_train['Fare']=(x_train['Fare']-x_train['Fare'].mean())/x_train['Fare'].std()
x_train=pd.get_dummies(x_train)
x_train.head()


# In[ ]:


age_guesser.fit(x_train,y_train)


# In[ ]:


needs_age=neighbors_data[neighbors_data.isnull().any(axis=1)]
needs_age.head()


# In[ ]:


age_guess=needs_age.pop('Age')
needs_age.head()


# In[ ]:


needs_age['Fare']=(needs_age['Fare']-scale_mean)/scale_std
needs_age=pd.get_dummies(needs_age)
needs_age.head()


# In[ ]:


age_guess=index[age_guesser.predict(needs_age)]
print(age_guess)


# In[ ]:


needs_age=neighbors_data[neighbors_data.isnull().any(axis=1)]
has_age=neighbors_data[~neighbors_data.isnull().any(axis=1)]
needs_age['Age']=age_guess
needs_age.head()


# In[ ]:


new_data=pd.concat([needs_age,has_age])
new_data.head()


# In[ ]:


new_data.sort_index()
new_data.head()


# In[ ]:


new_data.describe()


# In[ ]:


training_data.describe()


# In[ ]:


import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


# In[ ]:


def prepare_data(df):
    age_mean=df['Age'].mean()
    age_std=df['Age'].std()
    fare_mean=df['Fare'].mean()
    fare_std=df['Fare'].std()
    class_mean=df['Pclass'].mean()
    class_std=df['Pclass'].std()
    labels=['Age','Fare','Pclass']
    df[labels]=(df[labels]-df[labels].mean())/df[labels].std()
    return df


# In[ ]:


scaled=prepare_data(new_data)
scaled.head()


# In[ ]:


scaled=pd.get_dummies(scaled)
scaled.head()


# In[ ]:


y=scaled.pop('Survived')
x=scaled
x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[ ]:


model = keras.Sequential(
[
    keras.layers.Dense(64, activation="relu",input_shape=(len(x_train.keys()),)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1,activation='sigmoid')
]
)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')


# In[ ]:


model.fit(x_train,y_train,epochs=500,verbose=1)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


prediction=model.predict(x_test).astype(int)
print(confusion_matrix(y_test,prediction))


# In[ ]:




