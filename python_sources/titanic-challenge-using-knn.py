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


df= pd.read_csv("/kaggle/input/titanic/train.csv")
df1=pd.read_csv("/kaggle/input/titanic/test.csv")
df2=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print("Sucess !")


# # Loading the data and performing the basic analysis

# In[ ]:


df1.head()


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna()


# In[ ]:


df.shape


# # Now, we are checking for which feature has a good impact on the chances of survival

# In[ ]:


df[["Pclass","Survived"]].groupby("Pclass").mean()


# In[ ]:


df[["Sex","Survived"]].groupby("Sex").mean()


# In[ ]:


df[["Embarked","Survived"]].groupby("Embarked").mean()


# Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


import re as re


# In[ ]:


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in df:
    df['Title'] = df['Name'].apply(get_title)

print(pd.crosstab(df['Title'], df['Sex']))


# In[ ]:


for dataset in df:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

print (df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# # From the above analysis it is clear that we PClass has the greatest impact on the chances of survival

# # We will be using KNN to predict wether the person was able to survive or not

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model=KNeighborsClassifier(n_neighbors=3)


# In[ ]:


X=df[["Pclass"]]
y=df[["Survived"]]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=6)
model.fit(X,y)


# In[ ]:


pred=model.predict(df1[["Pclass"]])


# In[ ]:


print("Pclass","Survived")
for i in range(0,100):
    print(df1["Pclass"].values[i],pred[i])


# In[ ]:



output = pd.DataFrame({'PassengerId': df1.PassengerId, 'Survived': pred})
output.to_csv('KNN_submission.csv', index=False)
print("Your submission was successfully saved!")

