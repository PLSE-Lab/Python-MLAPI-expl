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


import tensorflow as tf
import pandas as pd
import seaborn as sb

sb.set()


# Let's read in the train dataset. ( Dropping PassengerId, as it's just a primary key )

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train = train.drop(['PassengerId'], 1)


# Let's have a look at the column headers available in the dataframe

# In[ ]:


train.columns


# Clearly, as the feature ***Name*** wouldn't be having much impact on survival chances, let's drop it too.

# In[ ]:


train = train.drop(['Name'],1)


# Also, the features ***Parch*** and ***SibSp*** collectively represent the number of family members. Hence, let's club them into a single feature column.

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch']
train = train.drop(['SibSp','Parch'],1)
train = train.drop(['Ticket'],1)


# Now, let's take a look at the unique elements contained in each column feature. 

# In[ ]:


print('FamilySize: ' + str(train['FamilySize'].unique())) 
print('Embarked: ' + str(train['Embarked'].unique())) 
print('Pclass: ' + str(train['Pclass'].unique()))
print('Sex: ' + str(train['Sex'].unique())) 


# Let's start by examining *null* entries in the Database

# In[ ]:


print(train['Cabin'].isnull().sum())


# This implies, that majority of the entries are null in the training data. Let's, replace null entries with the word 'NA'.

# In[ ]:


train['Cabin']=train['Cabin'].fillna('NA')


# The Cabin names are of vast variety, however it can be seen, that they effectively belong to groups (A, B, C, .. T):

# In[ ]:


print(train['Cabin'].unique())


# Let's replace these values with their respective group characters, i.e. (A, B, C, .. T)

# In[ ]:


train['Cabin']=train['Cabin'].astype(str).str[0]


# Let's see, how many null entries we have left:

# In[ ]:


print(train.isnull().sum())


# Clearly, as the ***Embarked*** feature has only two *null* entries, let's replace them with the most frequently occuring value in the column.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])


# Now, in order to resolve the 177 null entries for the feature ***Age*** let's take median of the given ***Sex*** in a ***Pclass*** and replace *null* elements with it  
# *( the median value )*.

# In[ ]:


for i in ['male','female']:
    for j in [3,1,2]:
        temp_dataset=train[(train['Sex']==i) &  (train['Pclass']==j)]['Age'].dropna()
        train.loc[(train.Age.isnull()) & (train.Sex==i) & (train.Pclass==j),'Age']=int(temp_dataset.median())


# # Feature Engineering

# Let's first have a look at our train dataset:

# In[ ]:


train.head()


# As, we work with numerical values, The feature values for the fields:
# 1. Sex
# 2. Cabin
# 3. Embarked 
# need to be coverted to numeric values

# # Sex
# Let's first visualize the survival rate for the either gender:

# In[ ]:


sb.catplot(x = 'Sex', y = 'Survived', data = train, kind = 'bar')


# Clearly, females have a higher survival possibility than males. Let's, encode Male with 1 & Female with 2.

# In[ ]:


train.loc[(train['Sex'] == 'male'),'Sex']=1
train.loc[(train['Sex'] == 'female'),'Sex']=2


# # Embarked
# Let's first visualize the survival rate vs Embark:

# In[ ]:


sb.catplot(y="Embarked",x='Survived',data=train,kind='bar')


# Survival rate for C > Q > S. Let's accordingly encode 'C': 3, 'Q': 2 & 'S': 1

# In[ ]:


train.loc[(train['Embarked'] == 'S'),'Embarked']=1
train.loc[(train['Embarked'] == 'Q'),'Embarked']=2
train.loc[(train['Embarked'] == 'C'),'Embarked']=3


# # Cabin
# Let's visualize Cabin vs Survival Rate:

# In[ ]:


sb.catplot(y='Cabin',x='Survived',data=train,kind='bar')


# According to the survival values found, we'll encode our Cabin classes.

# In[ ]:


train.loc[(train['Cabin'] == 'T'),'Cabin']=0
train.loc[(train['Cabin'] == 'N'),'Cabin']=1
train.loc[(train['Cabin'] == 'A'),'Cabin']=2
train.loc[(train['Cabin'] == 'G'),'Cabin']=3
train.loc[(train['Cabin'] == 'C'),'Cabin']=4
train.loc[(train['Cabin'] == 'F'),'Cabin']=5
train.loc[(train['Cabin'] == 'E'),'Cabin']=6
train.loc[(train['Cabin'] == 'B'),'Cabin']=7
train.loc[(train['Cabin'] == 'D'),'Cabin']=8


# Now, that we've succesfully encoded all our non-numeric values, let's again take a look at out dataset:

# In[ ]:


train.head()


# We also, see that it might be better to split the ***Fare*** and ***Age*** categories into groups. Let's split ***Age*** into 5 and ***Fare*** into 3 groups respectively. Here are the category ranges:

# In[ ]:


train['Age_Band']=pd.cut(train['Age'],5)
print(train['Age_Band'].unique())

train['Fare_Band']=pd.cut(train['Fare'],3)
print(train['Fare_Band'].unique())


# In[ ]:


train = train.drop(['Age_Band'], 1)
train = train.drop(['Fare_Band'], 1)


# In[ ]:


train.loc[(train['Age']<=16.136),'Age']=1
train.loc[(train['Age']>16.136) & (train['Age']<=32.102),'Age']=2
train.loc[(train['Age']>32.102) & (train['Age']<=48.068),'Age']=3
train.loc[(train['Age']>48.068) & (train['Age']<=64.034),'Age']=4
train.loc[(train['Age']>64.034) & (train['Age']<=80.),'Age']=5

train.loc[(train['Fare']<=170.776),'Fare']=1
train.loc[(train['Fare']>170.776) & (train['Fare']<=314.553),'Fare']=2
train.loc[(train['Fare']>314.553) & (train['Fare']<=513),'Fare']=3


# Final values:

# In[ ]:


train.head()


# Now, let's split the train set into input data and labels

# In[ ]:


train_x = train.drop(['Survived'],1)
train_x = train_x.astype('float64').to_numpy()
train_labels = train['Survived'].to_numpy()


# # The Model
# We'll be using a Fully Connected Sequential model having two hidden layers with ( 32 & 16 neurons respectively ) and 1 output layer. 

# In[ ]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation = tf.nn.relu),
        tf.keras.layers.Dense(16, activation = tf.nn.relu),
        tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])


# We'll be using Adam optimization and binary_crossentropy loss, as this is a binary classification problem.

# In[ ]:


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


# **All Set!!!, Let's run the model!** *(25 epochs is an experimental value, and not precisely accurate)*

# In[ ]:


model.fit(train_x, train_labels, epochs=25)

