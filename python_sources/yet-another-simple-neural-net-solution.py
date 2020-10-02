#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout


# ## Loading Data

# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

train_data_copy = train_data.copy()
test_data_copy = test_data.copy()


# In[ ]:


train_data.head()


# ## Data Preprocessing

# ### Merge Train and Test Data
# 
# We will merge train and test data for making it easier to apply same preprocessing steps

# In[ ]:


test_data['Test'] = 1
train_data['Test'] = 0
data = train_data.append(test_data, sort = False)

drop_cols = list()
one_hot_encoding_cols = list()
normalization_cols = list()


# ### PassengerId
# 
# PassengerId is simply unique identifiers for each passengers and doesn't have any impact on the outcome. So we will simply remove this column from our training and testing dataset.

# In[ ]:


data.drop('PassengerId', axis = 1, inplace = True)


# ### Pclass
# This column has no missing value and only 3 class and classes has a ordered relationship. So we don''t need to apply any encoding for this cloumn.

# In[ ]:


data['Pclass'].value_counts(dropna = False)

normalization_cols.append('Pclass')


# ### Name
# 
# Assuming someones name in its entirety doesn't have any effect on that person's survival, title on the other hand can tell us a lot about that person like class, age, gender, marital status etc. And it also much easier to process than the full name. So, we will drop the name column and keep the titles only.

# In[ ]:


data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
data.drop('Name', axis = 1, inplace = True)


# In[ ]:


data['Title'].value_counts()


# In[ ]:


mapping = {'Col': 'Army', 'Mlle' : 'Miss', 'Major' : 'Army', 'Sir': 'Royal',
          'Mme': 'Mrs', 'Capt' : 'Army', 'Don' : 'Royal', 'Jonkheer' : 'Royal',
          'Ms' : 'Miss', 'Countess' : 'Royal', 'Lady': 'Royal'}
           
data.replace({'Title': mapping}, inplace=True)


# In[ ]:


one_hot_encoding_cols.append('Title')
data.head()


# ### Sex

# In[ ]:


data['Sex'].value_counts(dropna = False)


# In[ ]:


label_encoder = preprocessing.LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

data.head()


# ### Age

# In[ ]:


data[data['Test'] == 0].groupby(['Title', 'Sex']).Age.mean()


# In[ ]:


data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Army'), 'Age'] = 56.60

data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Dr'), 'Age'] = 49.00
data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Dr'), 'Age'] = 40.60

data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Master'), 'Age'] = 4.57

data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Miss'), 'Age'] = 21.85

data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Mr'), 'Age'] = 32.36

data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Mrs'), 'Age'] = 35.78

data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Rev'), 'Age'] = 43.16

data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Royal'), 'Age'] = 40.50
data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Royal'), 'Age'] = 42.33


# In[ ]:


normalization_cols.append('Age')
data.head()


# ### SibSp

# In[ ]:


data['SibSp'].value_counts()
normalization_cols.append('SibSp')


# ### Parch

# In[ ]:


data['Parch'].value_counts()
normalization_cols.append('Parch')

data.head()


# ### Ticket

# In[ ]:


data.drop('Ticket', axis =1, inplace = True)


# ### Fare

# In[ ]:


normalization_cols.append('Fare')


# ### Cabin

# In[ ]:


data['HasCabin'] = data['Cabin'].isnull() == False
data['HasCabin'].replace(False, 0, inplace = True)
data['HasCabin'].replace(True, 1, inplace = True)

data.drop('Cabin', axis =1, inplace = True)


# ### Embarked

# In[ ]:


one_hot_encoding_cols.append('Embarked')


# ### Applying Encoding and Normalization

# In[ ]:


data.head()


# In[ ]:


data = pd.get_dummies(data = data, columns = one_hot_encoding_cols)


# In[ ]:


data.head()


# In[ ]:


std = data[data['Test'] == 0][normalization_cols].std(axis = 0)
mean = data[data['Test'] == 0][normalization_cols].mean(axis = 0)

data[normalization_cols] = (data[normalization_cols] - mean) / std


# In[ ]:


data.head(10)


# ## Fitting

# In[ ]:


train_data = data[data['Test'] == 0].drop(columns = ['Test'])

test_data = data[data['Test'] == 1].drop(columns = ['Survived', 'Test'])


# In[ ]:


train_data.head()
train_data.shape


# In[ ]:


test_data.head()


# In[ ]:


X = train_data.iloc[: , 1:].to_numpy()
y = train_data.iloc[:, 0].to_numpy()

print(str(X.shape))
print(str(y.shape))


# In[ ]:


def create_model():

    model = Sequential()
    model.add(Dense(14, input_dim = 19, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation = 'relu'))
    
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model


# In[ ]:


epochs = 20
model = create_model()
history = model.fit(X, y, epochs=epochs, validation_split = 0.3, batch_size=10)


# In[ ]:


epochs = 20
model = create_model()
history = model.fit(X, y, epochs=epochs, batch_size=10, verbose = 0)


# In[ ]:


X_test = test_data.to_numpy()


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


submission = pd.DataFrame(test_data_copy[['PassengerId']])
submission['Survived'] = prediction
submission['Survived'] = submission['Survived'].apply(lambda x: 0 if x < 0.5 else 1)


# In[ ]:


submission.to_csv('submission.csv', index = False)


# In[ ]:


submission.head(10)

