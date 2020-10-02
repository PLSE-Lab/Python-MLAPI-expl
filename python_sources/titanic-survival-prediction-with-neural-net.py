#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy.random import seed
from tensorflow import set_random_seed


# In[ ]:


def missingData(data):
    return (data.isnull().sum() / data.shape[0]) * 100


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df = pd.concat([train, test], axis=0, sort=True)


# In[ ]:


df.head()


# In[ ]:


df['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=True)


# In[ ]:


df[df.Survived.notnull()].Title.value_counts()


# In[ ]:


df[((df.Title == 'Major') | (df.Title == 'Col') | (df.Title == 'Capt')) & (df.Survived == 1)].Title.value_counts()


# In[ ]:


df[(df.Title == 'Dr') & (df.Survived == 1)].Title.value_counts()


# In[ ]:


df[(df.Title == 'Rev') & (df.Survived == 1)].Title.value_counts()


# In[ ]:


df[((df.Title == 'Sir') | (df.Title == 'Countess') | (df.Title == 'Jonkheer') | (
    df.Title == 'Don') | (df.Title == 'Lady')) & (df.Survived == 1)].Title.value_counts()


# In[ ]:


df['Mil'] = ((df.Title == 'Major') | (df.Title == 'Col') | (df.Title == 'Capt')).astype(int)
df['Doc'] = (df.Title == 'Dr').astype(int)
df['Rev'] = (df.Title == 'Rev').astype(int)
df['Nob'] = ((df.Title == 'Sir') | (df.Title == 'Countess') | (df.Title == 'Jonkheer') | (df.Title == 'Don') | (df.Title == 'Lady')).astype(int)


# In[ ]:


train[(train.Parch > 0) & (train.Survived == 1)].Parch.value_counts()


# In[ ]:


train[(train.SibSp > 0) & (train.Survived == 1)].SibSp.value_counts()


# In[ ]:


df['FamilySize'] = df.Parch + df.SibSp + 1


# In[ ]:


df[(df.Survived == 1) & (df.FamilySize > 1)].FamilySize.value_counts()


# In[ ]:


df['LargeFam'] = (df.FamilySize >= 5).astype(int)


# In[ ]:


df['SmallFam'] = df.FamilySize.apply(lambda x: x < 5 and x > 1).astype(int)


# In[ ]:


df['Child'] = (df.Age <= 18).astype(int)


# In[ ]:


df.Embarked.fillna('S', inplace=True)


# In[ ]:


fare_mean = dict(df.groupby(['Embarked', 'Pclass']).Fare.mean())
fare_mean


# In[ ]:


df.Fare.fillna(pd.Series([fare_mean[x, y] for x, y in zip(df.Embarked, df.Pclass)]), inplace=True)


# In[ ]:


age_mean = dict(df.groupby('Title')['Age'].mean())
age_mean


# In[ ]:


df.Age.fillna(pd.Series([age_mean[x] for x in df.Title]), inplace = True)


# In[ ]:


df.Age.min()


# In[ ]:


df.Age.max()


# In[ ]:


age_bucket = np.linspace(0, 81, 10)
age_bucket


# In[ ]:


pd.cut(df.Age, age_bucket).value_counts()


# In[ ]:


pd.cut(df.Age, age_bucket).value_counts().plot(kind='bar')


# In[ ]:


pd.cut(df[df.Survived == 1].Age, age_bucket).value_counts().plot(kind = 'bar')


# In[ ]:


df['AgeBucket'] = pd.cut(df.Age, age_bucket)


# In[ ]:


df.groupby('AgeBucket').Survived.mean() * 100


# In[ ]:


import seaborn as sns


# In[ ]:


sns.catplot(x='AgeBucket', y='Survived', data=df, kind = 'point', aspect=2.0, hue='Sex')


# In[ ]:


df['Sex'] = df.Sex.apply(lambda x: 1 if x == 'male' else 0)


# In[ ]:


df.columns


# In[ ]:


df.drop(['Name', 'Cabin', 'Parch', 'SibSp', 'Title', 'PassengerId', 'Ticket', 'AgeBucket'], inplace=True, axis=1)


# In[ ]:


categorical = ['Embarked', 'Pclass']
continuous = ['Age', 'Fare', 'FamilySize']


# In[ ]:


for var in categorical:
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]


# In[ ]:


scaler = MinMaxScaler()
for var in continuous:
    df[var] = df[var].astype('float32')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))


# In[ ]:


df.head()


# In[ ]:


X_train = df[df.Survived.notnull()].drop('Survived', axis=1)
y_train = df[df.Survived.notnull()].Survived
X_test = df[df.Survived.isnull()].drop('Survived', axis=1)


# In[ ]:


def createModel():
    seed(42)
    set_random_seed(42)
    
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# In[ ]:


model = createModel()


# In[ ]:


history = model.fit(X_train, y_train, epochs=155, batch_size=41, validation_split=0.1, verbose=1, shuffle=True)


# In[ ]:


model.fit(X_train, y_train, epochs=155, batch_size=41, verbose=False)


# In[ ]:


test['Survived'] = model.predict(X_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')


# In[ ]:


solution = test[['PassengerId', 'Survived']]
solution.to_csv("titanic_nn.csv", index=False)

