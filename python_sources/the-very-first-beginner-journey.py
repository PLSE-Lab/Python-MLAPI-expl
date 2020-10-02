# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:29:37 2019

@author: walter
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# importando dados de teste
dadosg = pd.read_csv('../input/titanic/test.csv')

# Missing Data: Age, Cabin
dg = dadosg
dg['Survived_0'] = 0
print(list(dg.columns.values))
dg = dg[['PassengerId','Survived_0', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
#dg.Age.plot(kind='hist')
#dg.Fare.plot(kind='hist')
dg['Age'] = dg['Age'].fillna(np.mean(dadosg.iloc[:, 5]))
describeg = dg.describe()
dg.loc[dg.Age == 24.136300679929494, 'Age_null'] = 1
dg.loc[dg.Age != 24.136300679929494, 'Age_null'] = 0
dg['Cabins'] = dg.iloc[:, 10].str.split(' ')
dg['Cabin'] = dg['Cabins'].str.get(0)
dg.loc[pd.isnull(dg.Cabin) == True, 'Cabin_null'] = 1
dg.loc[pd.isnull(dg.Cabin) != True, 'Cabin_null'] = 0
dg['Embarked'] = dg['Embarked'].fillna('S')

# Dummy sex
Encoder = LabelEncoder()
dg['Sex'] = Encoder.fit_transform(dg.iloc[:, 4].values)

# organizing columns 
print(list(dg.columns.values))
dg = dg[['Survived_0', 'Sex', 'Age', 'Fare', 'PassengerId', 'Name', 'Ticket', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Cabins', 'Cabin_null', 'Age_null', 'Embarked']]

# Dummy variables

describe2 = dg.describe()
dg['Title'] = (dg['Name'].str.split(', ').str.get(1).str.split('.').str.get(0))#.strip()
dg.loc[dg.Title == 'the Countess', 'Title'] = 'Countess'
dg['Title'] = dg['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dg['Title'] = dg['Title'].replace('Mlle', 'Miss')
dg['Title'] = dg['Title'].replace('Ms', 'Miss')
dg['Title'] = dg['Title'].replace('Mme', 'Mrs')
dg['Title'] = Encoder.fit_transform(dg.iloc[:,15].values).astype(float)
dg.dtypes
dg.drop(['Name', 'Ticket', 'Cabin', 'Cabins'], axis = 1,inplace = True)
dg = pd.get_dummies(dg, columns=['Pclass', 'Embarked'], drop_first = True)
dg.iloc[:,10:] = dg.iloc[:,10:].astype(float)
dg['Fare'] = dg['Fare'].fillna(0)
#dg['CategoricalFare'] = pd.qcut(dg['Fare'], 5)
#g['CategoricalAge'] = pd.qcut(dg['Age'], 5)
    # Mapping Fare
dg.loc[ dg['Fare'] <= 7.91, 'Fare']  					    = 0
dg.loc[(dg['Fare'] > 7.91) & (dg['Fare'] <= 14.454), 'Fare'] = 1
dg.loc[(dg['Fare'] > 14.454) & (dg['Fare'] <= 31), 'Fare']   = 2
dg.loc[ dg['Fare'] > 31, 'Fare'] 						    = 3
dg['Fare'] = dg['Fare'].astype(int)
    
    # Mapping Age
dg.loc[ dg['Age'] <= 16, 'Age'] 					   = 0
dg.loc[(dg['Age'] > 16) & (dg['Age'] <= 32), 'Age'] = 1
dg.loc[(dg['Age'] > 32) & (dg['Age'] <= 48), 'Age'] = 2
dg.loc[(dg['Age'] > 48) & (dg['Age'] <= 64), 'Age'] = 3
dg.loc[ dg['Age'] > 64, 'Age']                      = 4
dg['a'] = dg['SibSp'] + dg['Parch'] +1
dg['CategoricalFamily'] = pd.cut(dg['a'],bins=[0,1,2,4, float('11')])
dg.drop(['a','Survived_0','Cabin_null','Age_null','SibSp', 'Parch'], axis = 1,inplace = True)
dg = pd.get_dummies(dg, columns=['Age','Fare','Title','CategoricalFamily'], drop_first = True)
dg.iloc[:,:] = dg.iloc[:,:].astype(int)


#####################################################################  df


# importando dados de treino
dados = pd.read_csv('../input/titanic/train.csv')
describe = dados.describe()

# Missing Data: Age, Cabin
df = dados
#df.Age.plot(kind='hist')
#df.Fare.plot(kind='hist')
df['Age'] = df['Age'].fillna(np.mean(dados.iloc[:, 5]))
df.loc[df.Age == 29.69911764705882, 'Age_null'] = 1
df.loc[df.Age != 29.69911764705882, 'Age_null'] = 0
df['Cabins'] = df.iloc[:, 10].str.split(' ')
df['Cabin'] = df['Cabins'].str.get(0)
df.loc[pd.isnull(df.Cabin) == True, 'Cabin_null'] = 1
df.loc[pd.isnull(df.Cabin) != True, 'Cabin_null'] = 0
df['Embarked'] = df['Embarked'].fillna('S')

# Dummy sex
Encoder = LabelEncoder()
df['Sex'] = Encoder.fit_transform(dados.iloc[:, 4].values)

# organizing columns 
print(list(df.columns.values))
df = df[['Survived', 'Sex', 'Age', 'Fare', 'PassengerId', 'Name', 'Ticket', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Cabins', 'Cabin_null', 'Age_null', 'Embarked']]

# Dummy variables

describe2 = df.describe()
df['Title'] = (df['Name'].str.split(', ').str.get(1).str.split('.').str.get(0))#.strip()
df.loc[df.Title == 'the Countess', 'Title'] = 'Countess'
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = Encoder.fit_transform(df.iloc[:,15].values).astype(float)
df.dtypes
df.drop(['Name', 'Ticket', 'Cabin', 'Cabins'], axis = 1,inplace = True)
df = pd.get_dummies(df, columns=['Pclass', 'Embarked'], drop_first = True)
df.iloc[:,10:] = df.iloc[:,10:].astype(float)
df['Fare'] = df['Fare'].fillna(0)
#df['CategoricalFare'] = pd.qcut(df['Fare'], 5)
#df['CategoricalAge'] = pd.qcut(df['Age'], 5)
    # Mapping Fare
df.loc[ df['Fare'] <= 7.91, 'Fare'] 						    = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
df.loc[ df['Fare'] > 31, 'Fare'] 						    = 3
df['Fare'] = df['Fare'].astype(int)
    
    # Mapping Age
df.loc[ df['Age'] <= 16, 'Age'] 					   = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[ df['Age'] > 64, 'Age']                      = 4

df['a'] = df['SibSp'] + df['Parch'] +1
df['CategoricalFamily'] = pd.cut(df['a'],bins=[0,1,2,4, float('11')])
df.drop(['a','Cabin_null','Age_null','SibSp', 'Parch'], axis = 1,inplace = True)
df = pd.get_dummies(df, columns=['Age','Fare','Title','CategoricalFamily'], drop_first = True)
df.iloc[:,:] = df.iloc[:,:].astype(int)
print(df.columns)
print(dg.columns)
df = df[['Survived', 'Sex', 'Pclass_2', 'Pclass_3', 'Embarked_Q',
       'Embarked_S', 'Age_1.0', 'Age_2.0', 'Age_3.0', 'Age_4.0', 'Fare_1',
       'Fare_2', 'Fare_3', 'Title_1.0', 'Title_2.0', 'Title_3.0', 'Title_4.0',
       'CategoricalFamily_(1.0, 2.0]', 'CategoricalFamily_(2.0, 4.0]',
       'CategoricalFamily_(4.0, 11.0]']]
dg = dg[['PassengerId', 'Sex', 'Pclass_2', 'Pclass_3', 'Embarked_Q',
       'Embarked_S', 'Age_1.0', 'Age_2.0', 'Age_3.0', 'Age_4.0', 'Fare_1',
       'Fare_2', 'Fare_3', 'Title_1.0', 'Title_2.0', 'Title_3.0', 'Title_4.0',
       'CategoricalFamily_(1.0, 2.0]', 'CategoricalFamily_(2.0, 4.0]',
       'CategoricalFamily_(4.0, 11.0]']]

x_train = df.iloc[:,1:]
y_train =  df.iloc[:,0]
x_test = dg.iloc[:,1:]

# Model SVC
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)
dg['Survived'] = classifier.predict(x_test)


# Exportando Kaggle Submission
final = dg[['PassengerId', 'Survived']]
final.iloc[:,:] = final.iloc[:,:].astype(int)
final.to_csv('Submission_SVC.csv', index=0)