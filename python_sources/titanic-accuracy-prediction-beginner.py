import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline

#data retrieval 
train = pd.read_csv('../input/titanic/train.csv')

#data cleaning
train.head()

#CABIN
train.fillna(value = {'Cabin':0},inplace = True)
i = 0
for item in train['Cabin']:
    if item != 0:
        train.loc[i,'Cabin'] = 1
    i = i+1

#AGE
train['Age'].fillna(0,inplace= True)
i = 0
for item in train['Pclass']:
    if item == 1:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 38.233
    elif item == 2:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 29.877
    else:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 25.1406
    i = i+1
    
train.dropna(inplace = True) 
train['Embarked'].isnull().unique()

#modifying data for model
sex = pd.get_dummies(train['Sex'],drop_first= True)
pclass = pd.get_dummies(train['Pclass'],drop_first= True)
emb = pd.get_dummies(train['Embarked'],drop_first= True)
train.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)
train = pd.concat([train,sex,pclass,emb],axis = 1)

#creating the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis = 1), train['Survived'], test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)

#predicting
predictions = log.predict(X_test)

#evaluating the model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#################################################################
train = pd.read_csv('../input/titanic/test.csv')

train.fillna(value = {'Cabin':0},inplace = True)
i = 0
for item in train['Cabin']:
    if item != 0:
        train.loc[i,'Cabin'] = 1
    i = i+1

train['Age'].fillna(0,inplace= True)
i = 0
for item in train['Pclass']:
    if item == 1:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 38.233
    elif item == 2:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 29.877
    else:
        if train.loc[i,'Age'] == 0 :
            train.loc[i,'Age'] = 25.1406
    i = i+1
    
train.dropna(inplace = True) 
train['Embarked'].isnull().unique()


sex = pd.get_dummies(train['Sex'],drop_first= True)
pclass = pd.get_dummies(train['Pclass'],drop_first= True)
emb = pd.get_dummies(train['Embarked'],drop_first= True)
train.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)
train = pd.concat([train,sex,pclass,emb],axis = 1)

predictions2 = pd.DataFrame({'Passenger ID':train['PassengerId'],'Survived':log.predict(train)})
predictions2.set_index('Passenger ID')
predictions2.to_csv('titan.csv')
predictions2