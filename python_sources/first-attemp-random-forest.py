"""
Created on Sat Jun 03 20:45:36 2017

@author: Dandy hehe
"""

import pandas as pd
import numpy as np
import time

start_time = time.clock()
dtrain = '../input/train.csv'
dtest = '../input/test.csv'

pd_train = pd.read_csv(dtrain)
pd_test = pd.read_csv(dtest)

column_name = list(pd_train)
del column_name[1]
del column_name[-2]

#delete cabin feature, too much missing value
del pd_train['Cabin'], pd_test['Cabin'], pd_train['Ticket'], pd_test['Ticket']

#simplify into checking whether someone go with family or not
pd_train['Fam'] = (pd_train['SibSp']+pd_train['Parch'])>0
pd_test['Fam'] = (pd_test['SibSp']+pd_test['Parch'])>0
pd_train = pd_train.drop(['SibSp','Parch'], axis = 1)
pd_test = pd_test.drop(['SibSp','Parch'], axis = 1)

#Impute missing values with mean and median
pd_train['Age']= pd_train['Age'].fillna(28)
pd_test['Age']= pd_test['Age'].fillna(28)
pd_train['Embarked'] = pd_train['Embarked'].fillna('S')
pd_test['Fare'] = pd_test['Fare'].fillna(32.2042079685746)

#Transform embarked to numerical
pd_train['Embarked'] = pd_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
pd_test['Embarked'] = pd_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Map sex feature, somehow got ValueError: Cannot convert NA to integer
#pd_train['Sex'] = pd_train['Sex'].map( {'Male': 0, 'Female': 1} ).astype(int)
#pd_test['Sex'] = pd_test['Sex'].map( {'Male': 0, 'Female': 1} ).astype(int)

for i in range(len(pd_train['Sex'])):
    if pd_train['Sex'][i]=='male':
        pd_train['Sex'][i] = 0
    else:
        pd_train['Sex'][i] = 1
for i in range(len(pd_test['Sex'])):
    if pd_test['Sex'][i]=='male':
        pd_test['Sex'][i] = 0
    else:
        pd_test['Sex'][i] = 1

X_train = pd_train.drop(['Survived', 'Name', 'PassengerId'], axis=1).copy()
Y_train = pd_train['Survived'].copy()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
X_test = pd_test.drop(['Name', 'PassengerId'], axis=1).copy()
Y_test = rf.predict(X_test)

pred_out = open('predicted1.csv', 'w')
pred_out.truncate()
pred_out.write('PassengerId,Survived\n')
for i in range(len(Y_test)):
    pred_out.write(str(pd_test['PassengerId'][i])+','+str(Y_test[i])+'\n')
pred_out.close()
