# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace= True)
print(train.columns)

train['Age'].fillna(train['Age'].median(), inplace = True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
train['Fare'].fillna(train['Fare'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Sex'] = le.fit_transform(test['Sex'])
test['Embarked'] = le.fit_transform(test['Embarked'])

scaler = MinMaxScaler()
train[['Age','Fare', 'SibSp','Parch']] = scaler.fit_transform(train[['Age','Fare', 'SibSp','Parch']])
test[['Age','Fare', 'SibSp','Parch']] = scaler.fit_transform(test[['Age','Fare','SibSp','Parch']])

X_train = train.drop('Survived',axis = 1).values
y_train = train['Survived'].values
y_train_onehot = pd.get_dummies(train['Survived']).values
X_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1).values


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
output_rf = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rf_predict})
output_rf.to_csv('submission_rf.csv', index=False)


xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
output_xgb = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgb_preds})
output_xgb.to_csv('submission_xgb.csv', index=False)


model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train_onehot, epochs=300)

y_pred = model.predict(X_test)
results = np.argmax(y_pred,axis = 1)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': results})
output.to_csv('submission_nn.csv', index=False)
