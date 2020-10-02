# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_dataset = pd.read_csv("../input/train.csv", index_col = 'PassengerId')
test_dataset = pd.read_csv("../input/test.csv", index_col = 'PassengerId')

# Exploring datasets
train_dataset.head()
test_dataset.head()

# Making training and test sets on basis of analysis
X_train = train_dataset.iloc[:, [1, 3, 4, 5, 6, 10]]
y_train = train_dataset.iloc[:, 0]
X_test = test_dataset.iloc[:, [0, 2, 3, 4, 5, 9]]

import tensorflow as tf
print(tf.__version__)

X_train['Embarked'] = X_train['Embarked'].astype(str)
X_test['Embarked'] = X_test['Embarked'].astype(str)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X_train[['Age']])
X_train[['Age']] = imputer.transform(X_train[['Age']])
imputer1 = Imputer()
imputer1 = imputer1.fit(X_test[['Age']])
X_test[['Age']] = imputer1.transform(X_test[['Age']])

# Taking care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label1 = LabelEncoder()
X_train['Pclass'] = label1.fit_transform(X_train['Pclass'])
label2 = LabelEncoder()
X_train['Sex'] = label2.fit_transform(X_train['Sex'])
label3 = LabelEncoder()
X_train['Embarked'] = label3.fit_transform(X_train['Embarked'])
onehot1 = OneHotEncoder(categorical_features = [[0, 1, 3, 4, 5]])
X_train = onehot1.fit_transform(X_train).toarray()
label4 = LabelEncoder()
X_test['Pclass'] = label4.fit_transform(X_test['Pclass'])
label5 = LabelEncoder()
X_test['Sex'] = label5.fit_transform(X_test['Sex'])
label6 = LabelEncoder()
X_test['Embarked'] = label6.fit_transform(X_test['Embarked'])
onehot2 = OneHotEncoder(categorical_features = [[0, 1, 3, 4, 5]])
X_test = onehot2.fit_transform(X_test).toarray()

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(576, input_shape = (24,), activation = tf.nn.relu),
    tf.keras.layers.Dense(576, activation = tf.nn.relu),
    tf.keras.layers.Dense(576, activation = tf.nn.relu),
    tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
])

# Compiling the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting model to training set
model.fit(X_train, y_train, batch_size = 33, epochs = 10)

# Predicting survival for test set
y_pred = model.predict(X_test)
for i in range(y_pred.shape[0]):
    if(y_pred[i] > 0.5):
        y_pred[i] = 1
    else:
        y_pred[i] = 0
y_pred = y_pred.astype(int)

# Submission CSV file
submit = test_dataset
submit['Survived'] = y_pred
submit.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1, inplace = True)
submit.to_csv('titanic1.csv', index = True)