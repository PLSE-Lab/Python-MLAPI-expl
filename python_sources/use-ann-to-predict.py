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
# Importing the dataset
train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')
test_Result = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

#Choose Relevent information
X_train = train_dataset[['Pclass','Sex','Age','Embarked']]
Y_train = train_dataset[['Survived']]

X_test = test_dataset[['Pclass','Sex','Age','Embarked']]
Y_test = test_Result[['Survived']]

#Check Null Values
X_train.isnull().sum()
#Replace Null Values
X_train['Embarked']=X_train['Embarked'].replace(np.nan, 'na', regex=True)
X_train['Age']=X_train['Age'].fillna((X_train['Age'].mean()),inplace = False)

#Check Null Values
X_test.isnull().sum()
#Replace Null Values
X_test['Age']=X_test['Age'].fillna((X_test['Age'].mean()),inplace = False)
#Convert to array
X_train = X_train.values
Y_train = Y_train.values
X_test  = X_test.values
Y_test  = Y_test.values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
X_train[:, 1] = label.fit_transform(X_train[:, 1])
X_train[:, 3] = label.fit_transform(X_train[:, 3])
X_test[:, 1] = label.fit_transform(X_test[:, 1])
X_test[:, 3] = label.fit_transform(X_test[:, 3])


onehotencoder = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder.fit_transform(X_train).toarray()
# remove one column to ignore trap 
X_train = X_train[:, :6]

X_test = onehotencoder.fit_transform(X_test).toarray()

#Normalize Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Build Neural Network using Keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.python.keras import Sequential

Model = Sequential()

# Add Layer

Model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))

Model.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer
Model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
Model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

Model.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)



# Predicting the Test set results
y_pred = Model.predict(X_test)
y_pred = (y_pred > 0.5)