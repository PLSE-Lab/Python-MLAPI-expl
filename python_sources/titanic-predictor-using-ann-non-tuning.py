import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[:, [2,4,5,6,7,9,11]].values
y = train.iloc[:, 1].values

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:,2:3] = imputer.transform(X[:, 2:3])


label_encoder_X2 = LabelEncoder()
X[:, 1] = label_encoder_X2.fit_transform(X[:,1])
label_encoder_X3 = LabelEncoder()
X[:, -1] = label_encoder_X3.fit_transform(X[:,-1])

onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, :-1]

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size = 0.2)
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.fit_transform(test_X)

model = Sequential()

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation ='relu', input_dim = 9))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer= 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_X,train_y,batch_size = 10, epochs = 100)

y_pred = model.predict(test_X)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(test_y, y_pred)


