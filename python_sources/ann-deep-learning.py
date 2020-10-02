# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#import a dataset
dataset = pd.read_csv('../input/heart.csv')
X = dataset.iloc[:, 0:13].values #takinf all the culums -1 and taking all the rows
y = dataset.iloc[:, 13].values 

#spliting / making trainingset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.20, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Starting the ANN algorithem 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initial the ANN
classifier  = Sequential()

#adding the input layer and the first hidden layer 
classifier.add(Dense(output_dim = 7,init = 'uniform' , activation = 'relu' , input_dim = 13 ))

#adding headen layers (using rectifier activation function // output dim as average of input and output parameters (11+1)/2 
classifier.add(Dense(output_dim = 7,init = 'uniform' , activation = 'relu' ))

#adding the output layer(using sigmoid activation function // if the output needed more than one category then use softmax activation function) 
classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))

#combile the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

#fitting ANN model to training set
classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)

#create the classifier algorithem 

#predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
 
#making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.