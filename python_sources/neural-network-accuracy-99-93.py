# import my libraries

import pandas as pd
import numpy as np

# read data

data = pd.read_csv('../input/creditcard.csv')

# define features and target

X = data.drop('Class',axis=1)
y = data['Class']

# scale my features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

# divide in trai and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# create neural network

# import more libraries for neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# create the nueral network

clas = Sequential()
clas.add(Dense(units=9,kernel_initializer='uniform',activation='relu',input_dim=30))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=9,kernel_initializer='uniform',activation='relu',input_dim=30))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
clas.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train neural netork

clas.fit(X_train,y_train,batch_size=10,epochs=5,verbose=2)

# evalute model

scores = clas.evaluate(X_test,y_test,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])