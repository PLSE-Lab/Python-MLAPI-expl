import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as py
from keras.models import Sequential
import tensorflow as tf 
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Flatten



data_input=pd.read_csv('fashion-mnist_train.csv')
''' now we have the input data '''
x_train=data_input.drop(columns=['label'])
y=data_input['label']
x_test=pd.read_csv('fashion-mnist_test.csv')
x_train=np.array(x_train)
x_test_sam=x_test.drop(columns=['label'])
y_test=x_test['label']
x_test_sam=np.array(x_test_sam)
y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)

y=np.array(y)
print(x_train)
print(y)
y=y.reshape(-1,1)

''' now we will be using one hot encoding method to get the desired output'''
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse= False)
encoder1=OneHotEncoder(sparse= False)

y = encoder.fit_transform(y)
y_test=encoder1.fit_transform(y_test)

print(y.shape)
print(x_train.shape)

''' now we will be moving towards our neural network since we have preprocessed out data'''
''' the library that we will be working with is Keras'''
exp1=Sequential()
exp1.add(Dense(1000))
exp1.add(Activation('relu'))
exp1.add(Dense(512))
exp1.add(Activation('relu'))
exp1.add(Dense(10))
exp1.add(Activation('softmax'))


exp1.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


exp1.fit(x_train,y)
exp1.summary()
score=exp1.evaluate(x_test_sam,y_test)


'''
exp2=Sequential()
exp2.add(Flatten(input_shape=x_train.shape[1:]))
exp2.add(Dense(200))
exp2.add(Activation('sigmoid'))
exp2.add(Dense(100))
exp2.add(Activation('sigmoid'))
exp2.add(Dense(10))
exp2.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
exp2.fit(x_train,y)
score=exp2.evaluate(x_test_sam,y_test)
'''