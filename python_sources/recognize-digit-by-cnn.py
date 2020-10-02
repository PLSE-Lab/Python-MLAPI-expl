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

import tensorflow as tf
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten
from keras.layers import MaxPool2D,MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
#print(X_train)


x_train = X_train.drop('label',axis=1).values
#print(x_train)
X_test=X_test.values

y_train = X_train['label'].values
#print(y_train)

X_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
print(x_train.shape[0])

#x_train  = x_train/x_train.max()#normalize
y_train = keras.utils.to_categorical(y_train,10)

model=Sequential()
input_shape = (28,28,1)
model.add(Conv2D(20,kernel_size=(5,5),strides=1,padding='valid',activation='relu',input_shape=input_shape,kernel_initializer='uniform'))
print('hello')
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(50,kernel_size=(5,5),strides=(1,1),activation='relu',padding='valid',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.fit(x=X_train,y=y_train,batch_size=256,epochs=25,verbose=1)


y_predict = model.predict(X_test[6].reshape(1,28,28,1))
print(y_predict)


plt.imshow(X_test[6].reshape(28,28),cmap='gray')
plt.show()

model.summary()

