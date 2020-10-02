import pandas as pd
import numpy as np


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

labels = train.label
labels = np.array(labels)

train = train.drop(train.columns[0], axis=1) 
train = np.array(train).reshape(-1,1,784)

print(train.shape)
print(test.shape)


# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')


def onehot(labels):
      n_values = np.max(labels) + 1
      np.eye(n_values)[labels]
    
yy_train = onehot(train)
yy_test = onehot(test)

model = Sequential()
model.add(Dense(output_dim=16,input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dense(input_dim=300,output_dim=300))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=["accuracy"])
model.fit(x_train,yy_train,nb_epoch=5,batch_size= 32)
x = model.predict(x_test)
loss,accuracy = model.evaluate(x_test,yy_test)
