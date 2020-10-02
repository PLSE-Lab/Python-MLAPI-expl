import pandas as pd
import numpy as np
import tensorflow  as tf
from keras import *
from keras.layers.convolutional import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train=pd.read_csv(r"../input/train.csv")
x_test=pd.read_csv(r"../input/test.csv")
train,validate=train_test_split(train,test_size=0.05)
x_train=train.iloc[:,1:]
y_train=train.iloc[:,[0]]
x_validate=validate.iloc[:,1:]
y_validate=validate.iloc[:,[0]]
x_train=x_train.values.reshape(-1,28,28,1)/255;
x_validate=x_validate.values.reshape(-1,28,28,1)/255;
x_test=x_test.values.reshape(-1,28,28,1)/255;
le=OneHotEncoder(dtype=np.int8)
le.fit(y_train.values)
y_train=le.transform(y_train)
le.fit(y_validate)
y_validate=le.transform(y_validate)

cnn=Sequential()
cnn.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',strides=1,activation ='relu', input_shape = (28,28,1)))
cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
cnn.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',strides=1,activation ='relu', input_shape = (28,28,1)))
cnn.add(MaxPooling2D(pool_size=(4,4),strides=(4,4)))
cnn.add(layers.core.Flatten())
cnn.add(layers.core.Dropout(0.5))
cnn.add(layers.core.Dense(10, activation = "softmax"))
optimizer = optimizers.Adam()
cnn.compile(optimizer = optimizer , loss = losses.categorical_crossentropy, metrics=["accuracy"])

history=cnn.fit(x=x_train,y=y_train,batch_size=32,epochs=1,verbose=1,validation_data=(x_validate,y_validate),shuffle=True)
predicts=cnn.predict(x_test).argmax(axis=1)
df=pd.DataFrame({'ImageId':np.arange(1,len(x_test)+1),"Label":predicts})
df.to_csv('my_submission.csv',index=False)


predicts=cnn.predict(x_test).argmax(axis=1)

