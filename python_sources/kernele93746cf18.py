from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
network=models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3),activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64,activation='relu'))
network.add(layers.Dense(10,activation='softmax'))
network.summary()
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
train_data=np.loadtxt(r'../input/train.csv',delimiter=',',skiprows=1)
test_images=np.loadtxt(r'../input/test.csv',delimiter=',',skiprows=1)
train_images=train_data[:,1:].reshape((42000,28,28,1)).astype('float32')/255
train_labels=to_categorical(train_data[:,0],10)
test_images=test_images.reshape((28000,28,28,1)).astype('float32')/255
# val_images=train_data[30000:,1:].astype('float32')/255
# val_labels=to_categorical(train_data[30000:,0],10)
hi=network.fit(train_images,train_labels,epochs=20,batch_size=64)
test_labels=network.predict_classes(test_images)
a=list(np.arange(len(test_labels))+1)
data={'ImageId':a,'Label':list(test_labels)}
df=pd.DataFrame(data)
df.to_csv('test_predict1.csv',index=0)