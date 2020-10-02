#!/usr/bin/env python
# coding: utf-8

# # PyData Kaunas
# * Self-Driving Robots
# * Digit Recognition Workshop
# * v 1.1

# In[ ]:


import tensorflow as tf
tf.test.gpu_device_name()


# # look at data
# 
# * l1_1 https://www.kaggle.com/c/digit-recognizer/data
# * l1_2 https://docs.python.org/3/library/os.path.html#os.path.join
# * l1_3 https://pandas.pydata.org/pandas-docs/stable/10min.html
# * l1_4 https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

# In[ ]:


import os
pth=os.path.join('..','input','train.csv')

import pandas as pd
traindf=pd.read_csv(pth)
traindf.head()


# In[ ]:


pth=os.path.join('..','input','test.csv')
testdf=pd.read_csv(pth)
testdf.head()


# # create validation set
# * l2_1 https://keras.io/utils/#to_categorical
# * l2_2 http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# In[ ]:


trainYl=traindf['label']
trainYl.head()


# In[ ]:


from keras.utils import to_categorical
trainY=to_categorical(trainYl,10)
trainY


# In[ ]:


traindf.head()


# In[ ]:


#trainX=traindf.drop(columns=['label']).as_matrix()
trainX=traindf.drop(['label'],axis=1).as_matrix()
trainX


# In[ ]:


from sklearn.model_selection import train_test_split
trainX,valX,trainY,valY=train_test_split(trainX,trainY,test_size=.1)


# # create linear model
# * l3_1 https://keras.io/getting-started/sequential-model-guide/
# * l3_2 https://keras.io/models/sequential/
# * l3_3 https://en.wikipedia.org/wiki/Feedforward_neural_network
# * l3_4 https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L743
# * l3_5 https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py 

# In[ ]:


trainX.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=784,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))


# In[ ]:


test=testdf.as_matrix()
preds=model.predict(test)
print(preds.shape)
preds


# In[ ]:


import numpy as np
preds=np.argmax(preds,axis=1).tolist()
preds


# In[ ]:


pth=os.path.join('..','input','sample_submission.csv')
#sample=pd.read_csv(pth)
#sample.head()


# In[ ]:


idx=[i for i in range(1,len(preds)+1)]
predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
predsdf.head()


# In[ ]:


predsdf.to_csv('sub.csv',index=False)


# # Multi-layer perceptron
# 

# model = Sequential()
# model.add(Dense(124, input_dim=784,activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))

# preds=model.predict(test)
# preds=np.argmax(preds,axis=1).tolist()
# idx=[i for i in range(1,len(preds)+1)]
# predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
# predsdf.to_csv('sub.csv',index=False)

# # "deep network"

# model = Sequential()
# model.add(Dense(1400, input_dim=784,activation='relu'))
# model.add(Dense(700, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))

# model = Sequential()
# model.add(Dense(100, input_dim=784,activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))

# preds=model.predict(test)
# preds=np.argmax(preds,axis=1).tolist()
# idx=[i for i in range(1,len(preds)+1)]
# predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
# predsdf.to_csv('sub.csv',index=False)

# model = Sequential()
# model.add(Dense(100, input_dim=784,activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX,trainY,epochs=8,validation_data=(valX,valY))

# preds=model.predict(test)
# preds=np.argmax(preds,axis=1).tolist()
# idx=[i for i in range(1,len(preds)+1)]
# predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
# predsdf.to_csv('sub.csv',index=False)

# # VGG16
# * l4_1 https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# def normInput(x):return (x-meanPx)/stdPx
# meanPx=trainX.mean().astype(np.float32)
# stdPx=trainX.std().astype(np.float32)
# 
# trainX2=trainX.reshape(-1,28,28,1)
# valX2=valX.reshape(-1,28,28,1)
# test2=test.reshape(-1,28,28,1)
# print(trainX2.shape,valX2.shape,test2.shape)

# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.imshow(trainX2[0][:,:,0])

# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Lambda,Flatten
# 
# model=Sequential()
# model.add(Lambda(normInput,input_shape=(28,28,1)))
# model.add(Conv2D(32,3,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(10,activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX2,trainY,epochs=8,validation_data=(valX2,valY))

# In[ ]:


model.summary()


# In[ ]:


model.__dict__


# In[ ]:


model.layers[1].__dict__


# In[ ]:


model.layers[1].output_shape


# In[ ]:


model.layers[3].output_shape


# preds=model.predict(test)
# preds=np.argmax(preds,axis=1).tolist()
# idx=[i for i in range(1,len(preds)+1)]
# predsdf=pd.DataFrame(data={'ImageId':idx,'Label':preds})
# predsdf.to_csv('sub.csv',index=False)

# from keras.layers.normalization import BatchNormalization
# from keras.layers import Dropout
# model = Sequential()
# model.add(Lambda(normInput,input_shape=(28,28,1)))
# model.add(Conv2D(32,3,activation='relu'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization(axis=1))
# model.add(Conv2D(64,3,activation='relu'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization(axis=1))
# model.add(Conv2D(64,3,activation='relu'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization(axis=1))
# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(512,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(10,activation='softmax'))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(trainX2,trainY,epochs=8,validation_data=(valX2,valY))

# # data augumentation
# * https://keras.io/preprocessing/image/
# * https://keras.io/models/sequential/

# from keras.preprocessing.image import ImageDataGenerator
# idg=ImageDataGenerator(rotation_range=20,shear_range=.2,zoom_range=.2)
# plt.imshow(idg.flow(trainX2,trainY)[0][0][0][:,:,0]) 

# model.fit_generator(idg.flow(trainX2,trainY),steps_per_epoch=500,validation_data=(valX2,valY))

# # pseudo labels

# preds=model.predict(test2)
# predsV=model.predict(valX2)
# trainXp=np.concatenate((trainX2,test2,valX2))
# trainYp=np.concatenate((trainY,preds,predsV))
# model.fit(trainXp,trainYp,epochs=1,validation_data=(valX2,valY))

# preds=model.predict(test2)
# preds=np.argmax(preds,axis=1).tolist()
# idx=[i for i in range(1,len(preds)+1)]
# preds=pd.DataFrame(data={'ImageId':idx,'Label':preds})
# preds.to_csv('sub.csv',index=False)

# In[ ]:




