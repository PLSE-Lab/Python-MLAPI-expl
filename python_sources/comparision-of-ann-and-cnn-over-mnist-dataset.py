#!/usr/bin/env python
# coding: utf-8

# <h3>Comparision of ANN and CNN over MNIST Dataset</h3>
# <h5>This is the original dataset with 60/10 split</h5>
# <h5>The results are at the end of the notebook</h5>

# In[ ]:


#import packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from keras.datasets import mnist

import matplotlib.pyplot as plt
import pandas as pd


# <h1>ANN Model</h1>

# In[ ]:


#load mnist dataset
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
print('images in train dataset : {}'.format(xtrain.shape))
print('images in test dataset  : {}'.format(xtest.shape))


# In[ ]:


plt.imshow(xtrain[0])


# In[ ]:


#preprocessing
num_pixels = xtrain.shape[1]*xtrain.shape[2]

xtrain_ann = xtrain.reshape(xtrain.shape[0], num_pixels).astype('float32')
xtest_ann = xtest.reshape(xtest.shape[0], num_pixels).astype('float32')
print('new shape in train dataset : {}'.format(xtrain_ann.shape))
print('new shape in test dataset  : {}'.format(xtest_ann.shape))


# In[ ]:


xtrain_ann = xtrain_ann/255.0
xtest_ann = xtest_ann/255.0

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

num_classes = ytest.shape[1]
print('shape of output : {}'.format(num_classes))


# In[ ]:


#create model
def classificationModelANN():
  model = Sequential()
  model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(num_classes, activation='relu'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


# In[ ]:


#fit model, evaluate model
modelANN = classificationModelANN()

annhist = modelANN.fit(xtrain_ann, ytrain, validation_data = (xtest_ann, ytest), epochs = 10, verbose = 1)

scoreANN = modelANN.evaluate(xtest_ann, ytest, verbose=0)


# In[ ]:


print('Accuracy: {}%'.format(round(scoreANN[1],3)))


# In[ ]:


#plot acuuracy graph
plt.plot(range(1,11), annhist.history['val_accuracy'], label='valid')
plt.plot(range(1,11), annhist.history['accuracy'], label='train')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.grid()
plt.legend()
plt.show()


# <h1>CNN Model</h1>

# In[ ]:


#load packages
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers import Flatten


# In[ ]:


#preprocessing
xtrain_cnn = xtrain.reshape(xtrain2.shape[0], 28, 28, 1).astype('float32')
xtest_cnn = xtest.reshape(xtest2.shape[0], 28, 28, 1).astype('float32')

xtrain_cnn = xtrain_cnn/255
xtest_cnn = xtest_cnn/255

#ytest, ytrain is same for ann and cnn


# In[ ]:


#create model
def classificationModelCNN():
  model = Sequential()
  model.add(Conv2D(16, (4,4), strides=(1,1), activation ='relu', input_shape=(28,28,1)))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Conv2D(32, (4,4), strides=(1,1), activation ='relu', input_shape=(28,28,1)))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  model.add(Flatten())

  model.add(Dense(100, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  return model


# In[ ]:


#fit model, evaluate model
modelCNN = classificationModelCNN()

cnnhist = modelCNN.fit(
    xtrain_cnn,
    ytrain,
    validation_data = (xtest_cnn, ytest),
    epochs = 10,
    verbose = 1
)

scoreCNN = modelCNN.evaluate(xtest_cnn, ytest, verbose = 0)


# In[ ]:


print('Accuracy: {}%'.format(round(scoreCNN[1],5)))  


# In[ ]:


#plot accuracy graph
plt.plot(range(1,11), cnnhist.history['val_accuracy'], label='valid')
plt.plot(range(1,11), cnnhist.history['accuracy'], label='train')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0.8,1)
plt.grid()
plt.legend()
plt.show()


# In[ ]:


#plot ann vs cnn over validation accuracy
plt.plot(range(1,11), cnnhist.history['val_accuracy'], label='CNN')
plt.plot(range(1,11), annhist.history['val_accuracy'], label='ANN')
plt.title('CNN vs ANN')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0.4,1)
plt.grid()
plt.legend()
plt.show()


# In[ ]:


print('ANN Accuracy : {}'.format(round(scoreANN[1]*100,5)))
print('CNN Accuracy : {}'.format(round(scoreCNN[1]*100,5)))

