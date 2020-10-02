#!/usr/bin/env python
# coding: utf-8

# **Loading required libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
get_ipython().run_line_magic('matplotlib', 'inline')


# **Reading the training dataset from CSV file and displaying it.**

# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# * **Dividing the Dataset into X and Y**                                                 
# Converting y label to categorical variable that is doing one hot encoding of the labels.
# * 0 = [1,0,0,0,0,0,0,0,0,0]
# * 1 = [0,1,0,0,0,0,0,0,0,0]
# * 2 = [0,0,1,0,0,0,0,0,0,0]
# and so on till 9[](http://)

# In[ ]:


x = df.drop('label',1)
y = df['label']
y = to_categorical(y)


# **Normalizing the data so that processing becomes fast**

# In[ ]:


x = x/255.0


# **As we can see that the each row of the csv represents one image i.e digit in this case. We now want to reshape the data in order to make the one dimensional array of pixels that is one row into 28x28 square image.
# Reshaping the values of X into 42000,28,28,1. 42000 are the number of images there in the training data. 28x28 is the image size and 1 represents the number of channels in the image. ! represent that it is a gray scale image.
# Had it been coloured image we would have mentioned 3 instead of 1.**

# In[ ]:


x = x.values.reshape(42000,28,28,1)


# In[ ]:


x.shape


# **Displaying the image of the data.**

# In[ ]:


plt.imshow(x[3].reshape(28,28),cmap = 'gray')


# Defining the model now
# 1. First Layer is a convolution layer with 32 neurons and having a kernel size of (5,5). The padding is kept same which means we are not reducing the dimensions of the image after convolution. Convolution layer will extract features from the images and help it generalize in order to train the neural netwrok and find optimal values for the kernels. Padding will add an additional layer of zeros on all the sides of the image. This convolution layer is followed by the maxpooling layer. We add maxpooling layer for 2 reasons. First is to reduce the dimensions of the image so that processing becomes faster. Secondly to remove the spartiality from the image which is obtained after the convolution layer.
# 
# 2. Second layer is same as first layer
# 3. Third and forth layer have Convolution layer with 64 neurons each and kernel size of (3,3) and activation function of relu with padding parameter as same followed by maxpooling layer.
# 4. Then we flatten the output from convolution layer which is fed to the dense connected layers having 256 neurons and relu activation function followed by a dropout layer of 20% deactivation of neurons.
# 5. Finally we have the output layer having 10 neurons because those are the number of classes in our y label which uses a softmax function which gives the probability of a particular input belonging to the particular y class.

# In[ ]:


model1 = Sequential()
model1.add(Convolution2D(32,(5,5),activation='relu',input_shape = (28,28,1),padding='same'))
model1.add(MaxPooling2D(2,2))
model1.add(Convolution2D(32,(5,5),activation='relu',padding='same'))
model1.add(MaxPooling2D(2,2))
model1.add(Convolution2D(64,(3,3),activation='relu',padding = 'same'))
model1.add(MaxPooling2D(2,2))
model1.add(Convolution2D(64,(3,3),activation = 'relu',padding = 'same'))
model1.add(Flatten())
model1.add(Dense(256,activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(10,activation = 'softmax'))


# In[ ]:


model1.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model1.summary()


# In[ ]:


history = model1.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)


# In[ ]:


values = history.history
validation_acc = values['val_acc']
training_acc = values['acc']
validation_loss = values['val_loss']
training_loss = values['loss']
epochs = range(10)


# In[ ]:


plt.plot(epochs,validation_acc,label = 'Validation Accuracy')
plt.plot(epochs,training_acc,label = 'Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,validation_loss,label = 'Validation Loss')
plt.plot(epochs,training_loss,label = 'Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


model2 = Sequential()
model2.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model2.add(Convolution2D(32,(3,3),activation='relu'))
model2.add(Convolution2D(64,(3,3),activation='relu'))
model2.add(Convolution2D(64,(3,3),activation = 'relu'))
model2.add(Flatten())
model2.add(Dense(256,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(10,activation = 'softmax'))


# In[ ]:


model2.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model2.summary()


# In[ ]:


history = model2.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)


# In[ ]:


values = history.history
validation_acc = values['val_acc']
training_acc = values['acc']
validation_loss = values['val_loss']
training_loss = values['loss']
epochs = range(10)


# In[ ]:


plt.plot(epochs,validation_acc,label = 'Validation Accuracy')
plt.plot(epochs,training_acc,label = 'Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,validation_loss,label = 'Validation Loss')
plt.plot(epochs,training_loss,label = 'Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


model3 = Sequential()
model3.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model3.add(MaxPooling2D(2,2))
model3.add(Convolution2D(64,(3,3),activation='relu'))
model3.add(Flatten())
model3.add(Dense(128,activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(10,activation = 'softmax'))


# In[ ]:


model3.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model3.summary()


# In[ ]:


history = model3.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)


# In[ ]:


values = history.history
validation_acc = values['val_acc']
training_acc = values['acc']
validation_loss = values['val_loss']
training_loss = values['loss']
epochs = range(10)


# In[ ]:


plt.plot(epochs,validation_acc,label = 'Validation Accuracy')
plt.plot(epochs,training_acc,label = 'Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,validation_loss,label = 'Validation Loss')
plt.plot(epochs,training_loss,label = 'Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


model4 = Sequential()
model4.add(Convolution2D(32,(5,5),activation='tanh',input_shape = (28,28,1),padding='same'))
model4.add(MaxPooling2D(2,2))
model4.add(Convolution2D(32,(5,5),activation='tanh',padding='same'))
model4.add(MaxPooling2D(2,2))
model4.add(Convolution2D(64,(3,3),activation='tanh',padding = 'same'))
model4.add(MaxPooling2D(2,2))
model4.add(Convolution2D(64,(3,3),activation = 'tanh',padding = 'same'))
model4.add(Flatten())
model4.add(Dense(256,activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(10,activation = 'softmax'))


# In[ ]:


model4.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model4.summary()


# In[ ]:


history = model4.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)


# In[ ]:


values = history.history
validation_acc = values['val_acc']
training_acc = values['acc']
validation_loss = values['val_loss']
training_loss = values['loss']
epochs = range(10)


# In[ ]:


plt.plot(epochs,validation_acc,label = 'Validation Accuracy')
plt.plot(epochs,training_acc,label = 'Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,validation_loss,label = 'Validation Loss')
plt.plot(epochs,training_loss,label = 'Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


plot_model(model1,to_file='model1.png',show_layer_names=True,show_shapes=True)
plot_model(model2,to_file='model2.png',show_layer_names=True,show_shapes=True)
plot_model(model3,to_file='model3.png',show_layer_names=True,show_shapes=True)
plot_model(model4,to_file='model4.png',show_layer_names=True,show_shapes=True)


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub.head()


# In[ ]:


test = test/255.0


# In[ ]:


test.shape


# In[ ]:


test = test.values.reshape(28000,28,28,1)


# In[ ]:


predict_model1 = model1.predict_classes(test)


# In[ ]:


predict_model2 = model2.predict_classes(test)


# In[ ]:


predict_model3 = model3.predict_classes(test)


# In[ ]:


predict_model4 = model4.predict_classes(test)


# In[ ]:


final_prediction = 0.25*predict_model1+0.25*predict_model2+0.25*predict_model3+0.25*predict_model4


# In[ ]:


final_prediction = np.round(final_prediction)


# In[ ]:


answer = []
for i in range(len(final_prediction)):
    answer.append(int(final_prediction[i]))


# In[ ]:


predict = pd.Series(answer,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)


# In[ ]:


submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:


predict = pd.Series(predict_model1,name="Label")
submission_1 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission_1.to_csv("submission1.csv",index=False)


# In[ ]:


predict = pd.Series(predict_model2,name="Label")
submission_2 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission_2.to_csv("submission2.csv",index=False)


# In[ ]:


predict = pd.Series(predict_model3,name="Label")
submission_3 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission_3.to_csv("submission3.csv",index=False)


# In[ ]:


predict = pd.Series(predict_model4,name="Label")
submission_4 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission_4.to_csv("submission4.csv",index=False)


# In[ ]:


submission['Label'] = int(submission['Label'])


# In[ ]:




