#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Data

# In[ ]:


train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()


# # Data Preparation
# 
# Split the data into lables and images. Convert the pixels into an array with 28x28 matrix of pixel values

# In[ ]:


target=train['label']
train=train.drop('label',axis=1)
train=train.values.reshape(-1,28,28)
train=train/255
plt.imshow(train[1,:,:])
print(target[1])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train ,  x_test ,y_train, y_test= train_test_split(train,target,test_size=0.2,random_state=7)
print(x_train.shape)
print(x_test.shape)


# In[ ]:


test=test.values.reshape(-1,28,28)
test=test/255
test.shape


# # Define Model Architecture
# 

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


def get_model(input_shape,drop_rate):
  model=Sequential([
                    Conv2D(32,(3,3),input_shape=input_shape,padding='SAME',activation='relu'),
                    Conv2D(32,(3,3),padding='SAME',activation='relu'),
                    MaxPool2D((2,2)),
                    Dropout(drop_rate),
                    BatchNormalization(),
                    Conv2D(64,(3,3),padding='SAME',activation='relu'),
                    Conv2D(64,(3,3),padding='SAME',activation='relu'),
                    MaxPool2D((2,2)),
                    Dropout(drop_rate),
                    BatchNormalization(),
                    Flatten(),
                    Dense(128,activation='relu'),
                    Dense(10,activation='softmax')
  ])
  return model


# I have used two 3x3 convolutional layers with 32 filters each with padding so that image size doesnt vary. This is followed by max pool so that we reduce the image size making training easier and reduce overfiting. Dropout layer and batch normalization will further reduce overfitting.
# Then we repeat two more 3x3 convolutional layers with 64 filters. Finally the flatten layer converts the matrix to a one dimensional array which then goes through dense layers and finally a softmax activation which classifies it as a digit. 
# 
# This achitecture can be further modified by adding more layers or altering the number of filters. There is always a better combination that would improve the results.

# In[ ]:


mod=get_model([28,28,1],0.3)
mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# model is compiled and a drop out rate of 0.3 is used.
# 
# # Data Augmentation
# 
# Now we shall use data augmentation to increase the size of test data.
# What happens in augmentation is the function randomly either rotates, zooms or shears the original image to create new images that are similar but different to the original. This way the algorithm gets many more training examples and can adapt to variations in the image. For example if a munber is tilted by 15 degrees its still the same number. So through data augmentation the model is better equiped to handle changes in the test data. So it can greatly reduce overfitting and make the model much more robust. This is a very good technique to use when there is less data available
# 
# The advantages of ImageDataGenerator is that it does not take any storage space the images are generated one by on as it is called by the fit fucnction. This way we can train the model on unlimited number of variations without worrying about storage constraints.
# 
# Here we set a rotation range of 20 degrees sothat the image is rotated only +/- 20 degrees similarly shear is selt to 0.45 and zoom range is set. The function pics a random number within the range and applies a particular transform at random to get a slightly varied image

# In[ ]:


data_aug=ImageDataGenerator(rotation_range=20,shear_range=0.4,zoom_range=[0.75,1.3])
history=mod.fit_generator(data_aug.flow(x_train[...,np.newaxis],y_train,batch_size=100),epochs=500,
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15),
                                     tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=15)],validation_data=(x_test[...,np.newaxis],y_test))


# In[ ]:


pred=mod.predict(test[...,np.newaxis])
labels=np.argmax(pred,axis=1)
results=pd.DataFrame({'label':labels},index=range(1,test.shape[0]+1))


# In[ ]:


#results.to_csv('result3.csv')


# In[ ]:


hist=pd.DataFrame(history.history)
hist.plot(y='accuracy')
plt.plot(hist['val_accuracy'])


# In[ ]:


hist.plot(y='loss')
plt.plot(hist['val_loss'])


# # Conclusion
# 
# Using data augmentation we can see that we have achieved good accuracy and overfitting is reduced
