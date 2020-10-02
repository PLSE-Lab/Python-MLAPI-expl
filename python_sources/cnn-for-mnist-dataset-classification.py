#!/usr/bin/env python
# coding: utf-8

# # Classification of the MNIST dataset using convolutional neural networks (CNNs)
# 
# This jupyter notebook shows an example of application of **convolutional neural networks (CNNs)** to perform image classification. In particular, this powerful machine learning tool is applied to the famous MNIST dataset, which contains handwritten digits. The CNN implemented here allows **reaching a 99% classification accuracy**.

# In[ ]:


# import the needed libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd


# ## Importing the dataset
# The images have dimension 28x28 pixels and they are in gray scale (i.e., each pixel takes values in [0,255]).

# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


# convert to numpy matrices
train_mat=train.to_numpy()
test_mat=test.to_numpy()


# Some preprocessing is needed. In particular, we need to:
# 
# 1. **reshape the images matrices** to have a tensor of dimension 28x28x1 that will be the input of the CNN;
# 2. **normalize the tensor elements** to take values in [0,1];
# 3. map the label value to a **one-hot-encoding vector** that will be the output of the CNN.

# In[ ]:


# extract labels from training dataset
y_train=train_mat[:,0]
x_train=train_mat[:,1:]

x_test=test_mat

# reshape arrays to have 2D images
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

# data normalization
x_train=x_train/255
x_test=x_test/255

# labels to categories
y_train = to_categorical(y_train)


# The figure shows the first 16 pictures in the training dataset.

# In[ ]:


for i in range(0,16):
  plt.subplot(4,4,i+1)
  plt.imshow(x_train[i,:,:,0])
  plt.axis('off');


# Before proceding with the design of the CNN, the split the training data into two sets for training and validation.

# In[ ]:


from sklearn.model_selection import train_test_split

# Split the train and the validation set for the fitting
X_t, X_v, Y_t, Y_v = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)


# ## Design of the convolutional neural network
# In the following, we design the CNN and we show a summary of its architecture.
# 

# In[ ]:


model = Sequential()
model.add(Conv2D(32,(3,3),strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(64,(3,3),strides=1,padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.summary()


# The CNN is then compiled. We use the **stochastic gradient descent** optimizer and the goal is to minimize the categorical crossentropy.

# In[ ]:


model.compile(optimizer='sgd',metrics=['accuracy'],loss='categorical_crossentropy')


# ## Training of the CNN model
# The CNN is then trained for 200 epochs using a batch of size 100.

# In[ ]:


history=model.fit(X_t,Y_t,validation_data=(X_v,Y_v),batch_size=100,epochs=100,verbose=False)


# In[ ]:


model.evaluate(x_train,y_train)


# The following two graphs show how the accuracy and loss measures evolves while training the network.
# 
# 

# In[ ]:


# Plot training & validation accuracy values
plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.subplot(1,2,2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right');


# ## Predictions on the test set
# The trained model is run on the test dataset.

# In[ ]:


predictions=model.predict(x_test)
labels=np.argmax(predictions,axis=1)
results = pd.Series(labels,name="Label")
final = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)


# Finally, the predictions are exported to a CSV file.

# In[ ]:


final.to_csv("submission.csv",index=False)

