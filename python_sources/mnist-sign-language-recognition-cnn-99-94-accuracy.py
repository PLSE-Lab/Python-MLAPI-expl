#!/usr/bin/env python
# coding: utf-8

# # In this notebook, we will train the CNN to recognize Americal Sign Language, on dataset released by MNIST and improve the accuracy of the model by using:
# 1. Data Augmentation
# 2. Learning Rate Scheduler
# 3. Batch Normalisation and Regularisation

# ![Image](https://www.kaggleusercontent.com/kf/23902291/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..b4PwWh0-JaEztBfEfxEvxg.R-XjDU5wnsx1xXkhlRqHs5HJNUhIFT33qWmyrKVIggB9AXKOMkL2FkyjBKxlwZOGmXXSgdUSPWyARbtEogVVIPiHkrpR2nYWnlo-lIrhSgRKeepXELPAjTRP9kyiFsGzGUMkHsElPncT_rZ9pqxjBkCGtNYIVqPhpKhBrASPPt7X2Ye69XEBEqMkFO2DrUSwceeRTruc-Y3tRKL6mWxuxFMrJQCETl8pzQe-dQb9ivOEHg_IQlyB0SsGHljdd8MXmv4Y5X-MA1EnFTz6ZkjlLjzfTjX_vuMAe7b1fdnuO7lJxMvR6wUH5qvu5oVkWHgw8MBhpOH1K9MDUJAYnM1-6417kkZtuIHY9KcJpdX3nv41vWxz-AmWKexJiN-HIRk4gVMSnzf9dwWaaznKKdAWYwPZdXkAS-NcKws2ylUS0EUFh-jXS9T4EUrTr9J6-9TCv707WOJola-Lfjk_CJEoeMdKkYVFSiraRVlEstwxLuavwkMUwRwBBuXaZm6STFvoEABiGgB_CiZkan41iO-Tiyg-lBpsICaj_-SKH6Vd0ijs8WSOzKDDWusI3NwtqqZOdamczkGVFUHHmubbS5KnqUdKIO4q-UctePM7lFqZUhKhsBuKdC_bHbWargINs-xyHZRXtOJwiDY7OVdpBkevbpNF7fmk0_lfNpTmGP7OdLC7UNB-gwd1M_vbjgPu5qxu.9kdSjumTYY33e-DaRJUX4Q/__results___files/__results___1_0.png)
Importing the requried libraries for our task
# In[ ]:


import tensorflow as tf
import keras
from keras.callbacks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *


# **About Dataset:** Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).

# Loading the training dataset

# In[ ]:


traindata = pd.read_csv('./../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')


# In[ ]:


traindata.head(4)


# In[ ]:


traindata.shape


# First Column of data corresponds to given label, while columns 1 to 785 represents its 784 pixel values. There are total 27455 images of size 28,28 and are in grayscale mode.

# Lets look at some of the images from training data

# In[ ]:


f = plt.figure(figsize=(20,6))
ax = f.add_subplot(161)
ax2 = f.add_subplot(162)
ax3 = f.add_subplot(163)
ax4 = f.add_subplot(164)
ax5 = f.add_subplot(165)
ax6 = f.add_subplot(166)
ax.imshow(traindata.iloc[0].values[1:].reshape(28,28))
ax2.imshow(traindata.iloc[5].values[1:].reshape(28,28))
ax3.imshow(traindata.iloc[20].values[1:].reshape(28,28))
ax4.imshow(traindata.iloc[456].values[1:].reshape(28,28))
ax5.imshow(traindata.iloc[999].values[1:].reshape(28,28))
ax6.imshow(traindata.iloc[1500].values[1:].reshape(28,28))
plt.show()


# Separating the labels into trainlabel and image pixels into trainimages

# In[ ]:


trainlabel=traindata['label'].values
traindata.drop('label',inplace=True,axis=1)
trainimages = traindata.values
#reshape it to (28,28,1)-> (height,width,channels)
trainimages=trainimages.reshape(-1,28,28,1)


# Loading the test data

# In[ ]:


testdata = pd.read_csv('./../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')


# Similiar to train data, First column represent labels while others represents its pixel values

# In[ ]:


testlabel=testdata['label'].values
testdata.drop('label',inplace=True,axis=1)
testimages = testdata.values
testimages=testimages.reshape(-1,28,28,1)


# Importing Image Data generator for data augmentation in run time.ImageDataGenerator allows us to augment images on runtime without affecting the local copies. It provides us an advantage of trying different augmentation techniques and can experiment it with its values

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# 1. Augmenting the training data
# 2. Normalising pixel values from (0,255) to (0,1)
# 3. Splitting the data into 80% training and 20% validation

# In[ ]:


traingen=ImageDataGenerator(rotation_range=20,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.1,
                            horizontal_flip=True,
                            rescale=1/255.0,#normalising the data
                            validation_split=0.2 #train_val split
                            )


# Generating training and validation data from Image generator we created above

# In[ ]:


traindata_generator = traingen.flow(trainimages,trainlabel,subset='training')
validationdata_generator = traingen.flow(trainimages,trainlabel,subset='validation')


# Creating the test data Generator, we will only normalise it to (0,1)

# In[ ]:


testgen=ImageDataGenerator(rescale=1/255.0)


# In[ ]:


testdata_generator = testgen.flow(testimages,testlabel)


# Creating the Neural Network using Keras

# In[ ]:


model=Sequential([])

model.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(26,activation="softmax"))


# Structure of the Model and jouney of an image through Neural Network:

# In[ ]:


model.summary()


# Compiling the model, 
# 1. loss function: sparse_categorical_crossentropy since it is a multiclass classification model
# 2. optimizer: adam
# 3. metrics: accuracy-> no of images classified correctly

# In[ ]:


model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])


# It might occur that, we will overshoot the optima. Therefore, we should cancel training using callback, once the validation accuracy reaches 99.5 %

# In[ ]:


# Define a Callback class that stops training once accuracy reaches 99.5%
class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True
callback=myCallback()


# To stop fluctuating between the accuracy, we will decrease the learning rate on each epoch

# In[ ]:


dynamicrate = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)


# # # # Training the Model for 50 epochs

# In[ ]:


history=model.fit(traindata_generator,epochs=50,validation_data=validationdata_generator,callbacks=[callback,dynamicrate])


# Training Accuracy at the end of 15th epoch: 99.63
# 
# Validation Accuracy at the end of 15th epoch: 99.76
# 
# Since, validation accuracy has reached  above 99.5, callback function stopped further training of the model

# # 1. Plotting training vs validation accuracy
# #  2. Plotting training vs validation loss
# 

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Test the model on test data using model.evulate_generator and pass the test data generator which we have already created above.
# 
# It returns two arguments 
# 1. Loss
# 2. Accuracy

# In[ ]:


loss,accuracy = model.evaluate_generator(testdata_generator)


# In[ ]:


print("test accuracy: "+ str(accuracy*100))


# In[ ]:




