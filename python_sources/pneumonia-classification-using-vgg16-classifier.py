#!/usr/bin/env python
# coding: utf-8

# # Using VGG16 for Pneumonia classification
# 

# ## Fundamental terms:
# 
# # 1. ** Overfitting **
# #       Overfitting happens when a neural system model remembers designs in a dataset as opposed to learning the general thought/example of the information.
# #       This is usually cause by an overly complicated neural network model.
# 
# #     
# # 2. **Underfitting**
# #         Underfitting happens when a neural system model doesn't perceive designs in the dataset.
# #         This is usually caused by a too simple neural network model, or when there's too much noise in the dataset.
# #         

# **The main purpose of this project is to show you how to unleash the power of convolutional neural networks in order to detect pneumonia in a person based of off their chest x-ray image.**
# 
# 

# 

# #### Importing essential libraries
# # Now we're importing all necesarry libraries for us to work with.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt #Ploting charts
from glob import glob #retriving an array of files in directories
from keras.models import Sequential #for neural network models
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical #For One-hot Encoding
from keras.optimizers import Adam, SGD, RMSprop #For Optimizing the Neural Network
from keras.callbacks import EarlyStopping


# Exploring the paths of the dataset.
# # This is where our data is stored.

# In[ ]:


#Cheking datasets
import os
paths = os.listdir(path="../input")
print(paths)


# ## Data Analysis and Preprocessing

# ### Now we're going to explore the dataset that contains chest x-ray images of people who have pneumonia and people who don't.
# # Our main goal is to predict if a person has pneumonia or not based of off their chest x-ray image.
# # 
# # #### So now we'll display one chest x-ray image of a person with pneumonia and one with a person without pneumonia, just to have a glimpse of what each image looks like in general.

# Getting all images in the dataset

# In[ ]:


path_train = "../input/chest_xray/chest_xray/train"
path_val = "../input/chest_xray/chest_xray/val"
path_test = "../input/chest_xray/chest_xray/test"


# #### Pneumonia:

# In[ ]:


img = glob(path_train+"/PNEUMONIA/*.jpeg") #Getting all images in this folder


# Converting the first image we get from the above directory/path into a numpy array

# In[ ]:


img = np.asarray(plt.imread(img[0]))


# Plotting the image

# In[ ]:


plt.imshow(img)


# In[ ]:


img.shape #Checking the shape of this image. It seems like a two deminsional shape (1422 x 1152)


# #### Normal:

# In[ ]:


img = glob(path_train+"/NORMAL/*.jpeg") #Getting all images in this folder


# In[ ]:


img = np.asarray(plt.imread(img[0]))


# In[ ]:


plt.imshow(img)


# In[ ]:


img.shape


# ### Transforming the images
# # * Now we're applying a technique called Data Augmentation.
# # * We're changing the sizes of the images to 226 x 226 and we'll flip the images horizontally as well so that we can have more data(images) to train on.

# #### In our dataset we're given three sets of images:
# # 1. The training set. These are images we're going to train the neural network on.
# # 2. The validation set. These are images we're going to use to check if the model is underfitting or overfitting, while training and compare the training and validation results in real time.
# # 3. The test set. These are images we're going to use to check how good our neural network is with data it has not seen before.

# In the following example, we're attempting to avoid overfitting by augmenting our image data.
# # Data augmentation means we're going to make slight variations to our data so that we have more data, without losing semantic meaning in our data.
# # The augmentation occurs in the parameters of the ImageDataGenerator method. To get a better understanding of these parameters you can check out [this link](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/) and [this one.](https://keras.io/preprocessing/image/)
# # 
# 
# # **horizontal_flip** set to true implies that some images in the data will be randomly horizontally flipped, as chest x-ray images don't have any significant meaning when horizontally flipped(at least for machine learning purpose).
# # 
# # **channel_shift_range** will randomly shift the channels of our images. Image channel refers to the RGB color scheme, which implies some imges will slightly vary in color.
# # 
# # **rotation_range** will slighty rotate the image according value given to it.
# # 
# # **zoom_range** will slightly zoom in to the image according value given to it.
# 

# In[ ]:


#Data preprocessing and analysis

train_data = glob(path_train+"/NORMAL/*.jpeg")
train_data += glob(path_train+"/PNEUMONIA/*.jpeg")
data_gen = ImageDataGenerator() #Augmentation happens here
classes = ["NORMAL", "PNEUMONIA"]
#But in this example we're not going to give the ImageDataGenerator method any parameters to augment our data.


# In[ ]:


val_batches = data_gen.flow_from_directory(path_val, target_size = (226, 226), classes = classes, class_mode = "categorical")
test_batches = data_gen.flow_from_directory(path_test, target_size = (226, 226), classes = classes, class_mode = "categorical")
train_batches = data_gen.flow_from_directory(path_train, target_size = (226, 226), classes = classes, class_mode = "categorical")


# In[ ]:


train_batches.image_shape


# ## The Artificial Neural Network
# # ### This particular neural network is called a convolutional neural network because it has convolutional layers that convolve the images/arrays of data it's being trained on.
# # This model is based off a model that won the ImageNet competition...

# One of the best things about being in a tech industry is that fellow smart techies who've created cool and robust neural network are generous enough to share their model architecture with us, so we don't have re-invent the wheel.
# # This will save us some time and headache. We're going to use a method known as **transfer learning**. This means instead of creating a brand new neural net that's going to be time consuming, we can just use a pre-trained [good] model and fine tune it in order for it to work for our own scenario.
# # 
# # Usually when people do transfer learning, they use both the architecture and weights of a pre-trained model. But in this tutorial we're only using the architecture of a pretrained model, not their weights.

# 

# In[ ]:


#This is a Convolutional Artificial Neural Network
#VGG16 Model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_batches.image_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[ ]:


#Viewing the summary of the model
model.summary()


# ### Training the neural net

# Now the training begins

# We're training our model for 5 epochs.
# # This means we're giving the model 5 chances to learn patterns about our data.
# # 
# # During preparing we will apply a method called Early Stopping. This procedure will quit preparing of the model if there's no improvement during the preparation procedure.
# # In the below example of early stopping, our parameter **patience** tells the model to stop training if there's no improvements after 3 consecutive epochs, and monitor tells the model which metric to look at in order to apply early stopping.
# 

# In[ ]:


optimizer = Adam(lr = 0.0001)
early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit_generator(epochs=5, callbacks=[early_stopping_monitor], shuffle=True, validation_data=val_batches, generator=train_batches, steps_per_epoch=500, validation_steps=10,verbose=2)
prediction = model.predict_generator(generator=train_batches, verbose=2, steps=100)


# ## Ploting the model performance

# Now we're going to plot the model's performance
# # Obsertvation:-
# 
# #   If validation/test accuracy is greater than training accuracy, that's good, it means our model has managed to learn and get a general idea/pattern of our data.
# #   But if training accuracy is greater than validation/testing accuracy, tha's not good. That means our model is overfitting.
# #  
# #  
# #  The opposite is true for the loss chart.
# #  The ideal situation is to have validation/test loss way low. But train loss should not be lower than test/validation loss.

# ## Model Accuracy Chart

# In[ ]:




# summarize history for accuracy

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# ## Model Loss Chart

# In[ ]:



# summarize history for loss

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# In[ ]:





# 

# 
