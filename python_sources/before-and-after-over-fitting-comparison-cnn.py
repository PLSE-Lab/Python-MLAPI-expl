#!/usr/bin/env python
# coding: utf-8

# **Before and After Overfitting Comparision on Stanford Cars Dataset**
# 
# This kernel will analyze how a Deep CNN will perform on a dataset that has 196 classes and around 40-45 images per class. In the first training session, only 10 epochs were used and then metrics like loss and accuracy of both test set and training set were compared. As we have very less images per class, it is not worthy to even give 10% accuracy on test set so, overfitting was done so that atleast we can get good results on training set. After overfitting, it can be seen that Loss on test set increases significantly as expected and there is very less improvement on accuracy on test set.

# In[ ]:


#Importing helpful libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image
import os
print(os.listdir("../input"))


# In[ ]:


#Train and Test folder directory
os.listdir("../input/car_data/car_data")


# In[ ]:


#Making seperate variables for train and test directories
train_dir = "../input/car_data/car_data/train"
test_dir = "../input/car_data/car_data/test"


# In[ ]:


#Dictionary that has name of car class as key and image name as its values
car_names_train = {}

for i in os.listdir(train_dir):
    car_names_train[i] = os.listdir(train_dir + '/' + i)


# In[ ]:


#Code to create two lists for class name and image directories corresponding to it
car_images_ls = []
car_names_ls = []
car_classes = []
car_directories = []

for i in car_names_train:
    car_classes.append(i)

for i,j in enumerate(car_names_train.values()):
    for img in j:
        car_images_ls.append(img)
        car_names_ls.append(car_classes[i])
        
for i in range(len(car_names_ls)):
    car_directories.append(train_dir + '/' + car_names_ls[i] + '/' + car_images_ls[i])


# In[ ]:


#Sample image to check the consistency of the two lists
plt.imshow(Image.open(car_directories[1000]))
plt.title(car_names_ls[1000])


# Creating a dataframe containing all the image directories and the car class corresponding to it

# In[ ]:


#Creating a data frame from the above two lists
df = pd.DataFrame(data = [car_directories, car_names_ls], index = ["Directories", "Car Class"]).T
df.head()
df.to_csv('car_names_directories.csv', index = False)


# In[ ]:


#Importing various modules from the Keras Library that are used in Deep Learning
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers


# **Hyper-Parameters**
# 
# Image size is set to be  256*256 pixels and epochs are set equal to 10 because after that, we are just overfitting the CNN.

# In[ ]:


#Pre-Defining some hyper-parameters
img_width, img_height = 256, 256
nb_train_samples = 8144
nb_validation_samples = 8041
epochs = 10
steps_per_epoch = 256
batch_size = 64
n_classes = 196


# In[ ]:


#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./ 255,
    zoom_range=0.2,
    rotation_range = 8,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# **Creating a Dense Convolution NN**
# 
# 4 Convolution layers are used to have a high variance. We can use a less dense model too as there are few images per class, but when overfitting is in mind we should make a denser model

# In[ ]:


#Creating the Convolution Neural Network
cnn = Sequential()
cnn.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (256,256,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.22))
cnn.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (256,256,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.22))
cnn.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'same', activation = 'relu', input_shape = (256,256,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (256,256,3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Flatten())
cnn.add(Dropout(0.18))
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(196, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


#Let's take a look at the CNN we created
cnn.summary()


# In[ ]:


#Training begins here
model_history = cnn.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)

cnn.save_weights('stanford_cars_folder_cnn_weights.h5')


# **Model Performance Visualization**
# 
# This model performance visualization is just before the model starts to overfit. It can be seen that on both the Training set and Test/Validation set, loss and accuracy is poor and the reason for that is:
# * Too many classes
# * Very less images per class
# * Model was not trained enough (If we trained the model for more epochs, we will overfit the training set)

# In[ ]:


#Visualization of how validation and training set performs per epoch
plt.figure(0, figsize = (5,5))
plt.plot(model_history.history['acc'],'orange')
plt.plot(model_history.history['val_acc'],'blue')
plt.legend(['Train-Accuracy','Val-Accuracy'])
_ = plt.title('Train vs Val Accuracy')
_ = plt.xlabel('Epochs')
_ = plt.ylabel('Accuracy')

plt.figure(1, figsize = (5,5))
plt.plot(model_history.history['loss'],'orange')
plt.plot(model_history.history['val_loss'],'blue')
plt.legend(['train','validation'])
_ = plt.xlabel("Num of Epochs")
_ = plt.ylabel("Loss")
_ = plt.title("Training Loss vs Validation Loss")


# **OVERFITTING Alert**
# 
# Just for the sake of results on train set, overfitting is done. The model is now trained for 30 more epochs and accuracy achieved on training set is over 92%.

# In[ ]:


"""TO OVERFIT THE MODEL"""
#As there is insufficient images per class (40-45 images per set for 196 classes) to get satisfactory
#results, I overfitted the model to perform best on the training set
model_history = cnn.fit_generator(
    train_generator,
    epochs=epochs + 20,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)

cnn.save_weights('stanford_cars_folder_cnn_weights_OVERFITTED.h5')


# **OVERFITTED Model Performance Visualization**
# 
# This model performance is when overfitting starts. After training on 30 more epochs, the model just learns how to classify only on the training set. The loss function on test set keeps on increasing as expected because of overfitting, although there is a very little accuracy improvement on the test set but it is still useless. 
# The reason of poor performance on the test set is:
# * Even after Data Augmentation, we have a very little image set per class. The golden rule for Image classification is atleast 1000 images should be used per class to get good results.
# * After overfitting the training set, the model learns by heart on how to classify the images only on the training set. It is just like you are learning the answers to some questions again and again and you perform good on those questions and getting 92/100, but when new questions come you cannot even get 6/100. If you were a student, you will look to learn from more questions other than those in which you are getting 92/100 to get better marks on test.
# * By overfitting, we get a high variance and that is the reason for very high loss.

# In[ ]:


#Visualization of Validation and training set performance after overfitting
plt.figure(0, figsize = (5,5))
plt.plot(model_history.history['acc'],'orange')
plt.plot(model_history.history['val_acc'],'blue')
plt.legend(['Train-Accuracy','Val-Accuracy'])
_ = plt.title('Train vs Val Accuracy (OVERFITTED)')
_ = plt.xlabel('Epochs')
_ = plt.ylabel('Accuracy')

plt.figure(1, figsize = (5,5))
plt.plot(model_history.history['loss'],'orange')
plt.plot(model_history.history['val_loss'],'blue')
plt.legend(['train','validation'])
_ = plt.xlabel("Num of Epochs")
_ = plt.ylabel("Loss")
_ = plt.title("Training Loss vs Validation Loss (OVERFITTED)")


# References:
# https://www.kaggle.com/jutrera/training-a-densenet-for-the-stanford-car-dataset
