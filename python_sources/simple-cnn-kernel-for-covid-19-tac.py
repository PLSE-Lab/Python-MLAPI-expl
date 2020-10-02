#!/usr/bin/env python
# coding: utf-8

# <h1> Another Simple Convolutional Neural Network (First Notebook) </h1>
# 
# Sorry about my bad English ;-) (I'm not native...)<br>
# The objective of this Notebook is create a simple to follow CNN to anyone interested and start practising and learning.<br>
# I would appreciate any feedback and of course a vote will motivate me to continue, but as the main objective is to learn I prefer coment focus on my learnig, what can be done better etc.
# 
# Perhaps I'll improve this notebook with the comments, some of them I can imagine now as "use cross validation" but as it's a first minimun CNN (but I think it's ok for this objective) no more improvements to make it simple to understand will be made at the moment, maybe I will make a second version of the notebook if the improvements mean more than a marginal improvement of the result...<br>
# Kindly regards to everyone reading this!!!

# In[ ]:


# First imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import random
import shutil


# We'll start deleting previous output if it exist.

# In[ ]:


#Delete working directory from previous experiment if exists
fileListToDelete = ['/kaggle/working/train', '/kaggle/working/validation', '/kaggle/working/test']
for fileName in fileListToDelete:
    try:
        shutil.rmtree(fileName)
        break
    except:
        print("Directory ",fileName," not found to delete.")

#Create directories for training, validation and test data
subdirs  = ['train/', 'validation/', 'test/']
for subdir in subdirs:
    labeldirs = ['CT_COVID', 'CT_NonCOVID']
    for labldir in labeldirs:
        newdir = subdir + labldir
        os.makedirs(newdir, exist_ok=True)


# Then we'll randomly copy files from input to output in train, validation and test directories.

# In[ ]:


# Copy randomly files from directories to working directory separated in train, validation and set with a likelihood of 70& train, 20% validation, 10% test
train_size = 0        
validation_size = 0
test_size = 0 
path_COVID = os.path.join('/kaggle/input/covidct/CT_COVID/')
path_NonCOVID = os.path.join('/kaggle/input/covidct/CT_NonCOVID/')
covid_images = glob(os.path.join(path_COVID,"*.png"))
covid_images.extend(glob(os.path.join(path_COVID,"*.jpg")))
noncovid_images = glob(os.path.join(path_NonCOVID,"*.png"))
noncovid_images.extend(glob(os.path.join(path_NonCOVID,"*.jpg")))

for filename in covid_images: #copy to positive directory
    randNumber = random.uniform(0, 1)
    if(randNumber<0.7) : #train set
        shutil.copy(filename, "/kaggle/working/train/CT_COVID/")
        train_size = train_size + 1
    elif(randNumber<0.9) : #validation set
        shutil.copy(filename, "/kaggle/working/validation/CT_COVID/")
        validation_size = validation_size + 1
    else: #test set
        shutil.copy(filename, "/kaggle/working/test/CT_COVID/")
        test_size = test_size +1
        
for filename in noncovid_images: #copy to negative directory
    randNumber = random.uniform(0, 1)
    if(randNumber<0.7) : #train set
        shutil.copy(filename, "/kaggle/working/train/CT_NonCOVID/")
        train_size = train_size + 1
    elif(randNumber<0.9) : #validation set
        shutil.copy(filename, "/kaggle/working/validation/CT_NonCOVID/")
        validation_size = validation_size + 1
    else: #test set
        shutil.copy(filename, "/kaggle/working/test/CT_NonCOVID/")
        test_size = test_size +1


# We have the data as we wish, time to start with Keras!

# In[ ]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


# We'll use data augmentation with ImageDataGenerator() class from Keras to aid us to avoid overfitting.<br><br>
# 
# Three separate sets for train, validation and test.<br>
# One improvement here will be to use cross-validation and tune sizes of the sets.<br>
# All images rescaled to 1./255 so we'll get a convenient value for the model between 0-1 (rescaled from 0-255).
# shear_range will randomly apply shear transformations to the data, zoom_range for randomly zooming images. only in the train set to avoid overfitting.<br>
# Not used parameters are horizontal_flip to randomly flip images in our dataset, rotation_range which is the value in degrees the image may be randomly rotated, width_shift_range and height_shift_range for randomly translating images.
# 

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)

validation_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


# Some variables
IMG_HEIGHT = 150
IMG_WIDTH = 150
train_dir = os.path.join('/kaggle/working/train/')
validation_dir = os.path.join('/kaggle/working/validation')
test_dir = os.path.join('/kaggle/working/test/')
batch_size = 16
epochs = 15


# Generation of training, validation and test set.

# In[ ]:


print("Training set:")
training_set = train_datagen.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
print("Validation set:")
validation_set = train_datagen.flow_from_directory(batch_size=batch_size,
                                                           directory=validation_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
print("Test set:")
test_set = test_datagen.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# We get to the creation of the CNN model.<br>
# *Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Neural Network with Dropout*

# In[ ]:


#Creation of the CNN

classifier = Sequential()                                                                           
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_HEIGHT, IMG_WIDTH, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(classifier.summary())


# We set a stop criteria if there is no improvement in some epochs, to avoid an eternal training with too much epochs.
# 

# In[ ]:


#epochs = 2; #Uncomment to test quick
#set early stopping criteria
from keras.callbacks import EarlyStopping
epochsWithOutImprovement = 3 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=epochsWithOutImprovement, verbose=1)
                               
history = classifier.fit_generator(training_set,
                         steps_per_epoch = train_size,
                         epochs = epochs,
                         validation_data = validation_set,
                         validation_steps = validation_size,
                         callbacks=[early_stopping])


# Plot the accuracy and loss value over epochs

# In[ ]:


# Plot accuracy and loss over epochs
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# We test the accuracy with the test set, so we can get a more real idea of the performance of the model.

# In[ ]:


test_accuracy = classifier.evaluate_generator(test_set,steps=test_size)
print('Testing Accuracy with TEST SET: {:.2f}%'.format(test_accuracy[1] * 100))


# As initial data is splitted randomly each time, we get different accuracy values for test set.<br>
# This time we get **84.36%**
# 
# Thanks for your interest and remember I wish any feedback to learn and improve, and I would appreciate a vote too.
# Regards.
