#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# *This is my first time using keras* (with tensorflow backend) hands-on to practice DL. I am using a video from youtube to guide myself and created this notebook as a memo and note to myself.
# 
# You can find the video playlist [here](https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2&t=0s).
# 
# - Being a newbie at DL, this is very much a very very basic model without any augmentations done. The validation scores and accuracy will be pretty horrible.
# 
# - Having said that, this kernel will be helpful in giving the most BASIC BASELINE model to improve upon, so it can be used to practice DL and other improvement methods like tranfer learning, using data generators, data augmentation etc.
# 
# - **I have deleted the unzipped files at the end, so that after the predictions are made, you can easily see your output .csv, or else it'll be filled with cat and dog images which were unzipped on output directory.**
# 
# - Incase, you are not able to get rid of cat and dog images, you'll find your csv in the output folder in "Data" section to right, in the editable kernel.

# In[ ]:


import numpy as np
import pandas as pd 
import os
import cv2

import matplotlib.pyplot as plt


# # Unzipping the data onto Kaggle itself
# The following code will unzip the image data. Although they will show up as normal test and train with ".zip", you should be able to use os and walkthrough inside the zip files to access each image after this step without any warnings.

# In[ ]:


get_ipython().system('unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train')
get_ipython().system('unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test')


# In[ ]:


TRAIN_DIR = '../working/train/train/'
TEST_DIR = '../working/test/test/'

train_images_filepaths = [TRAIN_DIR + last_file_name for last_file_name in os.listdir(TRAIN_DIR)]
test_images_filepaths = [TEST_DIR + last_file_name for last_file_name in os.listdir(TEST_DIR)]

print("Done")


# We have training data containing both cat as well as dog image. Let us make a list of cat and dog images from the train data and store them separately.
# 
# Getting the dogs and cats data sorted from the training dataset using list comprehension will do the trick for now.

# In[ ]:


train_dogs_filepaths = [TRAIN_DIR+ dog_file_name for dog_file_name in os.listdir(TRAIN_DIR) if 'dog' in dog_file_name]
train_cats_filepaths = [TRAIN_DIR+ cat_file_name for cat_file_name in os.listdir(TRAIN_DIR) if 'cat' in cat_file_name]

print("Done")


# ## Seeing a sample image
# 
# Each entry in the `train_dogs` list is a file path to one individual image of dog in jpg format.
# We will be converting each of this photo into an array so that each individual image can be represented as an array.

# In[ ]:


#Seeing a "color" image
test_img_file_path = train_dogs_filepaths[0]
img_array = cv2.imread(test_img_file_path,cv2.IMREAD_COLOR) #The last parameter can be switched with cv2.IMREAD_GRAYSCALE too
plt.imshow(img_array)
plt.show()


# In[ ]:


#Unhide the output to see how the image looks like in array form
print(img_array)


# In[ ]:


print(img_array.shape)


# The 3 at the end signifies that the image has 3 channels, each for
# * red
# * green
# * blue
# 
# Incase of grayscale images, there is no need for such three channels. Below is a quick implementation of it,

# In[ ]:


img_array_gray = cv2.imread(test_img_file_path,cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array_gray, cmap = "gray")
plt.show()

print(img_array_gray.shape)


# ## Resizing the photos
# 
# Each image in the file needn't be the same in it's dimensions. A snapshot of the images as given below.
# 
# NOTE : In the snapshot, the filenames don't match with that of Kaggle's data. Kaggle has each image filename with 'dog' or 'cat' mentioned in it (only for train data). This dataset used in the YT tutorial is downloaded from microsoft store, which has both the cats and dog files separated into two respective folders.
# 
# ![Dog Snapshot](https://i.imgur.com/0NiPett.jpg)
# 
# Hence, we will set a global image size, which will serve as a global dimesnion standard for all the images.

# In[ ]:


ROW_DIMENSION = 60
COLUMN_DIMENSION = 60
CHANNELS = 3 #For greyscale images put it to 1; put it to 3 if you want color image data

new_array = cv2.resize(img_array_gray,(ROW_DIMENSION,COLUMN_DIMENSION)) #A squarish compression on it's width will take place
plt.imshow(new_array,cmap = 'gray')
plt.show()


# # Prep train and test images
# 
# Now, we need to prep all the images in the datasets, ie, assigning them with the same global dimensions and other configurations so they stay uniform. 
# 
# We will also add a read_converted_img function that will take an image array as argument and display it back in the converted format, incase we want to see any image in future.
# 
# `prep_img` does exactly the same, except it returns the modified resized array.
# 
# ![](http://)We will return preped_data which will contain all the modified image arrays while the original filepaths linking to the original image files remain unchanged inside `train_dogs` and `train_cats`

# In[ ]:


def read_converted_img(to_read_img_array):
    plt.imshow(to_read_img_array,cmap = 'gray')
    plt.show()
    
def prep_img(single_image_path):
    img_array_to_resize = cv2.imread(single_image_path,cv2.IMREAD_COLOR)
    resized = cv2.resize(img_array_to_resize,(ROW_DIMENSION,COLUMN_DIMENSION),interpolation = cv2.INTER_CUBIC)
    return resized

def prep_data(list_of_image_paths):
    
    size = len(list_of_image_paths)
    
    #preped_data = np.ndarray((size, ROW_DIMENSION, COLUMN_DIMENSION,CHANNELS), dtype=np.uint8)
    preped_data = []
    
    '''
    for i in range(size):
        list_of_image_paths[i] = prep_img(list_of_image_paths)
    '''
    
    for i, image_file_path in enumerate(list_of_image_paths):
        '''
        image = prep_img(image_file_path)
        #preped_data[i] = image.T
        preped_data.append(image)
        '''
        preped_data.append(cv2.resize(cv2.imread(image_file_path), (ROW_DIMENSION,COLUMN_DIMENSION), interpolation=cv2.INTER_CUBIC))
        
        if(i%1000==0):
            print("Processed",i,"of",size)
        
        #print(image.shape)
        #print(preped_data.shape)
        
    return preped_data


# The two variables, `train_data` and `test_data` will be used for storing the modified prep data generated from `train_images` and `test_images` 

# In[ ]:


print("PREPING TRAINING SET")
train_data = prep_data(train_images_filepaths)
print("\nPREPING TEST SET")
test_data = prep_data(test_images_filepaths)
print("\nDone")


# Since train_data is a list, let us convert it into a numpy array. I will name it X_train for convenience.

# In[ ]:


X_train = np.array(train_data)

print(X_train.shape)
#print(train_data.shape)
#print(test_data.shape)


# Let us try visualising a photo from the converted training set. We'll use the `read_converted_img` method we defined above for this.

# In[ ]:


read_converted_img(X_train[0])


# # Preparing Y_train
# 
# Unlike the MNIST dataset, which has a separate column `label` depicting the outcome, we have no such column in this case. However, in the filepath names, each file in train.zip folder has 'dog' or 'cat' being written in it's filename.
# 
# The same is not true for testing images for obvious reasons. You can confirm this with the code below.
# 
# Let's start making the `y_train`!

# In[ ]:


print(train_images_filepaths[:3])
print("\n")
print(test_images_filepaths[:3])


# In[ ]:


#Preparing y_train

y_train = []

for path_name in train_images_filepaths:
    if('dog' in path_name):
        y_train.append(1)
    else:
        y_train.append(0)

print("Percentage of dogs is",sum(y_train)/len(y_train))


# We have equal number of cats and dog photos to train from. This is good.
# Let's convert this list to a numpy array too.

# In[ ]:


y_train = np.array(y_train)
y_train.shape


# # Choosing a model
# 

# In[ ]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout

print("Import Successful")


# In[ ]:


dvc_classifier = Sequential()

dvc_classifier.add(Conv2D(32,kernel_size = (3,3),
                         activation = 'relu',
                         input_shape = (ROW_DIMENSION,COLUMN_DIMENSION,3)))

dvc_classifier.add(Conv2D(32,kernel_size = (3,3),
                        activation = 'relu'))

dvc_classifier.add(Conv2D(64,kernel_size = (3,3),
                        activation = 'relu'))

dvc_classifier.add(Flatten())

dvc_classifier.add(Dense(128,activation = 'relu'))

dvc_classifier.add(Dropout(0.5))

dvc_classifier.add(Dense(1,activation = 'sigmoid'))

dvc_classifier.summary()


# ## Syntax Observations : Important for absolute Begineers
# 
# - **NOTE** : We passed a dense layer of size 1 at the end as this is a binary classification problem and one prediction is enough to find the other prediction. 
# Eg, if a picture is 80% likely to not be a cat, it is 80% likely to be a dog.
# 
# This is not the case with MNIST dataset that is tasked with recognising the digits. The digits can be any one of the 10 types, and binary classification isn't a viable option.
# 
# This is highlighted by the shape of our target arrays or the `y_train`
# 
# The shape of `y_train` in most tutorials for Digit Recognition such as [this one](https://www.kaggle.com/poonaml/deep-neural-network-keras-way), has shape of `y_train` as [num_of_testcases,classifiaction_categories] or simply. [num_of_testcases,10]
# 
# In this case, `y_train` is of shape (25000,)
# 
# Simply put, in MNIST Digit Recognition, a number 8 would have it's `y_train` row as
# [0,0,0,0,0,0,0,0,1,0]
# 
# In this excercise, if a picture is cat, it's `y_train` is given as
# [0]
# 
# - **NOTE 2** : If you decide to use same syntax as MNIST digit prediction, you need to show a cat image in it's y_train representation as [1,0] where 0th index is for Cats and 1th index is for Dogs.
# 
# Doing so also allows you to use the 'softmax' activation function at the last Dense Layer.
# The respective code changes to
# 
# `dvc_classifier.add(Dense(num_of_classes,activation = 'softmax'))`
# where num_of_classes = 2

# Compile the model.

# In[ ]:


dvc_classifier.compile(loss = keras.losses.binary_crossentropy,
                      optimizer = 'adam',
                      metrics = ['accuracy'])


# Fit the model.
# Since train_data has no need of splitting into X_train and y_train (due to there being no labels in train_data), we can safely conclude `X_train` would be = `train_data`.
# For readability, we'll copy the elements of it anyway in a new varible `X_train` and use it for fitting.

# In[ ]:


dvc_classifier.fit(X_train,y_train,
               batch_size = 128,
               epochs = 3,
               validation_split = 0.2)


# In[ ]:


#Trying to save a model
model_json = dvc_classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dvc_classifier.save_weights("model.h5")


# In[ ]:


from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


# In[ ]:


loaded_model.summary()


# In[ ]:


arr_test = np.array(test_data)


# In[ ]:


prediction_probabilities = dvc_classifier.predict(arr_test, verbose=0)


# Let us visualize some of the predictions the model made.

# In[ ]:


for i in range(5,11):
    if prediction_probabilities[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(prediction_probabilities[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-prediction_probabilities[i][0]))
        
    plt.imshow(arr_test[i])
    plt.show()


# # Generating output

# We will first remove the zipped files, as they fill up the output section and we are able to see no .csv file.
# We later make the .csv file 'submissions'

# In[ ]:


#Deletig the folders containing unzipped data so output section is free of pictures

import sys
import shutil

# Get directory name
mydir = "/kaggle/working"

try:
    shutil.rmtree(mydir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))


# In[ ]:


pred_vals = [float(probability) for probability in prediction_probabilities ]

submissions = pd.DataFrame({"id": list(range(1,len(prediction_probabilities)+1)),
                         "label": pred_vals})

submissions.to_csv("dogvcat_1.csv", index=False, header=True)

print("Time to submit the baseline model!")


# ## Improve upon the model from here
# 
# TODO

# In[ ]:




