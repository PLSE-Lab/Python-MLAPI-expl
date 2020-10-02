#!/usr/bin/env python
# coding: utf-8

# # Intro
# The Hand Gesture Recognition Database is a collection of near-infra-red images of ten distinct hand gestures. In this notebook we use end-to-end deep learning to build a classifier for these images.
# 
# This kernal is forked from: https://www.kaggle.com/lamine16/hand-gesture-recognition-database-with-cnn
# 
# We'll first load some packages required for reading in and plotting the images. 

# In[ ]:


import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
from matplotlib.pyplot import imshow


# Let's find the file structure of the dataset.
# Fist, the full dataset is divided into 10 folders.

# In[ ]:


sorted(os.listdir('../input/leapgestrecog/leapGestRecog/'))


# Each folder has 10 subfolders, which represent 10 calsses of the hand gestures.

# In[ ]:


sorted(os.listdir('../input/leapgestrecog/leapGestRecog/00/'))


# In each "Class Folder", there are 200 samples that has the same label.

# In[ ]:


print(len(os.listdir('../input/leapgestrecog/leapGestRecog/00/01_palm')))
sorted(os.listdir('../input/leapgestrecog/leapGestRecog/00/01_palm'))[:10]


# Now, let's show the first image of calss "01_palm".

# In[ ]:


# show the first sample image
sample_img_1 = Image.open('../input/leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0001.png')
sample_img_1


# Then, we can get the original image size. And try to resize the shape to half of its original size.

# In[ ]:


# show original image size
sample_img_2 = Image.open('../input/leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0002.png')
print('original image size:', sample_img_2.size)
# resize size of a image
sample_img_2 = sample_img_2.resize((320, 120))
print('resized image size:', sample_img_2.size)


# As described in the Data Overview, there are 10 folders labelled 00 to 09, each containing images from a given subject. In each folder there are subfolders for each gesture. We'll build a dictionary `lookup` storing the names of the gestures we need to identify, and giving each gesture a numerical identifier. We'll also build a dictionary `reverselookup` that tells us what gesture is associated to a given identifier.

# In[ ]:


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('../input/leapgestrecog/leapGestRecog/00/'):
    if not j.startswith('.'): # If running this code locally, this is to 
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup


# Next we read in the images, storing them in `x_data`. We store the numerical classifier for each image in `y_data`. Since the images are quite large and are coming from an infra-red sensor, there's nothing really lost in converting them to greyscale and resizing to speed up the computations.

# In[ ]:


x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('../input/leapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('../input/leapgestrecog/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('../input/leapgestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size


# Let's take a look at some of the pictures. Since each of the subfolders in `00` contained 200 images, we'll use the following piece of code to load one image of each gesture.

# In[ ]:


from random import randint
for i in range(0, 10):
    plt.imshow(x_data[i*200, :, :])
    plt.title(reverselookup[y_data[i*200 ,0]])
    plt.show()


# The first thing to note is that this is not a difficult classification problem. The gestures are quite distinct, the images are clear, and there's no background whatsoever to worry about. If you weren't comfortable with deep learning, you could do quite well with some straight-forward feature detection -- for example the '07_ok' class could easily be detected with binary thresholding followed by circle detection. 
# 
# Moreover, the gestures consistently occupy only about 25% of the image, and all would fit snugly inside a square bounding box. Again if you're looking to do basic feature detection, an easy first step would be to write a short script cropping everything to the relevant 120 x 120 square. 
# 
# But the point of this notebook is to show how effective it is to just throw a neural network at a problem like this without having to worry about any of the above, so that's what we're going to do. 
# 
# At the moment our vector `y_data` has shape `(datacount, 1)`, with `y_data[i,0] = j` if the `i`th image in our dataset is of gesture `reverselookup[j]`. In order to convert it to one-hot format, we use the keras function to_categorical:

# In[ ]:


y_data.shape


# In[ ]:


y_data[199:210]


# In[ ]:


import keras
from keras.utils import to_categorical
y_data = to_categorical(y_data)


# Now, y_data is categorized to (sample_count, class_num).

# In[ ]:


y_data.shape


# Our set of images has shape `(datacount, 120, 320)`. Keras will be expecting another slot to tell it the number of channels, so we reshape `x_data` accordingly. We also rescale the values in `x_data` to lie between 0 and 1.

# In[ ]:


# x_data = x_data.reshape((-1, 120, 320, 1))
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255


# We need a cross-validation set and a test set, and we'll use the `sklearn` package to construct these. In order to get an 80-10-10 split, we call `train_test_split` twice, first to split 80-20, then to split the smaller chunk 50-50. Note that we do this after the rescaling step above, to ensure that our train and test sets are coming from the same distribution.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)


# Now it's time to build our network. We'll use keras.

# In[ ]:


from keras import layers
from keras import models


# Since our images are big (we chose not to do any cropping) and the classification problem looks quite easy, we're going to downsample fairly aggressively, beginning with a 5 x 5 filter with a stride of 2. Note we have to specify the correct input shape at this initial layer, and keras will figure it out from then on. We won't worry about padding since it's clear that all the useful features are well inside the image. We'll continue with a sequence of convolutional layers followed by max-pooling until we arrive at a small enough image that we can add a fully-connected layer. Since we need to classify between 10 possibilities, we finish with a softmax layer with 10 neurons. 

# In[ ]:


model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# Finally, we fit the model.

# In[ ]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))


# At this point we would typically graph the accuracy of our model on the validation set, and choose a suitable number of epochs to train for to avoid overfitting. We might also consider introducing dropout and regularization. However, we can see we're getting perfect accuracy on the validation set after just one or two epochs, so we're pretty much done. Let's quickly confirm that this is carrying through to the test set:

# In[ ]:


[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))


# You'll get slightly different numbers each time you run it but you should be getting between 99.9 and 100% accuracy. Great!
