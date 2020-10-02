#!/usr/bin/env python
# coding: utf-8

# ### The Data
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive
# 
# The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

# In[ ]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


# ## Load in test and training data/ Preprocessing

# In[ ]:


train = pd.read_csv("../input/train1/train.csv")


# In[ ]:


test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


train.head()


# In[ ]:


a = train.label
train_labels = a.to_frame()

#important to have pixels np arrays to use "reshape" 
train_pixels = train.drop('label', 1).values
test_pixels = test.values


# In[ ]:


#reshape to input into cnn as [w][h[d]
train_pixels = train_pixels.reshape(train_pixels.shape[0], 28, 28,1).astype('float')
test_pixels = test_pixels.reshape(test_pixels.shape[0], 28, 28,1).astype('float32')


# In[ ]:


train_pixels.shape


# ## Preprocessing 

# ### standardize? 

# In[ ]:


train_pixels = train_pixels/255


# ### one-hot encode labels 

# In[ ]:


train_labels.shape


# In[ ]:


train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]

train_labels


# In[ ]:


train_labels.shape


# ## Design Neural Network

# Karas Sequential model = linear stack of layers 

# In[ ]:


model = Sequential()

#add convolutional layer
#32 kernals per conv layer, size of the kernals is 5x5
#input_shape [width][height][depth]
model.add(Conv2D(filters = 32,kernel_size = (5, 5), input_shape=(28, 28, 1), activation='relu'))

#add pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

#adding 2nd Conv layer!
model.add(Conv2D(32, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

#3rd conv
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

#add dropout layer, excludes 20% of neurons to avoid overfiting
model.add(Dropout(0.3))

#converts 2d matrix to vector... allows the output to be processed by standard fully connected layers.
model.add(Flatten())

#adds a fully connected layer with 256 neurons
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))


# ## Compile NN

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_labels.shape


# ## Set Optimizer

# In[ ]:





# ## Fit Model

# In[ ]:


model.fit(train_pixels, train_labels, epochs=10, batch_size=200, verbose=2)


# In[ ]:


predictions = model.predict_classes(test_pixels)


# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})


# In[ ]:


submissions.head()


# In[ ]:


submissions.to_csv("result.csv", index=False, header=True)


# ## Data Augmentation
# 
# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# 

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_pixels)


# In[ ]:


model.fit_generator(datagen.flow(train_pixels,train_labels, batch_size=200),epochs =10, verbose=2)


# In[ ]:


predictions = model.predict_classes(test_pixels)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("result.csv", index=False, header=True)


# # NOTES
# 
# -Finally got model to run. achieved 0.78 accuracy
# 
# -will see how much accuracy improves after converting input to float
# achieved 0.79 acc
# 
# -will see how much accuracy improves after standardizing input to 0-1 range
# jumped to 0.99 accuracy
# kaggle score of 97%
# 
# -fixed standardization from 225 to 255
# only 93% accuracy?
# 
# -dont think i changed anything but acc is up to 99.4%
# maybe this is because I am not using a random seed
# kaggle score of 8%???
# kaggle score of 98 now....
# 
# -adding one more conv layer
# kaggle slight increase
# 
# -adding dropout after each conv layer
# another slight increase
# kaggle is now at 99.1%
# 
# -adding another fully connected layer(dense 256)
# kaggle score does not change
# 
# 
# 
# -set epochs to 30?
# lowered kaggle score
# 
# -add augmented data
# lowered kaggle to 96
# 
# 
# addint more layers really increased run time
# 30sec to 1 min to 1.2 min per epoch
# 
# 
