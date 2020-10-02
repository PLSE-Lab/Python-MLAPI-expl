#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks with Image Augmentation

# In[ ]:


from IPython.display import Image
Image("../input/american_sign_language.PNG")


# ## About the Data

# The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. As noted in one recent replacement called the Fashion-MNIST dataset, the Zalando researchers quoted the startling claim that "Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel". To stimulate the community to develop more drop-in replacements, the Sign Language MNIST is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).
# 
# The full introduction can be seen here: 
# [https://www.kaggle.com/datamunge/sign-language-mnist/home](https://www.kaggle.com/datamunge/sign-language-mnist/home)

# This type of computations may be long, so I start with timer setting to know how much time the script will take.

# In[ ]:


import time
from time import perf_counter as timer
start = timer()


# Importing necessary modules:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


#  ## Data Load and Check

# In[ ]:


train = pd.read_csv('../input/sign_mnist_train.csv')
test = pd.read_csv('../input/sign_mnist_test.csv')
train.head()


# What are our data dimensions?

# In[ ]:


train.shape


# Here we can look at original photographs:

# In[ ]:


Image("../input/amer_sign2.png")


# Our `train` set is reworked to reduce a data size. In particular all images are in grayscale and their sizes are 28 * 28 pixels. I will show pictures  in a few steps.

# ## Data Preprocessing
# Let us start to extract information from our data. At first I take a look at labels.

# In[ ]:


labels = train['label'].values


# In[ ]:


unique_val = np.array(labels)
np.unique(unique_val)


# Is our data balanced?

# In[ ]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# As you can see all output numbers are about the same.
# 
# For our CNN network  I'm to create an output array with Label Binarizer from the labels.

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels


# Now I drop the label column from the 'train' set and will work with the rest of data.

# In[ ]:


train.drop('label', axis = 1, inplace = True)


# Now let us take out the image information from `train` object and put in into numpy array. What is a data type, range and dimensions?

# In[ ]:


images = train.values
print(images.dtype, np.round(images.min(), 4), np.round(images.max(), 4), images.shape)


# Let us see provided images using first 5 rows. 

# In[ ]:


plt.style.use('grayscale')
fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
for i in range(5): 
        axs[i].imshow(images[i].reshape(28,28))
fig.suptitle('Grayscale images')


# We are to normalize the data before applying CNN. Our data values range is from 0 to 255, so to normalize I divide every entry by 225.
# 

# In[ ]:


images =  images/255


# For validation during a model fitting we need to divide our train set in two parts. 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, stratify = labels, random_state = 7)


# Now I need to reshape our rows as square tables because I want to use a Convolution Neural Network method.

# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# ## Convolutional Neural Network Model, or CNN
# For CNN I am using keras library here.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


# Setting a number of classes,  a batch size and a number of epochs.

# In[ ]:


num_classes = 24
batch_size = 125
epochs = 50


# Here goes the CNN in all its beauty!

# In[ ]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='valid' ))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (4, 4), activation = 'relu', padding='valid' ))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer='nadam',
              metrics=['accuracy'])


# This part is for image augmentation during model fitting.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(shear_range = 0.25,
                                   zoom_range = 0.15,
                                   rotation_range = 15,
                                   brightness_range = [0.15, 1.15],
                                   width_shift_range = [-2,-1, 0, +1, +2],
                                   height_shift_range = [ -1, 0, +1],
                                   fill_mode = 'reflect')
test_datagen = ImageDataGenerator()


# And now it runs!

# In[ ]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# You see below how accuracy values improve with each epoch.

# In[ ]:


plt.style.use('tableau-colorblind10')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0.80, 1.05)
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


# Let's validate with the test data. At first it must be preprocessed in the same way as our data for model fitting. It means that  we are to remove its label column,  divide all values by 225 and rows should be reshaped as square arrays.

# In[ ]:


test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values/255
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape


# Here are predictions and an accuracy on our provided test set.

# In[ ]:


y_pred = model.predict(test_images)
from sklearn.metrics import accuracy_score
y_pred = y_pred.round()
accuracy_score(test_labels, y_pred)


# An accuracy may fluctuate due to randomness of applyed methods. 
# 
# ### My time count

# In[ ]:


end = timer()
elapsed_time = time.gmtime(end - start)
print("Elapsed time:")
print("{0} minutes {1} seconds.".format(elapsed_time.tm_min, elapsed_time.tm_sec))

