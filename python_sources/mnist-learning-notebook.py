#!/usr/bin/env python
# coding: utf-8

# ** Import requirements for the environment **
# 
# This is the standard import of the machine learning set for Numpy, Pandas, Matplotlib, Tensorflow, and Keras

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graphing library
get_ipython().run_line_magic('matplotlib', 'inline')

# Import tensorflow and keras. We will use the keras that is a part of the tensorflow package
# as opposed to the standalone keras
import tensorflow as tf
from tensorflow import keras


# **Import input files**
# 
# Kaggle stores all of the datasets attached to this notebook in the /kaggle/input directory underneath a directory named for the dataset.

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Load the training and testing data**
# 
# The test.csv file contains all of the data that will be used for the submission. This will be data that the model hasn't seen and will be well suited to measuring how effective the model generalized.

# In[ ]:


training_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
submission_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# **Print a sample of the training data**

# In[ ]:


training_data.head()


# ** Print a sample of the submission data **
# 
# Once we train the network, we will generate a result for a set of data that the network has not seen and use that for the submission.
# 

# In[ ]:


submission_df.head()


# In[ ]:


print("Training data (rows, columns):" + str(training_data.shape))
print("Submission data (rows, columns):" + str(submission_df.shape))


# **Test/Train/Split in Pandas** 
# 
# We will split up the data using Pandas, only, so that we can minimize our dependencies on any other libraries (and Pandas provides us all the functionality that we need).
# 

# In[ ]:


# We have a Dataframe that has labelled data of the right shape so let's split up that dataframe
# so that we can use it for test, train, and validation. This allows us to move forward with the data we DO have
#
# Sample the training_data Dataframe and select 70% of the data for the training_df, testing_df, validation_df dataframes
# We set the random_state to a specific number so we can reproduce the results

training_df = training_data.sample(frac=0.7, random_state=1)

# Drop the already selected values from the training_data dataframe, what is left is our testing data
testing_df = training_data.drop( training_df.index )

print("Training data (rows, columns):" + str(training_df.shape))
print("Testing data (rows, columns):" + str(testing_df.shape))

print( str(training_data.shape[0] - (training_df.shape[0] + testing_df.shape[0] )) + " rows not accounted for" )


# **Clean dataframe and extract labels**
# 
# The dataframes contain all of the data, but we need to extract the label column from the dataframe and also remove it from the training data since it is not trainable data

# In[ ]:


training_labels = training_df["label"].values
training_df = training_df.drop( columns=['label'])


# In[ ]:


# Confirm that the data is of the right shape
print( "Training data (rows, columns):" + str(training_df.shape) )
print( "Label Count: " + str(training_labels.size) )

Perform the same steps for the testing data
# In[ ]:


testing_labels = testing_df['label'].values
testing_df = testing_df.drop( columns=['label'])


# In[ ]:


# Confirm that the data is of the right shape
print( "Testing data (rows, columns):" + str(testing_df.shape) )
print( "Testing Label Count: " + str(testing_labels.size) )


# **Data Normalization**
# 
# The data that we have is not of the proper range. The pixel values run from min(0) to max(255) because these images were originally grayscale images. While this is fine for display data, we want to get these into the range of 0-1 for the training process. When working with machine learning data, we want all of our data normalized so there are no issues from having wild ranges of data values across features.

# In[ ]:


print( training_df.describe() )


# In[ ]:


# Normalize all the values in the dataframe so that they are values of range 0..255 by dividing the dataframe by 255 (the max value)

testing_df /= 255
training_df /= 255


# **Displaying Sample Data**
# 
# Let's reshape the data that we have into images instead of rolled out pixel values. This is accomplished using the reshape method. We do this because we're dealing with images and would want to use the CNN methods for our neural network.

# In[ ]:


#reshape the rows into (28x28) since that is the original image format for mnist data
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1 #this would be 3 for an RGB image, 4 for RGBA, but just 1 for grayscale image

#reshape the data into image shape
training_values = training_df.values.reshape( training_df.shape[0], IMG_WIDTH, IMG_HEIGHT )


# In[ ]:


plt.imshow(training_values[0], cmap=plt.get_cmap('gray'))
plt.title( training_labels[0] )


# **Reshaping data for CNN**
# 
# While a grayscale image has an implicit single channel for the grayscale, we need to explicitly set this before handing the data over to our CNN so that it will be able to process it properly

# In[ ]:


training_values = training_df.values.reshape( training_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )
testing_values = testing_df.values.reshape( testing_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )


# **One Hot Categorical Encoding**
# 
# The data that we have needs to be able to fit into one of 10 categories in the output layer so that we can categorize the output. Right now we don't have our labels in categories - we just have an array of results. What we need is for each element to be stretched so that each output represents each of the 10 possibilities that we have and the actual output is marked properly. 
# 
# This categorical encoding where we represent all of the categories and the correct output category is marked as "hot" is referred to as one-hot encoding.

# In[ ]:


print(training_labels.shape)
print(training_labels[:10])


# In[ ]:


NUM_CLASSES = 10
training_labels_categorical = tf.keras.utils.to_categorical(training_labels)
testing_labels_categorical = tf.keras.utils.to_categorical(testing_labels)

print( training_labels_categorical.shape )
print( testing_labels_categorical.shape )


# Here we can see that in the old training_label format we simply represented a single value for the output. As such the network would not be able to output to a set of probabilities for each of the possible outcomes.
# 
# In training_labels_new, the network can output to each of the possible outcomes with the knowledge that one of them is the the appropriate outcome and can use this to know to backprop error for results that don't land on the correct label. Note that the results of training_labels[0] results in that items index in the training_labels_categorical being marked as 1.

# In[ ]:


print( training_labels[0] )
print( training_labels_categorical[0])


# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam ,RMSprop

print("Tensorflow Version: " + tf.version.VERSION)
print("Keras Version: " + tf.keras.__version__)

# set the number of epochs for training the models
EPOCHS=35

# how many samples will the system see at one time before it updates its weights
BATCH_SIZE=64


# **Keras Models**
# 
# Now that we have all of the data properly formatted, it is time to build a model that will analyze the data and learn the patterns that are in it

# In[ ]:


model = tf.keras.Sequential()

model.add(layers.Conv2D(32, 3, 3, activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)))
model.add(layers.Conv2D(32, 3, 3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.50))     
model.add(layers.Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()        


history = model.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )


# In[ ]:


plt.plot( history.history['acc'])
plt.plot( history.history['val_acc'])
plt.title( 'Model accuracy')
plt.ylabel( 'accuracy')
plt.xlabel( 'epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


score = model.evaluate( testing_values, testing_labels_categorical, verbose=1)

print( "Model1 Score: " + str(score) )


# 
# **Model 2**
# 

# In[ ]:


model2 = tf.keras.Sequential()

model2.add(layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)))
model2.add(layers.MaxPool2D())
model2.add(layers.Dropout(0.40))

model2.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model2.add(layers.MaxPool2D())
model2.add(layers.Dropout(0.40))

model2.add(layers.Flatten())
model2.add(layers.Dense(128, activation="relu"))
model2.add(layers.Dropout(0.40))  

model2.add(layers.Dense(10, activation="softmax"))

model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.summary()   

history = model2.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )


# In[ ]:


plt.plot( history.history['acc'])
plt.plot( history.history['val_acc'])
plt.title( 'Model accuracy')
plt.ylabel( 'accuracy')
plt.xlabel( 'epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


score = model2.evaluate( testing_values, testing_labels_categorical, verbose=1)

print( "Model2 Score: " + str(score) )


# **Model 3**
# 

# In[ ]:


model3 = tf.keras.Sequential()

model3.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model3.add(layers.BatchNormalization())

model3.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model3.add(layers.BatchNormalization())

model3.add(layers.MaxPool2D(pool_size=(2,2)))
model3.add(layers.Dropout(0.25))

model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model3.add(layers.BatchNormalization())

model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model3.add(layers.BatchNormalization())
model3.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model3.add(layers.Dropout(0.25))

model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model3.add(layers.BatchNormalization())
model3.add(layers.Dropout(0.25))

model3.add(layers.Flatten())
model3.add(layers.Dense(256, activation = "relu"))
model3.add(layers.BatchNormalization())
model3.add(layers.Dropout(0.25))

model3.add(layers.Dense(10, activation = "softmax"))


model3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model3.summary()

history = model3.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )


# In[ ]:


plt.plot( history.history['acc'])
plt.plot( history.history['val_acc'])
plt.title( 'Model accuracy')
plt.ylabel( 'accuracy')
plt.xlabel( 'epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


score = model3.evaluate( testing_values, testing_labels_categorical, verbose=1)

print( "Model3 Score: " + str(score) )


# **Model 4**
# 
# In Model 4 we will perform some Data Augmentation to provide more training data for the system. Having more training data is always useful when training a neural network. Data Augmentation takes an existing set of training data and returns a *new* dataset. It is important to note that the original data is **NOT** included in the returned dataset.
# 
# We will also use learning rate decay. Decay will help us to get closer to a minimum by taking smaller steps later in the learning process. If we keep the learning rate too high, we may skip over a minimum. If you are confused by this, look at how back prop works.

# In[ ]:


modelCheckPoint = ModelCheckpoint( filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                   monitor="val_acc",
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')

# generate more training data
datagen = ImageDataGenerator(
        rotation_range= 8,  
        zoom_range = 0.13,  
        width_shift_range=0.13, 
        height_shift_range=0.13)


initial_learning_rate = 0.001
optimizer = Adam(lr=initial_learning_rate, decay= initial_learning_rate / (EPOCHS*1.3))

model4 = tf.keras.Sequential()

model4.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model4.add(layers.BatchNormalization())

model4.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model4.add(layers.BatchNormalization())

model4.add(layers.MaxPool2D(pool_size=(2,2)))
model4.add(layers.Dropout(0.25))

model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model4.add(layers.BatchNormalization())

model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model4.add(layers.BatchNormalization())
model4.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model4.add(layers.Dropout(0.25))

model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model4.add(layers.BatchNormalization())
model4.add(layers.Dropout(0.25))

model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation = "relu"))
model4.add(layers.BatchNormalization())
model4.add(layers.Dropout(0.25))

model4.add(layers.Dense(10, activation = "softmax"))


model4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model4.summary()

history = model4.fit_generator(  datagen.flow(training_values, training_labels_categorical, batch_size=BATCH_SIZE), validation_data=(testing_values, testing_labels_categorical), steps_per_epoch=training_values.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[modelCheckPoint] )


# In[ ]:


plt.plot( history.history['acc'])
plt.plot( history.history['val_acc'])
plt.title( 'Model accuracy')
plt.ylabel( 'accuracy')
plt.xlabel( 'epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( ['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


score = model4.evaluate( testing_values, testing_labels_categorical, verbose=1)

print( "Score: " + str(score) )


# In[ ]:


#prepare the submission data
submission_df /= 255
submission_values = submission_df.values.reshape( submission_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )

# model prediction on test data
predictions = model4.predict_classes(submission_values, verbose=0)

# submission
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)


# System is completed and should have predicted the classes for the submission_data that it has never seen before. This can be downloaded in the file below.

# <a href="./DR.csv"> Download File </a>
