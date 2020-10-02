#!/usr/bin/env python
# coding: utf-8

# **CNN keras model in tensorflow.**
# Here you'll learn how to create a simple convolutional keras model inside the tensorflow interface.
# Follow the commented lines on the code for explaination. This non code blocks will indicate section breaks.

# In[ ]:


#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import math
#let's begin by printing the tensorflow version
print(tf.__version__)
#We can define the paths to the train, test and potencially evaluation datasets here
train_file='/kaggle/input/fashionmnist/fashion-mnist_train.csv'
test_file='/kaggle/input/fashionmnist/fashion-mnist_test.csv'
#I like to define the size of the input images globally to reuse the code in other ML problems
img_width=28
img_height=28
print('Starting analysis')
#It will be latter used to determine the buffer size automatically
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Import train and test datasets using pandas

# In[ ]:


train_dataset=pd.read_csv(train_file,sep=',')
test_dataset=pd.read_csv(test_file,sep=',')
print('done')


# Define train and test data

# In[ ]:


def get_image(df):
    df=tf.reshape(df,(img_width,img_height,1))
    return df
def data_augmentation(image):
    #train images are augmented (rezised, rotated, flipped, etc) to increase the train dataset variety and avoid overfitting.
    image=tf.image.resize_images(image, (tf.random_uniform([],img_width*0.8,img_width*1.1),tf.random_uniform([],img_height*0.8,img_height*1.1)))
    image=tf.contrib.image.rotate(image,tf.random_uniform([], -0.25, 0.25) ,interpolation='NEAREST')
    image=tf.image.random_contrast(image,0.6,1.4,seed=None)
    image=tf.image.random_brightness(image,0.2,seed=None)
    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image=tf.image.per_image_standardization(image)
    return image
def data_standarization(image):
    image=tf.image.per_image_standardization(image)
    return image
def read_images(batch_size):
    #get the number of elements on each data set
    num_train_examples = len(train_dataset.iloc[0:,0].values)
    num_test_examples = len(test_dataset.iloc[0:,0].values)
    #convert the datasets into labels and images both in tensor format
    trimagebit =tf.data.Dataset.from_tensor_slices(train_dataset.iloc[0:,1:].values)
    trlabels =tf.data.Dataset.from_tensor_slices(train_dataset.iloc[0:,0].values)
    teimagebit =tf.data.Dataset.from_tensor_slices(test_dataset.iloc[0:,1:].values)
    telabels =tf.data.Dataset.from_tensor_slices(test_dataset.iloc[0:,0].values)
    # Convert images tensors to the correct shape
    trimage = trimagebit.map(get_image)
    teimage = teimagebit.map(get_image)
    #train images are augmented (rezised, rotated, flipped, etc) to increase the train dataset. (Helps to avoid overtfitting)
    trimage = trimage.map(data_augmentation,num_parallel_calls=AUTOTUNE)
    #test images are not augmented instead they are only standarized.
    teimage = teimage.map(data_standarization,num_parallel_calls=AUTOTUNE)
    # Create batches and zip the images and labels in the correct format for train
    trds =tf.data.Dataset.zip((trimage,trlabels))
    trds = trds.repeat()
    trds = trds.batch(batch_size)
    trds = trds.prefetch(buffer_size=AUTOTUNE)
    teds =tf.data.Dataset.zip((teimage,telabels))
    teds = teds.repeat()
    teds = teds.batch(batch_size)
    teds = teds.prefetch(buffer_size=AUTOTUNE)
    return trds,teds,num_train_examples,num_test_examples
print('functions defined')


# Create keras model

# In[ ]:


def create_model():
    model = tf.keras.Sequential([
    #relu activation function is the standar for hidden layers and softmax is the standar for output layer in classification tasks.
    #this model is inpired in the VGG architecture, with (3,3) convolution layers and (2,2) max polling layes alternated in blocks. Of course the size is greatelly reduced because here the dataset and image sizes are small. 
    #The SpatialDropout2D and Dropout layers are similar in purpose and help to make the neural network robust by avoid overspecialitation and excesive dependence on specific neuronal paths.
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,input_shape=(img_height,img_width,1)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    #end of the first block
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Dropout(rate=0.25),
    #end of the second bock
    #the result is flettened in order to introduce it to dense layers (1-dimensional neuron layers)
    tf.keras.layers.Flatten(),
    #dense hidden layer with 256 neurons
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.5),
    #output layer, the numer of neurons should be equal to the number of classes in the dataset.
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
print('model defined')


# Prepare data for train

# In[ ]:


#Here is where we actually define the batch size and call the function we prepared for creating the tensors from the pandas dataset. 
batch_size = 50
print('Loading the data')
# Build the data input
trds,teds,num_train_examples,num_test_examples= read_images(batch_size)
print('Traning events:' + str(num_train_examples))
print('Testing events:' + str(num_test_examples))
print('data successfully loaded')


# Train the model

# In[ ]:


print('starting training')
model=create_model()
#Note, here I'm not saving the model, but you can save it by implementing callbacks.
model.fit(trds, epochs=20,steps_per_epoch=math.floor(num_train_examples/batch_size),validation_data=teds,validation_steps=math.floor(num_test_examples/batch_size))
print('------------------Done-----------------')


# Tha final accuracy I got after 20 epoch is 0.9264, if we optimize or change the model it might go up.
