#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install -U tensorboard_plugin_profile')


# In[ ]:


from datetime import datetime
from packaging import version

import os


# In[ ]:


import tensorflow as tf


#         **In this kernel I'm showing you some best practices in Deep Learning using TF 2x.
#         The things I'm going to show you in this kernel are:**
#         1. Introducing Tensorflow Profiler( Basics(How to analyze input pipelines)).
#         2. Then we're going to see what is the best practise to initialize weights of a model.
#         3. Transfer learning and effects of data Augmentation in accuracy.
#         

# Tensorflow Profiler is introduced in this year's TF Dev Summit. This is the same tool that Google's use for profiling their models
# Internally. And they made this tool Public in thhis year's Dev Summmit. 
# So, the TensorFlow Profiler provides a set of tools that you can use to measure the training performance and resource consumption of your TensorFlow models. And now it has been integerated in TensorBoard.

# In this kernel I'm going to introduce the TF profiler. We're going to see how to use TF profiler to Debug the Input Pipeline  and then we'll optimize it. We're going to use simplest dataset(Offcourse mnist) and a dummy model. 
# 
# First we'll train the dummy model and then we'll inspect it's training input pipeline using TF Profiler & finally we'll optimize it.
# 
# Let's load the mnist dataset using Tensorflow Datasets.

# In[ ]:


import tensorflow_datasets as tfds


# In[ ]:


(train, test), dataset_info = tfds.load(
    'mnist',
    split =['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True )


# In[ ]:


def rescale(image, label):
    return tf.cast(image, tf.float32) / 255., label
# rescaling the image
train  = train.map(rescale)
test = test.map(rescale)
# Batching the datasets
train = train.batch(128)
test = test.batch(128)


# In[ ]:


# Creating not so cool model :)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)


# Now we'll create a TensorBoard callback for computing the performance of the model and then we'll call it while training

# In[ ]:


logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

model.fit(train,
          epochs=2,
          validation_data=test,
          callbacks = [tboard_callback])


# In[ ]:


#Load the TensorBoard notebook extension.
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


# Launch TensorBoard and navigate to the Profile tab to view performance profile
get_ipython().run_line_magic('tensorboard', '--logdir=logs')


# After launching the tensorboard, switch over the profile tab.
# There you can see It's recommending some steps for optimisation.
# It's saying "Your program is HIGHLY input-bound because 77.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time. "
# 
# So, we've to optimize our input pipeline. Let's appply some optimising techniques of td.data.Datasets.
# 
# ![Screenshot%20%28137%29.png](attachment:Screenshot%20%28137%29.png)

# We're doing below optimisation techiniques:
# 
# 1. Caching: The tf.data.Dataset.cache transformation can cache a dataset, either in memory or on local storage. This will save some operations (like file opening and data reading) from being executed during each epoch
# 2. Prefetching: it overlaps the preprocessing and model execution of a training step. While the model is executing training step s, the input pipeline is reading the data for step s+1. Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
# 
# Now, let's us put this methods into actions

# In[ ]:


#again loading the datasets.
(train, test), dataset_info = tfds.load(
    'mnist',
    split =['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True )

# Creating the optimized input pipeline.
def rescale(image, label):
    return tf.cast(image, tf.float32) / 255., label
# rescaling the image
train  = train.map(rescale)
test = test.map(rescale)


train = train.batch(128)
# applying cache in training set
train = train.cache()
# applying prefetching in training sets
train = train.prefetch(tf.data.experimental.AUTOTUNE)


test = test.batch(128)
# applying cache in test set
test = test.cache()
# applying prefetching in test set
test = test.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


# again making and training the not so cool model but this time we'are using the optimised training

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(train,
          epochs=2,
          validation_data=test,
          callbacks = [tboard_callback])


# In[ ]:


# Launching the tensorboard.
get_ipython().run_line_magic('tensorboard', '--logdir=logs')


# But this time you can see, it's saying our program is not input bound. Thanks to TF Profiler :) .
# ![Screenshot%20%28139%29.png](attachment:Screenshot%20%28139%29.png)

# ### Now let's move to next section of the notebook, here we'll be seeing what is the practise to initialise the weights of the models.
# 
# First we're going to train a resnet model from sractch by initialiing the weights 

# In[ ]:


# downloading tf_flowers it contains 5 classes of 5 differebt flowers species.

train, train_info = tfds.load('tf_flowers', split='train[:80%]', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('tf_flowers', 
                          split='train[80%:]', 
                          as_supervised=True, 
                          with_info=True)


# In[ ]:


# preprocessing and making input pipeline
def resize(img, lbl):
  img_size = 224
  return (tf.image.resize(img, [img_size, img_size])/255.) , lbl

train = train.map(resize)
val = val.map(resize)

train = train.batch(32, drop_remainder=True)
val = val.batch(32, drop_remainder=True)


# In[ ]:


# function to create resnet model with random weights and imagenet weights
def resnet(imagenet_weights=False):
    num_classes = 5
    if imagenet_weights is False:
        return tf.keras.applications.ResNet50(include_top=True, 
                                        input_shape=(224, 224, 3), 
                                        weights=None, 
                                        classes=num_classes)
    if imagenet_weights is True:
        resnet = tf.keras.applications.ResNet50(include_top=False, 
                                        input_shape=(224, 224, 3), 
                                        weights='imagenet', 
                                        )
        resnet.trainable = True
        return  tf.keras.Sequential([resnet, 
                                     tf.keras.layers.GlobalAvgPool2D(), 
                                     tf.keras.layers.Dense(5, activation='softmax')])


# In[ ]:


# training the model in gpu if availiable
def try_gpu(i=0): 
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)


# In[ ]:


with strategy.scope():
  model = resnet(imagenet_weights=False)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])


# In[ ]:


# Saving validation accuracy ib the variable val_acc_1
val_acc_1 = history.history['val_accuracy']


# Now let's train the model again, but this time we're going to initialize the 
# weights of our resnet model to the imagenet weights(trained on the imagenet datasets).

# In[ ]:


with strategy.scope():
  model = resnet(imagenet_weights=True)

# compiling and training.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])


# In[ ]:


# Saving validation accuracy of this model in the variable val_acc_2
val_acc_2 = history.history['val_accuracy']


# Comparing the results of the validation accuracy of the first model whose weights are initialized randomly and the model which weights are not initialized randomly but with the weights of some model which is trained
# on similiar datasets(here imagenet).

# In[ ]:


print("Validation accuracy of first approach is {} VS Validation accuracy of secon approach is {}".format(max(val_acc_1), max(val_acc_2)))


# You can clearly see the accuracy of second approach is much more higher than the first one 

# In this section we're applying data some augmentation techniques and use transfer learning to see the impact of this technique in our validation accuracy.

# In[ ]:


# loading the tf_flowers dataset
train, train_info = tfds.load('tf_flowers', split='train[:80%]', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('tf_flowers', 
                          split='train[80%:]', 
                          as_supervised=True, 
                          with_info=True)


# In[ ]:


#function to augment the images
def augment(image,label):
  image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.image.random_flip_up_down(image)  # Randomily flips the image up and down 


  return image,label


# In[ ]:


train = train.map(augment).map(resize).batch(32, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
val = val.map(resize).batch(32, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


for x,y in train.take(1):
    print(x.shape)


# In[ ]:


def resnet_transfer_learning():
    # including_top=False means we're the last layer of resnet will not include fully connected dense layer
    resnet = tf.keras.applications.ResNet50(include_top=False, 
                                        input_shape=(224, 224, 3), 
                                        weights='imagenet', 
                                        )
    # freezing the layers of resnet.
    resnet.trainable = True
    # adding the dense layer to the resnet model
    return  tf.keras.Sequential([resnet, 
                                     tf.keras.layers.GlobalAvgPool2D(), 
                                     tf.keras.layers.Dense(5, activation='softmax')])


# In[ ]:


with strategy.scope():
  model = resnet_transfer_learning()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = callback = tf.keras.callbacks.EarlyStopping(patience=8)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])


# In[ ]:


# Saving validation accuracy of this model in the variable val_acc_2
val_acc_3 = history.history['val_accuracy']


# Now let's compare the results of all three aproaches.

# In[ ]:


print("Validation accuracy of first approach is {} VS Validation accuracy of second approach is {} VS Validation accuracy of second approach is {}".format(max(val_acc_1), max(val_acc_2), max(val_acc_3)))


# In this kernel we've first see the introduction of TF profiler and then we trained
# the resnet50 model with scratch by initializing the model weights randomaly vs initializing the model weights with the wieghts of some
# model which is trained on similiar tasks. And then we compare the results.
# Atlast we used image augmentation technique and used transfered learning and we comapared the results of these 3 approaches.
# 
# Atlast try fine tuning your model, use hyperparameter tuning and schedule the learning rates.

# Refrences: Tensorflow official documentations.
