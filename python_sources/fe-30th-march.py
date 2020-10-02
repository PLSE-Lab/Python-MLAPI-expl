#!/usr/bin/env python
# coding: utf-8

# In[40]:


get_ipython().system('pip install -U tensorflow_datasets')


# In[41]:


from __future__ import absolute_import, division, print_function


# Import TensorFlow and TensorFlow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm


print(tf.__version__)

# This will go away in the future.
# If this gives an error, you might be running TensorFlow 2 or above
# If so, the just comment out this line and run this cell again
tf.enable_eager_execution()  
#importing libraries that are necessary


# **IMPORT Fashion MNIST Dataset**

# In[42]:


dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


# tfds is a imported tensorflow dataset, and we'd like to load 'fashion_mnist'and then divide it into train and test dataset. 
# 
# cf) different approach from the MNIST example
# (x_train, y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

# **EXPLORE the DATA"

# In[43]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


# In[44]:


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


# **PROCESS THE DATA normalizing the data**
# like the case in the MNIST case (we divided every data by 255 (scaling)
# in this case we 'defined' a function called 'normalize' 

# In[45]:


def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)


# i=i+1 instead **i+=1**
# images=images/255 instead **images /= 255**

# In[46]:


# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))
#image.numpy().reshape creates an image and .take means take 1 image 

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()


# In[47]:


plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()


# **NOW MODELING STARTS until now we explored the data**

# In[48]:


#set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])


# 1. tf.keras.layers.Flatten(input_shape=(28, 28, 1))
# it means we'd like to 'flatten' a 28 by 28 grey scale image into a 1 demensional vector
# 2.  tf.keras.layers.Dense(128, activation=tf.nn.relu)
# a dense layer where takes every node created in a previous flattened layer weighting that input according to hidden parameters which will be learned during training, and outputs a single value to the next layer.
# 
# 3.  tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
# a dense output layer which has 10 (t-shirts to ankel boots) output and then with softmax function it has a form of probability

# **COMPILE THE MODEL**

# In[49]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# loss='sparse_categorical_crossentropy' is correspondent to the MSE in our celcius and Farenheit example
# optimizer= 'Adam'
# Metrics= Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# **TRAIN THE MODEL**

# In[50]:


BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# BATCH = 32 means we'd like to group 32 images and then test at each training steps. 

# colab)
# 1. Repeat forever by specifying dataset.repeat() (the epochs parameter described below limits how long we perform training).
# 2. The dataset.shuffle(60000) randomizes the order so our model cannot learn anything from the order of the examples.
# 3. And dataset.batch(32) tells model.fit to use batches of 32 images and labels when updating the model variables.

# **MODEL FITTING**

# In[51]:


model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))


# whole data/batch size=1 epoch 
# 60000/32=1875 
# the acc:0.89 means we are able to predict with 89% confidence with "TRAINIG DATASET"

# **MAKE PREDICTIONS AND EXPLORE**

# In[52]:


test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)


# In[53]:


for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)


# test dataset/test dataset batch size
# 10,000/32=313

# In[54]:


predictions.shape


# In[55]:


predictions[0]


# In[56]:


np.argmax(predictions[0])


# In[57]:


test_labels[0]


# In[58]:


def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[59]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[ ]:




