#!/usr/bin/env python
# coding: utf-8

# # MNIST with tf.keras, tf.data, and eager execution
# Code: https://github.com/tensorflow/workshops

# In[ ]:


import numpy as np
import tensorflow as tf

# Enable eager execution
tf.enable_eager_execution()


# In[ ]:


# custom function to load data from local file path rather than using `tf.keras.datasets.mnist.load_data()`
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# In[ ]:


(train_images, train_labels), (test_images, test_labels) = load_data('../input/mnist.npz')


# In[ ]:


TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Reshape from 2D to single dimension, (N, 28, 28) to (N, 784)
train_images = np.reshape(train_images, (TRAINING_SIZE, 784))
test_images = np.reshape(test_images, (TEST_SIZE, 784))

# Convert the array to float32 as opposed to uint8
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
train_images /= 255
test_images /= 255


# In[ ]:


NUM_DIGITS = 10

print('Before', train_labels[0]) # labels format before conversion
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_DIGITS)

print('After', train_labels[0]) # labels format after conversion
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_DIGITS)


# In[ ]:


# cast the labels to floats, needed later
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)


# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# create a tensorflow optimizer, rather than using the Keras version
# this is currently necessary when working in eager mode
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

# we will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


# ## Create a tf.data dataset
# First we will convert numpy arrays into a TensorFlow dataset. Then we will create a simple for loop that will serve as our introduction into creating custom training loops. Although this essentially does the same as `model.fit`, it allows us to get creative and customize the overall training process and collect different metrics throughout the process.

# In[ ]:


BATCH_SIZE=128
# Because tf.data may work with potentially _large_ collections of data, we do not shuffle the entire dataset by default. Instead we maintain a buffer of SHUFFLE_SIZE elements and sample from there.
SHUFFLE_SIZE=10000

# create the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)


# ## Iterate over the dataset
# Here, we'll iterate over the dataset and train our model using `model.train_on_batch`.

# In[ ]:


EPOCHS=5

for epoch in range(EPOCHS):
    for images, labels in dataset:
        train_loss, train_accuracy = model.train_on_batch(images, labels)
        
    # here we can gather any metrics or adjust our training parameters
    print('Epoch %d\t Loss: %.6f\t Accuracy: %.6f' % (epoch + 1, train_loss, train_accuracy))


# In[ ]:


loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy: %.2f' % (accuracy))


# ### Congratulations!
# We have just trained a model on MNIST using tf.keras, tf.eager and tf.data. Fork it and try with other sources, or learn more about these APIs in the documentations.
