#!/usr/bin/env python
# coding: utf-8

# NOTE: I rewrite various notebooks because that's how I learn. I do it on Kaggle because I like their community and other features. Please use and credit original source.
# 
# Source: https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/save_and_restore_models.ipynb

# # Save and restore models
# 
# Model progress can be saved during - and after - training. This means a model can resume where it left off and avoid long training times. Saving also means you can share your model and others can recreate your work. When publishing research models and techniques, most machine learning practitioners share:
# - code to create the model, and
# - the trained weights, or parameters, for the model
# 
# Sharing this data helps others understand how the model works and try it themselves with new data.
# 
# Caution: Be careful with untrusted code - TensorFlow models are code. See [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) for details.
# 
# ### Options
# There are different ways to save TensorFlow models - depending on the API you're using. This guide uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in TensorFlow. For other approaches, see the TensorFlow [Save and Restore](https://www.tensorflow.org/guide/saved_model) guide or [Saving in eager](https://www.tensorflow.org/guide/eager#object_based_saving).
# 
# ## Setup
# 
# ### Installs and imports
# 
# Install and import TensorFlow and dependencies:

# In[ ]:


from __future__ import absolute_import, division, print_function

import os


# In[ ]:


import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# In[ ]:


keras = tf.keras


# ### Get an example dataset
# 
# We'll use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to train out model to demonstraate saving weights. To speed up these demonstration runs, only usee the first 1000 examples:

# In[ ]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ### Define a model
# 
# Let's build a simple model we'll use to demonstrate saving and loading weights.

# In[ ]:


# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    return model

# Create a basic model instance
model = create_model()
model.summary()


# ## Save checkpoints during training
# 
# The primary use case is to automatically save checkpoints *during* and at the *end* of training.  This way you can use a trained model without having to retrain it, or pick-up training where you left off - in caes the training process was interrupted.
# 
# `tf.keras.callbacks.ModelCheckpoint` is a callback that performs this task. This callback takes a couple of arguments to configure checkpointing.
# 
# ### Checkpointing callback usage
# 
# Train the model and pass it the `ModelCheckpoint` callback.

# In[ ]:


checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data = (test_images, test_labels),
         callbacks = [cp_callback]) # pass callback to training


# This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:

# In[ ]:


get_ipython().system('ls {checkpoint_dir}')


# Create a new, untrained model. When restoring a model from only weights, you must have a model with the same architecture as the original model. Since it's the same model architecture, we can share weights despite that it's a different *instance* of the model.
# 
# Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):

# In[ ]:


model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print('Untrained model, accuracy: {:5.2f}%'.format(100*acc))


# Then load the weights from the checkpoint, and re-evaluate:

# In[ ]:


model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# ### Checkpoint callback options
# 
# The callback provides several  options to give the resulting checkpoints unique names, and adjust the checkpointing frequency.
# 
# Train a new model, and save uniquely named checkpoints every 5-epochs:

# In[ ]:


# include the epoch in the file name. (use `str.format`)
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback =  tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 period=5) #  save weights everry 5 epochs

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
         epochs=50, callbacks=[cp_callback],
         validation_data =  (test_images, test_labels),
         verbose=0)


# Now, look at the resulting checkpoints and choose the latest one:

# In[ ]:


get_ipython().system('ls {checkpoint_dir}')


# In[ ]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# Note: the default tensorflow format only saves the 5 most recent checkpoints.
# 
# To test, reset the model and load the latest checkpoint:

# ## What are these files?
# 
# The above code stores the weights to a collection of [checkpoint](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)-formatted files that contain only the trained weights in a binary format. Checkpoint contain:
# - One or moree shards that contain your model's weights
# - An index file that indicates whcih weights are stored in which shard.
# 
# If you are only training a model on a single machine, you'll have one shard with the suffix: `.data-00000-of-00001`

# ## Manually save weights
# 
# Above you saw how to load the weights into a model.
# 
# Manually saving the weights is just as simple, use the `Model.save_weights` method.

# In[ ]:


# save the weights
model.save_weights('./checkpoints/my_checkpoint')

# restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# ## Save the entire model
# 
# The entire model can be saved to a file taht contains the weight values, the model's configuration, and even the optimizer's configuration (depends on set up). This allows you to checkpoint a model and resume training later -  from the exact same state - without access to the original code.
# 
# Saving a full-functional model is very useful - you can load them in TensorFlow.js ([HDF5](https://js.tensorflow.org/tutorials/import-keras.html), [Saved Model](https://js.tensorflow.org/tutorials/import-saved-model.html)) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow lite ([HDF5](https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_), [Saved Model](https://www.tensorflow.org/lite/convert/python_api#exporting_a_savedmodel_)).
# 
# ### As an HDF5 file
# 
# Keras provides a basic save format using the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) standard. For our purposes, the saved model can be treated as a single binary blob.

# In[ ]:


model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from and HDF5 file.
model.compile(optimizer='adam',
             loss=tf.keras.losses.sparse_categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# save entire model to a HDF5 file
model.save('my_model.h5')


# Now recreate the model from that file:

# In[ ]:


# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()


# Check it's accuracy:

# In[ ]:


loss, acc = new_model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# This technique saves everything:
# - The weight values
# - The model's configuration (architecture)
# - The optimizer configuration
# 
# Keras saves models by inspecting the architecture. Currently, it is not able to save TensorFlow optimizers (from `tf.train`). When using those, you will need to re-compile the model after loading, and you will loose the state of the optimizer.

# # What's Next
# 
# That was a quick guide to saving and loading with `tf.keras`
# 
# - The [tf.keras guide](https://www.tensorflow.org/guide/keras) shows more about saving and loading models with `tf.keras`
# - See [Saving in eager](https://www.tensorflow.org/guide/eager#object_based_saving) for saving during eager execution
# - The [Save and Restore](https://www.tensorflow.org/guide/saved_model) guide has low-level details about TensorFlow saving.
