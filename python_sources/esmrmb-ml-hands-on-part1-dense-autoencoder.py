#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Tutorial 
# ## ESMRMB Lecture, Berlin September 2018
# ### Teodora Chitiboi, Sandy Engelhardt, Hans Meine 
# 
# This tutorial provides an introduction into Dense Autoencoders for reconstruction of MRI data.

# The following cell starts an external tool to download the example data set (this only needs to be done once):
# 
# > The 20 normal MR brain data sets and their manual
#   segmentations were provided by the Center for Morphometric Analysis at
#   Massachusetts General Hospital and are available at
#   http://www.cma.mgh.harvard.edu/ibsr/.

# The following lines import required functionality from the ["keras"](https://keras.io) and ["numpy"](http://www.numpy.org) packages:

# In[ ]:


from keras import layers
from keras.models import Model
import numpy as np


# We use Matplotlib for displaying results and images:

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load example data:

# In[ ]:


ibsr_data = np.load('../input/IBSR_v2_resampled_cropped_8bit_64x64.npz')
images = ibsr_data['input']
print('Dimensions and extents after loading:', images.shape)


# In[ ]:


# Reduce dimensions to (NumberOfDatasets, Height, Width)
dataset = np.squeeze(images, axis=1);
print('Dimensions after dimensionality reduction:', dataset.shape) # should be (2716, 64, 64)


# In[ ]:


# Divide dataset into train and test (order is already random)
x_train = dataset[:2300,:,:]
x_test  = dataset[2300:,:,:]

# Normalizing images to range [0...1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# In[ ]:


# Flattening images into 64*64 = 4096 vector
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('Flattened dimensions of training data:', x_train.shape)
print('Flattened dimensions of testing data: ', x_test.shape)


# ## Creating the model architecture

# In[ ]:


# mapping input vector with factor 128, (32*128 = 4096)
encode_dimension = 128

# Input placeholder
input_img = layers.Input(shape=(4096,))

# Encoding of the input
encoded_1 = layers.Dense(encode_dimension, activation='relu')(input_img)
encoded_2 = layers.Dense(encode_dimension, activation='relu')(encoded_1)

# Decoding/reconstruction of the input
decoded_1   = layers.Dense(4096, activation='sigmoid')(encoded_2)
decoded_2   = layers.Dense(4096, activation='sigmoid')(decoded_1)

# This maps the input to its reconstruction
# The aim is to fully recover the input image
autoencoder = Model(input_img, decoded_2)


# In[ ]:


autoencoder.summary()


# In[ ]:


# Set the optimizer (Adam is a popular choice), and the loss function
autoencoder.compile(optimizer = 'adam', loss='mean_squared_error')


# You may read up on further options for the loss (like `mean_absolute_error` or `binary_crossentropy`) at https://keras.io/losses/, and on other optimizers (such as `sgd` or `adadelta`) at https://keras.io/optimizers/.

# ## Train the autoencoder

# In[ ]:


# Potentially change num_epochs or batch_size
num_epochs = 25
history = autoencoder.fit(
    x_train, x_train,
    epochs=num_epochs,
    batch_size=16,
    shuffle=True,
    validation_data=(x_test, x_test))


# In[ ]:


# Test the autoencoder using the model to predict unseen data
decoded_imgs = autoencoder.predict(x_test)


# In[ ]:


# Following code is for displaying of results

n = 6 # number of images to display
plt.figure(figsize=(12, 4))
for i in range(n):
    # display image
    ax = plt.subplot(2, n, i + 1)
    ax.imshow(x_test[i].reshape(64, 64), cmap = 'gray')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(1.0)

    # display reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    ax.imshow(decoded_imgs[i].reshape(64, 64), cmap = 'gray')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect(1.0)
plt.show()


# In[ ]:


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# In[ ]:




