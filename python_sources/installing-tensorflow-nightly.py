#!/usr/bin/env python
# coding: utf-8

# # Look at dependents

# In[ ]:


get_ipython().system('pip show tensorflow')


# TensorFlow is required by only `fancyimpute`. Upgrading TensorFlow to nightly may break this package (if new version includes a non-backward compatible change used by this package). Probably not a big deal, this isn't an important package.
# 
# Once we add TensorFlow addons, it may break it.
# 
# It may reinstall different versions of its dependencies (see `Requires` above) which may in turn break other packages. Users should be cognizant that we won't guarantee that all packages are working after running the Pytorch script to install TensorFlow nightly.

# # Install TensorFlow nightly

# In[ ]:


get_ipython().system('pip install tf-nightly')


# # Test
# 
# Run a simple test to ensure TensorFlow installation succeeded.

# In[ ]:


import numpy as np
import tensorflow as tf

x_train = np.random.random((100, 28, 28))
y_train = np.random.randint(10, size=(100, 1))
x_test = np.random.random((20, 28, 28))
y_test = np.random.randint(10, size=(20, 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)

