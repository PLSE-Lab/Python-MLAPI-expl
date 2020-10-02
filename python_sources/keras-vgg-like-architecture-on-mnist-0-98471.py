#!/usr/bin/env python
# coding: utf-8

# # Keras VGG like architecture on MNIST (0.98471)

# All the imports needed.
# 
# Using numpy for numerical computations
# Using pandas for data manipulation (storing and retrieving from csv)
# 
# Using Keras in the Tensorflow (tensorflow.python.keras)
# * to_categorical: To convert labels to one-hot encoding
# * keras.callbacks: Logging tools to visualize my training (Tensorflow) and save the intermediate models with the best accuracies (ModelCheckpoint).
# 
# Other things are pretty standard.

# In[1]:


import numpy as np
import pandas as pd

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the training and the test data using Pandas

# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')


# In[3]:


df_train.head()


# Extracting the training set using `to_categorical` and other functions.

# In[4]:


y_train_temp = to_categorical(df_train.iloc[:, 0].values, num_classes=10)
x_train_temp = df_train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)


# ### Shuffling the data 
# 
# Using `numpy.random.permutation` to shuffle in case the data is not shuffled already.

# In[5]:


permut = np.random.permutation(x_train_temp.shape[0])

x_train = x_train_temp[permut]
y_train = y_train_temp[permut]


# In[6]:


print("shape of X: {}, shape of y: {}".format(x_train.shape, y_train.shape))


# ### Visualizing the mean image of the dataset. 
# 
# It sometimes helps in visualizing the distribution of data.

# In[7]:


# Print the mean image
mean_image = np.mean(x_train, axis=0).astype(np.uint8).reshape(28, 28)
plt.imshow(mean_image, cmap='gray')


# ### Building the model architecture.
# 
# The model is VGG like, because of
# 
# * the pattern (CNN->MaxPool)->(CNN->MaxPool)->Dense->Dense
# * 'same' paddings
# * (3, 3) filter shape
# * number of filters are power of 2s and gets double.
# * Pooling with strides (2, 2). (It is default hence not explicitly mentioned in the code)

# In[8]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.summary()


# ### Compiling the model
# 
# * Adam optimizer with `0.0001` learning rate.
# * `categorical_crossentropy` loss, as this is a multi-class classification problem.
# * `accuracy` as the metrics as this is the easiest option to know how good your model is performing.

# In[9]:


adam = Adam(lr=1e-4, decay=1e-6)
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])


# ### Callbacks objects
# 
# This might be the most useful and new thing that you might see in this code.
# 
# * Using `Tensorboard` callback for visualizing loss.
# * Using `ModelCheckpoint` callback to save the model periodically.
# 
# **NOTE**: I used `save_best_only` so that I save only the recent best model saving space for me. Otherwise I had to save all the models per epoch. Then I could select which one I want.

# In[ ]:


get_ipython().system('mkdir models')
get_ipython().system('mkdir logs')


# In[10]:


tensorboard = TensorBoard(write_grads=True, write_images=True)
chkpoint = ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)


# ### Training the model
# 
# Finally I train the model for 5 epochs with the above compilation (learning rate = `0.0001` and all).
# 
# I am using 20% of the data as validation data. It is the data used only to verify the correctness of the model, not for training.

# In[11]:


model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard, chkpoint], validation_split=0.2)


# ### Training the model again with smaller learning rate
# 
# Using learning rate of `1e-5` now.

# In[12]:


adam = Adam(lr=1e-5, decay=1e-6)
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, callbacks=[tensorboard, chkpoint], validation_split=0.2)


# ### Loading the best model
# 
# Now that the training is complete. Now I can find the best model I have, and use it for prediction on the test set.

# In[22]:


# best_model = load_model('models/weights.01-0.05.hdf5') (Skipping for now as this is a mannual process)
best_model = model


# Extracting the data from the the test set

# In[23]:


x_test = df_test.values.reshape(-1, 28, 28, 1)
print('Test set: {}'.format(x_test.shape))


# ### Prediction
# 
# Now I can call the `predict` function to get the prdictions.
# 
# **Note** that these values are probabilities of a given class. Then I select the `idx` or the class id of the most probable class using `np.argmax`.

# In[24]:


probs = best_model.predict(x_test, verbose=1)
preds = np.argmax(probs, axis=1)

print("Predictions: {}".format(preds.shape))


# ### Creating the submission file
# 
# Here I create the submission file as described in the competition data page.

# In[25]:


submission = pd.DataFrame({'ImageId': np.arange(1, len(preds)+1), 'Label': preds})
submission.to_csv('submission_0.05.csv', index=False)


# In[ ]:




