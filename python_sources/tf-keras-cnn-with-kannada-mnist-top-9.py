#!/usr/bin/env python
# coding: utf-8

# # Kannada MNIST: Simple CNN

# ### Import Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers

from sklearn.model_selection import train_test_split


# In[ ]:


seed = 18
np.random.seed(seed)


# In[ ]:


import warnings
warnings.filterwarnings('ignore') # ignore warnings


# In[ ]:


tf.test.gpu_device_name() # testing gpu


# ### The data

# In[ ]:


dig_mnist = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
train = pd.read_csv("../input/Kannada-MNIST/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.shape


# * 28x28 dimension flatten into 784 columns
# * 60000 examples in training
# * First column is the label

# In[ ]:


# Extracting label from training set
x = train.iloc[:,1:].values
y = train.label.values


# In[ ]:


sns.countplot(y) # Checking if it is homogenous


# It is homogenous, so no need to augment specific labels

# In[ ]:


test.head()


# We can just remove the "id" columns

# In[ ]:


x_test = test.drop("id", axis=1).iloc[:,:].values


# In[ ]:


dig_mnist.head()


# Same process that was used in the training set.

# In[ ]:


x_dig = dig_mnist.drop('label', axis=1).iloc[:,:].values
y_dig = keras.utils.to_categorical(dig_mnist.label)


# We now need to reshape the data from the 784 flat to the 28x28x1.

# In[ ]:


x.shape


# In[ ]:


x = x.reshape(x.shape[0], 28, 28, 1)
y = keras.utils.to_categorical(y, 10) # and transform y into vectors


# In[ ]:


x.shape


# In[ ]:


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_dig = x_dig.reshape(x_dig.shape[0], 28, 28, 1)


# ### Training preparation

# We also need to split our training set into true training and validation. We will use 10% of our dataset as validation.  

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = seed)


# We will use data augmentation to try to increase the performance of our model out of sample. 

# In[ ]:


# This values can change and were inspired by those used in other submissions.
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 10,
                                   zoom_range = 0.1,
                                   horizontal_flip = False)


# In[ ]:


# We also need to create one for the validation data set, but this will only rescale the image
val_datagen = ImageDataGenerator(rescale=1./255.)


# ### Model

# In[ ]:


# Different models were tested and inspired by other submissions
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding="same", input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64, (3,3), padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
   
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(256, (3,3), padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(256, (3,3), padding="same"),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])


# In[ ]:


# Let's check our model
model.summary()


# We can now define our optimizer and compile the model.

# In[ ]:


initial_learningrate=0.001
batch_size = 1024
epochs = 50


# In[ ]:


optimizer = RMSprop(learning_rate=initial_learningrate,
                   momentum=0.1,
                   centered=True,
                   name="RMSprop")


# In[ ]:


model.compile(loss="categorical_crossentropy", # We will use categorial crossentropy as usual 
              optimizer=optimizer,
              metrics=['accuracy'])


# We will also define 2 callbacks, one for reducing LR and another for early stopping, both when val_loss is no longer improving.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor="loss",
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00001)

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20, restore_best_weights=True)


# We are now ready to run our model.

# In[ ]:


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train)//batch_size,
    epochs=epochs,
    validation_data=val_datagen.flow(x_val, y_val),
    validation_steps=50,
    callbacks=[learning_rate_reduction, es])


# ### Model evaluation

# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()
plt.show()


# In[ ]:


x_dig = x_dig/255
x_dig = x_dig.reshape(x_dig.shape[0],28,28,1)


# In[ ]:


model.evaluate(x_dig,y_dig,verbose=2)


# ### Submission

# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


predictions = model.predict_classes(x_test/255.)


# In[ ]:


submission['label'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

