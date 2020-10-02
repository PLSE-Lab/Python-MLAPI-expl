#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)


# # 1. Data preparation

# ## 1.1 Load Data

# In[ ]:


# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1) # all except for label column


g = sns.countplot(Y_train)
Y_train.value_counts() # 


# We have ap. even number of all digits

# ## 1.2 Check for null values

# In[ ]:


X_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# No missing values in the train and test datasets.

# ## 1.3 Grayscale Normalization

# In[ ]:


# from 0-255 to 0-1
X_train = X_train / 255.0
test = test / 255.0


# ## 1.4 Reshaping

# In[ ]:


# Reshape image  (height = 28px, width = 28px , canal = 1)
# use this reshaping for CNN in Section 2.1
#X_train = X_train.values.reshape(-1,28,28,1)
#test = test.values.reshape(-1,28,28,1)


# In[ ]:


# use this reshape for 
X_train = X_train.values.reshape(-1,28,28)
test = test.values.reshape(-1,28,28)


# All data are in 28x28x1 3D matrices now (Keras requires one more dimension in the end which correspond to channels, only one channel is needed for gray scale images.

# ## 1.5 Label encoding 

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
#Y_train = to_categorical(Y_train, num_classes = 10) <- 


# We need to encode these lables to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]).
# (only needed for model from Section 2.1, not for 2.2 !)

# ## 1.6 Splitting into training and valid
# ation set

# In[ ]:


random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.15, random_state=random_seed)


# ## 1.7 Example Images

# In[ ]:


#g = plt.imshow(X_train[0][:,:,0])


# # 2. Modeling
# ## 2.1 CNN model 

# In[ ]:


## In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
#
#model = Sequential()
#model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                 activation ='relu', input_shape = (28,28,1))) #32 filters for the first conv2D layer
#model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                 activation ='relu')) #32 filters for the 2nd conv2D layer
#model.add(MaxPool2D(pool_size=(2,2))) #the area size pooled each time
#model.add(Dropout(0.25)) #regularization parameter= proportion of nodes in the layer are randomly ignored (setting their weights to zero) for each training sample. 
#
#model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                 activation ='relu'))
#model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.25))
#
#model.add(Flatten()) #to convert the final feature maps into a one single 1D vector
#model.add(Dense(256, activation = "relu")) #activation function max(0,x), used to add non linearity to the network
#model.add(Dropout(0.5))
#model.add(Dense(10, activation = "softmax")) #outputs distribution of probability of each class


# In[ ]:


# Define the optimizer
# loss function and an optimisation algorithm
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # faster than Stochastic Gradient Descent ('sgd') optimizer


# In[ ]:


# Compile the model
#model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Set a learning rate annealer
# reduce the LR by half if the accuracy is not improved after 3 epochs
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                            patience=3, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0.00001)
#epochs = 10 # 
#batch_size = 86


# In[ ]:


#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 2)


# Validation accuracy = 0.9905

# ## 2.2 Alternative (Simpler) Model

# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.997):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True


# In[ ]:


callbacks = myCallback()
model2 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model2.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
# model fitting
history = model2.fit(X_train, Y_train, epochs=30, callbacks=[callbacks])


# # Submitting predictions

# In[ ]:


# predict results
results = model2.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

