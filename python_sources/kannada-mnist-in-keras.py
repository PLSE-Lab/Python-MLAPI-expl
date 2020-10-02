#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout,Activation,LeakyReLU
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping 
from sklearn import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Let's load the data.

# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")


# In[ ]:


sample_sub = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")


# In[ ]:


print("Train set shape = " +str(train.shape))
print("Test set shape = " +str(test.shape))
print("Dif set shape = " +str(Dig_MNIST.shape))


# In[ ]:


train.head()


# We slice the dataframes to define the features and the labels

# In[ ]:


X=train.iloc[:,1:].values 
Y=train.iloc[:,0].values 
Y[:10]


# Now we must reshape the date to make it Keras friendly.

# In[ ]:


X = X.reshape(X.shape[0], 28, 28,1) 
print(X.shape)


# Now we convert the labels to categorical.

# In[ ]:


Y = keras.utils.to_categorical(Y, 10) 
print(Y.shape)


# In[ ]:


test.head()


# In[ ]:


x_test=test.drop('id', axis=1).iloc[:,:].values
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
x_test.shape


# In[ ]:


Dig_MNIST.head()


# In[ ]:


x_dig=Dig_MNIST.drop('label',axis=1).iloc[:,:].values
print(x_dig.shape)
x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)
x_dig.shape


# In[ ]:


y_dig=Dig_MNIST.label
y_dig.shape


# We split the data into training and validation set.

# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.10, random_state=42) 


# We use Keras ImageDataGenerator to artificially increase our training set.

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 0.1,
                                   zoom_range = 0.25,
                                   horizontal_flip = False)


# In[ ]:


valid_datagen = ImageDataGenerator(rescale=1./255) 


# The next function reduces the learning rate as the training advances.

# In[ ]:


def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch


# In[ ]:


es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)


# In[ ]:


from sklearn import metrics


# Let's fit the model on the whole training set.

# In[ ]:


model = Sequential([
    Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),
    Conv2D(64,  (3,3), padding='same'),
    BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),

    layers.MaxPooling2D(2, 2),
    Dropout(0.2),
    
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),
    
    layers.MaxPooling2D(2,2),
    Dropout(0.2),    
    
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    LeakyReLU(alpha=0.1),

   layers.MaxPooling2D(2,2),
    Dropout(0.2),
    
    
    Flatten(),
    Dense(256),
    LeakyReLU(alpha=0.1),
 
    BatchNormalization(),
    Dense(10, activation='softmax')
])
model.summary()


# In[ ]:


initial_learningrate=2e-3
batch_size = 1024
epochs = 50
input_shape = (28, 28, 1)


# In[ ]:


model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(
      train_datagen.flow(X,Y, batch_size=batch_size),
      steps_per_epoch=100,
      epochs=epochs,
      callbacks=[LearningRateScheduler(lr_decay)           
               ],
      validation_data=valid_datagen.flow(X_valid,Y_valid),
      validation_steps=50,  
      verbose=2)


# In[ ]:


preds_dig=model.predict_classes(x_dig/255)
metrics.accuracy_score(preds_dig, y_dig)


# In[ ]:


predictions = model.predict_classes(x_test/255.)


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


submission['label'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

