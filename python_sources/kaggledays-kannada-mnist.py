#!/usr/bin/env python
# coding: utf-8

# I forked notebook from [here](https://www.kaggle.com/bustam/cnn-in-keras-for-kannada-digits) [bustam](http://https://www.kaggle.com/bustam) and a small change in learning_rate_reduction gave me a little better score(0.99).
# 
# Thank you [bustam], you just made my first step of Kaggle and CNN journey more joyful. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt 


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


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(filters=64,  kernel_size=(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),    
    
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),
 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()


# In[ ]:



from keras.callbacks import ReduceLROnPlateau

initial_learningrate=2e-3
batch_size = 1024
epochs = 30
input_shape = (28, 28, 1)

model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

history = model.fit_generator(
      train_datagen.flow(X_train,Y_train, batch_size=batch_size),
      steps_per_epoch=100,
      epochs=epochs,
      callbacks=[learning_rate_reduction],
      validation_data=valid_datagen.flow(X_valid,Y_valid),
      validation_steps=50
)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn import metrics

preds_dig=model.predict_classes(x_dig/255)
metrics.accuracy_score(preds_dig, y_dig)


# In[ ]:


accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.figure()
plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()


# Let's fit the model on the whole training set.

# In[ ]:


predictions = model.predict_classes(x_test/255.)


# In[ ]:


predictions


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = predictions
submission.to_csv("submission.csv",index=False)


# In[ ]:




