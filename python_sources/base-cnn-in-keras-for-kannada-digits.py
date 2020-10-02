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


from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping

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


Dig_MNIST.head(20)


# In[ ]:


X_dig=Dig_MNIST.iloc[:,1:].values 
Y_dig=Dig_MNIST.iloc[:,0].values 
X_dig = X_dig.reshape(X_dig.shape[0], 28, 28,1)
Y_dig = keras.utils.to_categorical(Y_dig, 10) 
print(X_dig.shape,Y_dig.shape)


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
                                   horizontal_flip = False,
                                   vertical_flip = False)


# In[ ]:


valid_datagen = ImageDataGenerator(rescale=1./255.) 


# # trainning part

# In[ ]:


import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# ## Simple

# In[ ]:


a = Input(shape=(28,28,1,))
f = Flatten()(a)
b = Dense(128, activation="relu")(f)
b = Dense(10, activation="softmax")(b)
simple_model = Model(inputs=a, outputs=b)


# In[ ]:


simple_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size = 16
history = simple_model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size), validation_data=valid_datagen.flow(X_valid, Y_valid, batch_size=batch_size),  epochs=1)


# In[ ]:


evalu = simple_model.evaluate_generator(valid_datagen.flow(X_dig, Y_dig, batch_size=batch_size))
print("loss : " + str(evalu[0]))
print("accuracy : " + str(evalu[1]))


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()


# ## Convolution
# 

# In[ ]:


a = Input(shape=(28,28,1,))
c = Conv2D(128, (3,3), activation="relu")(a)
c = Conv2D(128, (3,3), activation="relu")(c)
c = Conv2D(128, (3,3), activation="relu")(c)
f = Flatten()(c)
b = Dense(128, activation="relu")(f)
b = Dense(10, activation="softmax")(b)
conv_model = Model(inputs=a, outputs=b)


# In[ ]:


conv_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size = 16
history = conv_model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size), validation_data=valid_datagen.flow(X_valid, Y_valid, batch_size=batch_size),  epochs=1)


# In[ ]:


evalu = conv_model.evaluate_generator(valid_datagen.flow(X_dig, Y_dig, batch_size=batch_size))
print("loss : " + str(evalu[0]))
print("accuracy : " + str(evalu[1]))


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()


# ## Hard model

# In[ ]:


a = Input(shape=(28,28,1,))
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(a)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(c)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(c)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
f = Flatten()(c)
b = Dropout(0.5)(f)
b = Dense(128, activation="relu", bias=False)(b)
b = BatchNormalization()(b)
b = Dropout(0.5)(b)
b = Dense(128, activation="relu", bias=False)(b)
b = BatchNormalization()(b)
b = Dense(10, activation="softmax")(b)
hard_model = Model(inputs=a, outputs=b)


# In[ ]:


hard_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size = 16
history = hard_model.fit_generator(train_datagen.flow(X, Y, batch_size=batch_size), validation_data=valid_datagen.flow(X_dig, Y_dig, batch_size=batch_size),  epochs=1)


# In[ ]:


evalu = hard_model.evaluate_generator(valid_datagen.flow(X_dig, Y_dig))
print("loss : " + str(evalu[0]))
print("accuracy : " + str(evalu[1]))


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()


# ## Hardmodel merge dig+train

# In[ ]:


X_merg = np.concatenate((X, X_dig))
Y_merg = np.concatenate((Y, Y_dig))


# In[ ]:


X_merg_train, X_merg_valid, Y_merg_train, Y_merg_valid = train_test_split(X_merg, Y_merg, test_size = 0.20, random_state=42) 


# In[ ]:


a = Input(shape=(28,28,1,))
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(a)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(c)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
c = Conv2D(128, (3,3), activation="relu", padding='same', bias=False)(c)
c = MaxPooling2D()(c)
c = BatchNormalization()(c)
f = Flatten()(c)
b = Dropout(0.5)(f)
b = Dense(128, activation="relu", bias=False)(b)
b = BatchNormalization()(b)
b = Dropout(0.5)(b)
b = Dense(128, activation="relu", bias=False)(b)
b = BatchNormalization()(b)
b = Dense(10, activation="softmax")(b)
hard_model = Model(inputs=a, outputs=b)


# In[ ]:


hard_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


batch_size = 16
history = hard_model.fit_generator(train_datagen.flow(X_merg_train, Y_merg_train, batch_size=batch_size), validation_data=valid_datagen.flow(X_merg_valid, Y_merg_valid, batch_size=batch_size),  epochs=12)


# In[ ]:


evalu = hard_model.evaluate_generator(valid_datagen.flow(X_dig, Y_dig))
print("loss : " + str(evalu[0]))
print("accuracy : " + str(evalu[1]))


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()


# # Submite

# In[ ]:


predictions = hard_model.predict(x_test/255.)


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


submission['label'] = np.argmax(predictions, axis = 1)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




