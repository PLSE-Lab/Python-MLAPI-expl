#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,BatchNormalization,Conv2D,Dense,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Let's load the data.

# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
sample_sub = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")


# In[ ]:


print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Dif shape: {Dig_MNIST.shape}")


# In[ ]:


train.head()


# We slice the dataframes to define the features and the labels

# In[ ]:


X=train.iloc[:,1:].values
Y=train.iloc[:,0].values
Y[:10]


# Now we must reshape the date to make it TensorFlow friendly.

# In[ ]:


X = X.reshape(X.shape[0], 28, 28,1) 
print(f"X data shape: {X.shape}")


# Now we convert the labels to categorical.

# In[ ]:


Y = tf.keras.utils.to_categorical(Y, num_classes=10) 
print(f"Y data shape: {Y.shape}")


# In[ ]:


test.head()


# In[ ]:


x_test=test.drop('id', axis=1).iloc[:,:].values
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
print(f"x_test shape: {x_test.shape}")


# In[ ]:


Dig_MNIST.head()


# In[ ]:


x_dig=Dig_MNIST.drop('label',axis=1).iloc[:,:].values
x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)
print(f"x_dig shape: {x_dig.shape}")


# In[ ]:


y_dig=Dig_MNIST.label
print(f"y_dig shape: {y_dig.shape}")


# We split the data into training and validation set.

# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state=42) 


# Visualizing the digits data
# 
# Function to plot one random digit along with its label

# In[ ]:


def plot_random_digit():
    random_index = np.random.randint(0,X_train.shape[0])
    plt.imshow(X_train[random_index][:,:,0], cmap='gray')
    index = tf.argmax(Y_train[random_index], axis=0)
    plt.title(index.numpy())
    plt.axis("Off")


# In[ ]:


plt.figure(figsize=[2,2])
plot_random_digit()


# Execute the cell multiple times to see random examples.
# 
# Looking at 50 samples at one go

# In[ ]:


plt.figure(figsize=[10,6])
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(X_train[i][:,:,0], cmap='gray')
    plt.axis('Off')


# We use TensorFlow ImageDataGenerator to artificially increase our training set

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 0.1,
                                   zoom_range = 0.25,
                                   horizontal_flip=False)


# In[ ]:


valid_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


model = Sequential([
    Conv2D(64, 3, padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    Conv2D(64, 3, padding='same'),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(128, 3, padding='same'),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    Conv2D(128, 3, padding='same'),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    Conv2D(256, 3, padding='same'),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    Conv2D(256, 3, padding='same'),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    Flatten(),
    Dense(256),
    BatchNormalization(scale=False, center=True),
    Activation('relu'),
    Dense(10, activation='softmax')
])
model.summary()


# In[ ]:


tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


# Specify the training configuration (optimizer, loss, metrics)

# In[ ]:


learning_rate=0.001
batch_size = 1024
epochs = 50
steps_per_epoch = 100
validation_steps = 50


# In[ ]:


model.compile(optimizer=Adam(lr=learning_rate),
              loss=categorical_crossentropy,
              metrics=['accuracy'])


# The next function reduces the learning rate as the training advances.

# In[ ]:


def scheduler(epoch):
    return learning_rate * 0.99 ** epoch


# In[ ]:


lr_scheduler = LearningRateScheduler(scheduler)

model_checkpoint  = ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True,
                                    save_weights_only=True, monitor='val_loss', verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

callbacks_list = [lr_scheduler, model_checkpoint, early_stopping]


# In[ ]:


history = model.fit_generator(
      train_datagen.flow(X_train,Y_train,batch_size=batch_size),
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=valid_datagen.flow(X_valid,Y_valid),
      validation_steps=validation_steps,
      verbose=1,
      callbacks=callbacks_list)


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(accuracy))

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


model.load_weights('model_best_checkpoint.h5')


# In[ ]:


results = model.evaluate(X_valid, Y_valid, batch_size=batch_size, verbose=0)

print(f"\nLoss: {results[0]}")
print(f"Accuracy: {results[1]}")


# Assessing performance on the train set.
# The predictions of the model for the training data.

# In[ ]:


preds_dig = model.predict_classes(x_dig/255)


# Plotting the confusion matrix

# In[ ]:


cm = confusion_matrix(y_dig, preds_dig)

plt.figure(figsize=[7,6])
sns.heatmap(cm, cmap="Reds", annot=True, fmt='.0f')
plt.show()


# In[ ]:


print(f"Accuracy: {accuracy_score(y_dig, preds_dig)}")
print(classification_report(y_dig, preds_dig))


# In[ ]:


predictions = model.predict_classes(x_test/255)
predictions[:10]


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# In[ ]:


submission['label'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

