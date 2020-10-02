#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


# constants
IMG_SIZE = 28
N_CHANNELS = 1 # because gray scale images


# In[ ]:


train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
pred_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')


# In[ ]:


train_df = train_df.append(test_df)


# In[ ]:


MNIST like datatset for Kannada handwritten digits
print (f'Training set: {train_df.shape}')
print (f'To be Predicted: {pred_df.shape}')


# In[ ]:


X_train = train_df.drop(['label'], axis = 1)
Y_train = train_df['label']
X_pred = pred_df.drop(['id'], axis = 1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.05)


# In[ ]:


X_train, X_test, X_pred = X_train.apply(lambda x: x/255), X_test.apply(lambda x: x/255), X_pred.apply(lambda x: x/255)


# In[ ]:


Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)


# In[ ]:


X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)


# In[ ]:


Y_train = Y_train.to_numpy()
fig, ax = plt.subplots(nrows=3, ncols=4)
count=0
for row in ax:
    for col in row:
        col.set_title(np.argmax(Y_train[count, :]))
        col.imshow(X_train[count, :, :, 0])
        count += 1
plt.show()


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# This will just calculate parameters required to augment the given data. This won't perform any augmentations
datagen.fit(X_train)


# In[ ]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.40))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.40))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])


# In[ ]:


# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


batch_size=32
epochs = 45


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,Y_test),
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def PlotLoss(his, epoch):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epoch), his.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

def PlotAcc(his, epoch):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epoch), his.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.show()


# In[ ]:


PlotLoss(history, epochs)
PlotAcc(history, epochs)


# In[ ]:


preds = model.predict(X_pred.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS))


# In[ ]:


pred_df['label'] = np.argmax(preds, axis=1)
preds = pred_df[['id', 'label']]
preds.to_csv('sub.csv', index=False)


# In[ ]:




