#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.preprocessing import MinMaxScaler


# Read the data and preprocess.
# 
# 1. MinMax scaler from scikit-learn (scales each feature independently to the maximum and minimum for that feature).

# In[ ]:


Xtrain = np.load("../input/k49-train-imgs.npz")['arr_0']
ytrain = np.load("../input/k49-train-labels.npz")['arr_0']
Xtest = np.load("../input/k49-test-imgs.npz")['arr_0']
ytest = np.load("../input/k49-test-labels.npz")['arr_0']


train_one_hot_labels = keras.utils.to_categorical(ytrain, num_classes=49)
test_one_hot_labels = keras.utils.to_categorical(ytest, num_classes=49)
n_train = ytrain.shape[0]
n_test = ytest.shape[0]
npix = Xtrain.shape[1]

Xtrain1 = Xtrain.reshape(n_train, -1)
Xtest1 = Xtest.reshape(n_test, -1)
scaler = MinMaxScaler()
Xtrain1 = scaler.fit_transform(Xtrain1).astype('float32')
Xtest1 = scaler.fit_transform(Xtest1).astype('float32')


# In[ ]:


temp = Xtrain.reshape(n_train, -1)
np.sum(np.min(temp, axis=0)), np.sum(np.max(temp, axis=0)/255)


# The snippet above confirms the following
# 
# - The minimum of each feature is 0. (First element of tuple is sum of minima of each feature)
# - The maximum of each feature is 255. (Second element of tuple is sum of maxima/255 of each feature)
# 
# This means that it's okay to use the MinMax scaler as no feature has a smaller range in the dataset. If one of the features was smaller, it would get scaled incorrectly with respect to the other features.

# # Model 1
# 
# Feed-forward neural net. 1 hidden layer of 128 neurons with dropout rate = 0.5. Adagrad optimizer with categorical crossentropy loss.

# In[ ]:


model = Sequential([
    Dense(128, input_shape=(784,)),
    Activation('relu'),
    Dropout(rate=0.5),
    Dense(49),
    Activation('softmax'),
])

# For a multi-class classification problem
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history = model.fit(Xtrain1, train_one_hot_labels, epochs=100, batch_size=256, validation_data = (Xtest1, test_one_hot_labels))


# In[ ]:


def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


plot_history(history)


# With 128 hidden neurons, it's still pretty slow to converge. And training accuracy doesn't go up very high like earlier (with 32 neurons in hidden layer), so we are not overfitting.
# 
# So now consider 2 hidden layers with 64 neurons each. And keep dropout at 50%.

# In[ ]:


model2 = Sequential([
    Dense(64, input_shape=(784,)),
    Activation('relu'),
    Dropout(rate=0.2),
    Dense(64),
    Activation('relu'),
    Dropout(rate=0.2),
    Dense(49),
    Activation('softmax'),
])

# For a multi-class classification problem
model2.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history2 = model2.fit(Xtrain1, train_one_hot_labels, epochs=100, batch_size=256, validation_data = (Xtest1, test_one_hot_labels))


# In[ ]:


plot_history(history2)


# It looks like using feed-forward networks reaches a max accuracy hovering around 70%. I think the next step should be CNN.
# 
# # CNN
# Start with one convolutional layer followed by pooling. We can add one or two more such units (conv + pool) as necessary.

# In[ ]:


from keras import backend as K
K.image_data_format()


if K.image_data_format() == 'channels_last':
    Xtrain2d = Xtrain.reshape(n_train, npix, npix, 1).astype('float32')/255
    Xtest2d = Xtest.reshape(n_test, npix, npix, 1).astype('float32')/255
    input_shape = (npix, npix, 1)
else:
    print("Images not resized")


# In[ ]:


model3 = Sequential()
model3.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(49, activation='softmax'))
model3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


history3 = model3.fit(Xtrain2d, train_one_hot_labels, epochs=100, batch_size=128, validation_data = (Xtest2d, test_one_hot_labels))


# In[ ]:


plot_history(history3)


# Without much tweaking, we directly get over 91% accuracy on the validation data. Clearly convolution is a great apporach, but need to see how to get the high 90s.

# In[ ]:




