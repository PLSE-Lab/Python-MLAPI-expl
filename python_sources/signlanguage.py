#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
print(train.shape)
print(test.shape)
train.head()


# In[ ]:


from IPython.display import Image
Image("../input/sign-language-mnist/amer_sign3.png")


# In[ ]:


test.tail()


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1)


# In[ ]:


plt.figure(figsize = (15,7))
g = sns.countplot(Y_train, palette ="icefire")
plt.title("Number of sign classes")
Y_train.value_counts()


# In[ ]:


# Normalization

X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
test = ( test - np.max(test))/(np.max(test)-np.min(test))
print("X_train Shape : ", X_train.shape)
print("test shape : ", test.shape)


# In[ ]:


test = test.drop(["label"], axis = 1)


# In[ ]:


test.shape


# In[ ]:


X_train.head()


# In[ ]:


# Reshape

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("X_train shape : ", X_train.shape)
print("test shape : ",test.shape)


# In[ ]:


Y_train.value_counts()


# In[ ]:


Y_train = Y_train.values.reshape(-1,1)
Y_train.shape


# In[ ]:


# Label Encoding
from keras.utils.np_utils import to_categorical
Y_train= to_categorical(Y_train)


# In[ ]:


from numpy import argmax
inverted = argmax(Y_train[4])
print(inverted)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train, random_state = 42, test_size = 0.1)


print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)


# In[ ]:


# Create CNN 
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size=(3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(25, activation = "softmax"))

model.summary()


# #### OPTIONAL  - Inception 
# import keras 
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
# from keras.regularizers import l2
# from keras.optimizers import SGD, RMSprop
# from keras.utils import to_categorical
# from keras.layers.normalization import BatchNormalization
# from keras.utils.vis_utils import plot_model
# from keras.layers import Input, GlobalAveragePooling2D
# from keras import models
# from keras.models import Model
# 
# input_img = Input(shape=(28, 28, 1))
# 
# ### 1st layer
# layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
# layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)
# 
# layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
# layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)
# 
# layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
# layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)
# 
# mid_1 = tensorflow.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
# 
# flat_1 = Flatten()(mid_1)
# 
# dense_1 = Dense(1200, activation='relu')(flat_1)
# dense_2 = Dense(600, activation='relu')(dense_1)
# dense_3 = Dense(150, activation='relu')(dense_2)
# output = Dense(nClasses, activation='softmax')(dense_3)
# 
# model = Model([input_img], output)
# 
# 
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# 
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In[ ]:


# Optimizer

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

# Model Compile

model.compile(optimizer = optimizer, loss ="categorical_crossentropy", metrics =["accuracy"])

epochs = 50
batch_size = 255

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")A
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

