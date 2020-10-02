# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# %% [code]
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
X_test=test
train_plot = sns.countplot(Y_train)
Y_train.value_counts()

print(X_train.shape)
print(X_test.shape)


# %% [code]
train.describe()

# %% [code]
#Plot a random image

img = X_train.iloc[10].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

# %% [code]
#Data preparation 

# Normalize the data
X_train = X_train / 255.0
x_test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

# %% [code]
# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test= x_test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",X_test.shape)

# %% [code]
# convert class labels (from digits) to one-hot encoded vectors
num_classes = 10
Y_train = keras.utils.to_categorical(Y_train, num_classes)
print(Y_train.shape)
print(Y_train)

# %% [code]
# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# %% [code]
print(X_train)


# %% [code]
print(X_val)

# %% [code]
print(Y_train)

# %% [code]
print(Y_val)

# %% [code]
X_train.dtype

# %% [code]
#Building the model

# %% [code]
#Allow you to create models layer-by-layer
model = Sequential()

# %% [code]
# specify input dimensions of each image
img_rows, img_cols = 28, 28
input_shape = (28, 28, 1)

# %% [code]
# model
model = Sequential()

# a keras convolutional layer is called Conv2D
# help(Conv2D)
# note that the first layer needs to be told the input shape explicitly

# first conv layer
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) # input shape = (img_rows, img_cols, 1)

# second conv layer
model.add(Conv2D(64, kernel_size=(3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten and put a fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu')) # fully connected
model.add(Dropout(0.5))

# softmax layer
model.add(Dense(num_classes, activation='softmax'))

# model summary
model.summary()


# %% [code]
#Fitting and Evaluating the Model
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
print(X_test.shape)
#Y_train = Y_train.reshape(-1, 1)
# usual cross entropy loss
# choose any optimiser such as adam, rmsprop etc
# metric is accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# %% [code]
# fit the model
# this should take around 10-15 minutes when run locally on a windows/mac PC 
batch_size=86
epochs=24
print(Y_train.shape)
print(X_train.shape)
X=model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_val, Y_val))

# %% [code]
#Accuracy 98.98%

# %% [code]
#Using Augmentation techniques to get better accuracy
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0.2,  # randomly rotate images in the range 2 degrees
        zoom_range = 0.2, # Randomly zoom image 2%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(X_train)

# %% [code]
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)

# %% [code]
#Using this model, we are able to achieve a score of 0.9738

# %% [code]
# Reshape
X_test = X_test.values.reshape(-1,28,28,1)
#X_test = X_test.as_matrix().reshape(28000, 784)

Prediction_final = pd.DataFrame(model.predict(X_test, batch_size=86))
Prediction_final = pd.DataFrame(Prediction_final.idxmax(axis = 1))
Prediction_final.index.name = 'ImageId'
Prediction_final = Prediction_final.rename(columns = {0: 'Label'}).reset_index()
Prediction_final['ImageId'] = Prediction_final['ImageId'] + 1
Prediction_final.head()

# %% [code]
Prediction_final.to_csv('mnist_submission.csv', index = False)