# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Keras imports
from keras.models import  Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils

# InceptionV3 model imports
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

tr_path = "../input/train.csv"
tst_path = "../input/test.csv"

# Load the training file. It is a matrix with 785 cols. 0 th column is the label & 1-785 are data
data = pd.read_csv(tr_path)
#np.random.shuffle(data)
X = data.ix[ :, 1:].values.astype('float32')
y = data.ix[ :, 0].values.astype('int32')

# Convert y to one-hot
y = np_utils.to_categorical(y)

# Pre-process input for zero mean and variance of 1
X = np.multiply(X , 1.0/255.0)

val_size = round(X.shape[0] * 0.2) # 20%

# Reshape the data 
input_imgs = X.reshape(X.shape[0], 28 , 28, 1)
print(input_imgs.shape)
X = None
data = None

X_val = input_imgs[0:val_size, :]
X_train = input_imgs[val_size: , :]

y_val = y[0:val_size, :]
y_train = y[val_size:, :]

print(X_train.shape)

# Hyper parameters
epochs = 1
batch_size = 50

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
#base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(128, activation='relu')(x)
# and a logistic layer -- let's say we have 10 classes
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
m = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

'''
# Create the model
m = Sequential()

# First conv layer [ size, 28 x 28 x 16]
m.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='valid'))
m.add(Activation('relu'))
m.add(MaxPooling2D(strides=(2, 2)))  #[ size, 14 x 14 x 16]


# Second conv layer [ size, 14 x 14 x 32]
m.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='valid'))
m.add(Activation('relu'))
m.add(MaxPooling2D(strides=(2, 2))) #[ size, 7 x 7 x 32]

m.add(Flatten())
m.add(Dropout(0.15))
# Fully connected layer and a softmax layer at the end for classification
m.add(Dense(128, activation="relu"))
m.add(Dense(10, activation="softmax")) # Final output classes (10)
'''
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in m.layers[:249]:
   layer.trainable = False
for layer in m.layers[249:]:
   layer.trainable = True
   
m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(m.summary())

model_info = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)

# Final evaluation of the model
scores = m.evaluate(X_val, y_val, verbose=2)
print("Error: %.2f%%" % ((1-scores[1])*100))


# Create the test set and predict results
test = pd.read_csv(tst_path).values.astype('float32')
# Pre-process test for zero mean and variance of 1
testX = np.multiply(test , 1.0/255.0)

# Reshape the data 
testX = testX.reshape(testX.shape[0], 28 , 28, 1)

test = None
targets = m.predict(testX, batch_size=32, verbose=0)

# Write the output file
ImageId = np.arange(testX.shape[0])+1

raw_data = {'ImageId': ImageId,
        'Label': np.argmax(targets, axis=1)}
df = pd.DataFrame(raw_data, columns = ['ImageId', 'Label'])
df.to_csv(path_or_buf = 'output.csv', index=None, header=True)
