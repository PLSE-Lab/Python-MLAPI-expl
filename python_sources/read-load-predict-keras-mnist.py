#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Keras is a  python Library for deep learning that wraps powerful numerical computing libraries like Tensorflow and theano
# 
# In this post you will discover how to develop a CNN to recognize objects in handwritten MNIST dataset by using keras. My major focus in this kernal is on
# *  Classifying the Digists in MNIST dataset by using keras
# *  Save&Load Model to Json
# *  Predicting test data and uploading in csv file format

# # Data Preparation
# As a first step, I load all the modules that will be used in this notebook:

# In[ ]:


import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
k.set_image_dim_ordering('th')
from sklearn.utils import shuffle
import pandas as pd
import h5py

#As is good practice, we next initialize the random number seed with a constant 
#to ensure the results are reproducible
seed = 7
numpy.random.seed(seed)


# In[ ]:


df = pd.read_csv("../input/train.csv")
df = shuffle(df)

print('Dimensions of the dataframe', df.shape)
print(df[:2])


# The training datsset has 42000 training samples with 785 columns(1 Label column and 784 column belongs to pixel, hence all the images are scaled with 28x28 pixels(width and height)

# Let's divide our input dataframe df into X and Y datasets. X dataset contains only pixel information and Y dataset contains the labels of the images from 0 - 9.  

# In[ ]:


y = df['label']
X = df.drop(labels=['label'], axis=1).as_matrix()


# The pixel values are in the range of 0 to 255 for each of the red, green and blue channels.
# 
# It is good practice to work with normalized data. Because the input values are well understood, we can easily normalize to the range 0 to 1 by dividing each value by the maximum observation which is 255.

# In[ ]:


# reshape to be [samples][pixels][width][height]
print("X:shape", X.shape[0])
X = X.reshape(X.shape[0], 1, 28, 28).astype('float32')
print("X:reshape", X.shape[0])

#normalize inputs 0-255, 0 -1
X = X.astype('float32')
X = X / 255.0

y = np_utils.to_categorical(y)
print(y.shape)
print(y[:5])


# The output variables defines as a vector of inputs from 0 - 1 . 
# We can use a one hot encoding to transform them into a binary matrix in order to best model the classification problem
# one hot encoding. We know this data has 10 classes from 0 -9 and one hot encode looks like this
# 
# 0 - [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 
# 1 - [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# #######
# 
# 9 -[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# Let's start with CNN structure and see how well it will evaluate the model. 
# We will start a structure with 2 convolutional followed by maxpooling, flattening and fullyconnected
# Convolution input layer - It extract 32 feature with 5x5 filter size with relu and weight constraint of max set to 1
# Maxpool layer with 2X2
# Dropout with 20%
# fully connected layer
# 

# In[ ]:


from sklearn.metrics import accuracy_score
import pickle
batch_size = 200
n_classes = 10
epochs = 10

def cnn_keras(X, y):
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(1, 28, 28), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = model
    print(model.summary())
    return model


# In[ ]:


def train_nn(X, y):
    model = cnn_keras(X, y)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    predict = model.predict_classes(X, verbose=0)
    scores = model.evaluate(X, y, verbose=0)
    
    # serialize model to JSON
    mnist_cnn_json = model.to_json()
    with open("mnist_model.json", "w") as json_file:
        json_file.write(mnist_cnn_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print(predict[:5])
    return model


# In[ ]:


train_nn(X, y)


# A total 10 numbers of epochs are choosen and the classification accuracy and loss printed for both training and testing datasets. The model is evaluated on test and the accuracy is 99.12%, pretty good

# # Model Validation

# In[ ]:


import pandas as pd
df_vald = pd.read_csv("../input/test.csv").as_matrix()

print('Dimensions of the dataframe', df_vald.shape)
#print(df_vald[:1])


#reshape to be samples fixels width height
X_vald = df_vald.reshape(df_vald.shape[0], 1, 28, 28).astype('float32')

#normalize_inputs
X_vald = X_vald/255.0


 


# The validation dataset test.csv has 28000 samples with 784 columns and doesen't have labels. So the task here is to 
# 
# 1)  Output a single line containing the ImageID
# 
# 2)  Predicted Digit
# 
# ImageId,    Label
# 
#  1,              3
#  
#  2 ,             7
#  
#  3,            8

# In[ ]:



model = cnn_keras(X, y)

prediction = pd.DataFrame()
imageid = []
for i in range(len(X_vald)):
    i = i + 1
    imageid.append(i)
prediction["ImageId"] = imageid 
prediction["Label"] = model.predict_classes(X_vald, verbose=0)
print(prediction[:2])
prediction.to_csv("prediction.csv", index=False)


# 
# 
# 
