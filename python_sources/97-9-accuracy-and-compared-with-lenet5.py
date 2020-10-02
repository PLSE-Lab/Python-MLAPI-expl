#!/usr/bin/env python
# coding: utf-8

# ## The Happy House 
# 
# ### Problem Statement:
# Predicting your emotion happy or not.
# 
# 
# 
# In this,
# 1. We are going to build 2 types of Architecture of Neural Network among those one is Lenet-5 
# 2. Comparing Various Optimizers
# 3. Variously trained models(Different epochs and batch size)
# 
# Dataset is pictures from the front door camera to check if the person is happy or not.  The dataset is labbeled.
# 
# **Details of the "Happy" dataset**:
# - Images are of shape (64,64,3)
# - Training: 600 pictures
# - Test: 150 pictures
# 
# 
# It is now time to solve the "Happy" Challenge.

# ## 1 - Importing Libraries and Data 

# In[ ]:


import numpy as np
import h5py

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def load_dataset():
    train_data = h5py.File('../input/train_happy.h5', "r")
    X_train = np.array(train_data["train_set_x"][:]) 
    y_train = np.array(train_data["train_set_y"][:]) 

    test_data = h5py.File('../input/test_happy.h5', "r")
    X_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:]) 
    
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))
    
    return X_train, y_train, X_test, y_test


# In[ ]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# In[ ]:


# Sample image from dataset
print("Image shape :",X_train_orig[10].shape)
imshow(X_train_orig[10])


# ## 2 - Building a model 

# This architecture is given in Deep learning specialization|

# In[ ]:


# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7,7), strides=(1,1), name='Conv2D')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), name='max_pool')(X)
    
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    model = Model(inputs = X_input, outputs=X, name='HappyModel')
        
    return model


# In[ ]:


# Model flow chart
happyModel = HappyModel(X_train[0].shape)
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


# In[ ]:


happyModel.summary()


# ## 3 - Predicting using various Optimizers

# 

# ### 3.1 - SGD (Stochastic gradient descent optimizer)

# About SGD  <a href="https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1">here</a>

# In[ ]:


happyModel_sgd = HappyModel(X_train.shape[1:])
happyModel_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


history_sgd = happyModel_sgd.fit(X_train,Y_train, epochs=5,batch_size=30)


# In[ ]:


train_accuracy = history_sgd.history['acc']
train_loss = history_sgd.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = happyModel_sgd.evaluate(x=X_test, y=Y_test)

print ("\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ### 3.2 - RMSprop Optimizer

# About RMSprop Optimizer  <a href="https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b">here</a>

# In[ ]:


happyModel_rms = HappyModel(X_train.shape[1:])
happyModel_rms.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


history_rms = happyModel_rms.fit(X_train,Y_train, epochs=5,batch_size=30)


# In[ ]:


train_accuracy = history_rms.history['acc']
train_loss = history_rms.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = happyModel_rms.evaluate(x=X_test, y=Y_test)

print ("\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ### 3.3 - Adam

# Adam optimizer is a combination of RMSprop and Momentum optimizers. More about Adam Optimizer  <a href="https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/">here</a>

# In[ ]:


happyModel_adam = HappyModel(X_train[0].shape)
happyModel_adam.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


history_adam = happyModel_adam.fit(X_train,Y_train, epochs=5,batch_size=30)


# In[ ]:


train_accuracy = history_adam.history['acc']
train_loss = history_adam.history['loss']

count = range(len(train_accuracy))
plt.plot(count, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(count, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = happyModel_adam.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ### Conclusion
# <ul> <li> Adam Optimizer got high accuracy i.e., 94%</li> </ul>

# # 4 - Predicting with different epochs trained models

# ### 4.1 - Epoch : 20

# In[ ]:


happyModelE = HappyModel(X_train.shape[1:])
happyModelE.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


THistoryE = happyModelE.fit(X_train,Y_train, epochs=20,batch_size=30)


# In[ ]:


train_accuracy = THistoryE.history['acc']
train_loss = THistoryE.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = happyModelE.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:


y_pred = happyModelE.predict(X_test)


# In[ ]:


y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)


# ### Observation
# Got 94% accuracy!

# ### 4.2 - Epoch : 30

# In[ ]:


happyModel3 = HappyModel(X_train.shape[1:])
happyModel3.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


# THistory (Train History)
THistory3 = happyModel3.fit(X_train,Y_train, epochs=30,batch_size=30)


# In[ ]:


train_accuracy = THistory3.history['acc']
train_loss = THistory3.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = happyModel3.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ### Observation
# <ul> <li>Model accuracy increased updo 97.9%</li> </ul>

# ### 4.3 - Epoch 40, Batch size : 16 (Coursera suggested model parameters)

# In[ ]:


happyModel2 = HappyModel(X_train[0].shape)
happyModel2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


THistory2 = happyModel2.fit(X_train,Y_train, epochs=40,batch_size=16)


# In[ ]:


train_accuracy = THistory2.history['acc']
train_loss = THistory2.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds2 = happyModel2.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds2[0]))
print ("Test Accuracy = " + str(preds2[1]))


# ## 5 - Predicting using LeNet-5

# LeNet-5 is proposed by Yann LeCun(received Turing award). It is base ConvNet for modern neural network (CNN).
# More about LeNet5 <a href="https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4">here</a> and 
# <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">paper</a>

# In[ ]:


# Building LeNet-5 
def create_model():
    model = Sequential()
    model.add(layers.Conv2D(filters=1, kernel_size=(1,1), strides=(2,2), name='Conv2D', input_shape=(64,64,3))) # For converting image to 32,32,1
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=1, activation = 'sigmoid'))
    
    return model


# In[ ]:


model = create_model()
model.summary()


# In[ ]:


plot_model(model, to_file='HappyModel.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


lenet5 = create_model()
lenet5.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
history = lenet5.fit(X_train,Y_train, epochs=20,batch_size=32)

train_accuracy = history.history['acc']
train_loss = history.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:


preds = lenet5.evaluate(x=X_test, y=Y_test)

print ("\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ## 6 - Test with your own image

# Change path to predict your pic is happy or not

# path = 'https://www.cifar.ca/images/default-source/bios/lmb_yannlacun.png'
# 
# img = image.load_img(path, target_size=(64, 64))
# imshow(img)
# 
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = preprocess_input(img)
# 
# print(lenet2.predict(img))

# ## Conclusion

# <ul>
# <li>LeNet 5 : 94.6% accuracy</li>
# <li>Coursera model :  97.9% accuracy with batch size 30 and epoch 20 where coursera suggested model got 96% accuracy</li>
# <li>Compared SGD, RMSprop and Adam Optimizers. Among those Adam got hight accuracy</li>
# </ul>
# *Note :* Accuracy is not same everytime 
