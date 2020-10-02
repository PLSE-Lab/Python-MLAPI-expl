#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Networks
# An Artificial neural network is a model of  computation inspired by the structure of neural networks in the brain.
# 
# [](https://miro.medium.com/max/1166/1*WNxN2ArLaGt0-Rm3tzWw1g.jpeg)
# 
# [](https://hackernoon.com/hn-images/1*zjzWdMucBfRbkqMzgm2xKg.png)

# ## Iris dataset

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
y = dataset.target.reshape(-1,1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y)


train_features, test_features, train_targets, test_targets = train_test_split(features,targets, test_size=0.2)

model = Sequential()
# first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#we can define the loss function MSE or negative log lokelihood
#optimizer will find the right adjustements for the weights: SGD, Adagrad, ADAM ...
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

model.fit(train_features, train_targets, epochs=10, batch_size=20, verbose=2)

loss, accuracy = model.evaluate(test_features, test_targets)

print("Accuracy on the test dataset: %.2f" % accuracy)


# ## Checkpoint 1
# Turing the parameters to get the accuracy more than 0.85

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
y = dataset.target.reshape(-1,1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y)


train_features, test_features, train_targets, test_targets = train_test_split(features,targets, test_size=0.2)

model = Sequential()
# first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#we can define the loss function MSE or negative log lokelihood
#optimizer will find the right adjustements for the weights: SGD, Adagrad, ADAM ...
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

model.fit(train_features, train_targets, epochs=30, batch_size=20, verbose=2)

loss, accuracy = model.evaluate(test_features, test_targets)

print("Accuracy on the test dataset: %.2f" % accuracy)


# ## Cifar10 dataset

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from scipy.misc import toimage


# In[ ]:


(x_train,y_train),(x_test,y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[ ]:


fig = plt.figure()
for i in range(0,25):
  ax = plt.subplot(5,5,i+1)
  plt.imshow(toimage(x_train[i]))
  code = y_train[i,0]
  print("{}-->{}".format(class_names[int(code)],code))


# In[ ]:


print("X_train: {},{}".format(x_train.shape,x_train.dtype))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(x_test.shape))


# In[ ]:


nTrain = x_train.shape[0]
nDimTrain = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
nTest = x_test.shape[0]
print("#Train: {},#Test:{}, nDim:{}".format(nTrain,nTest,nDimTrain))


# In[ ]:


x_train = x_train.reshape(nTrain,nDimTrain)
x_test = x_test.reshape(nTest,nDimTrain)
print("# reshape")
print("X_train: {}".format(x_train.shape))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))


# In[ ]:


y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print("X_train: {}".format(x_train.shape))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))


# In[ ]:


# Perceptron training
model =Sequential()
model.add(Dense(units=10,activation="softmax",input_shape=(nDimTrain,)))
# Compile the model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x_train / 255.0, y_train,
              batch_size=128,
              shuffle=True,
              epochs=epochs,
              validation_data=(x_test / 255.0, y_test),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, y_test)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


# In[ ]:


# Multi-layer Perceptron
model = Sequential()
model.add(Dense(units=50,activation="relu",input_shape=(nDimTrain,)))
model.add(Dense(units=10,activation="softmax"))
# Compile the model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x_train / 255.0, y_train,
              batch_size=128,
              shuffle=True,
              epochs=epochs,
              validation_data=(x_test / 255.0, y_test),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, y_test)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


# ## Checkpoint 2
# Try to change the number of hidden parameters to improve accuracy

# In[ ]:


# Multi-layer Perceptron
model = Sequential()
model.add(Dense(units=50,activation="relu",input_shape=(nDimTrain,)))
model.add(Dense(units=50,activation="relu",input_shape=(nDimTrain,)))
model.add(Dense(units=10,activation="softmax"))
# Compile the model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x_train / 255.0, y_train,
              batch_size=128,
              shuffle=True,
              epochs=epochs,
              validation_data=(x_test / 255.0, y_test),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, y_test)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


# ## Checkpoint 3
# Implement neural network model and predict the result using mnist dataset

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from scipy.misc import toimage
import numpy as np
import mnist

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

class_names = ['0','1','2','3','4','5','6','7','8','9']

fig = plt.figure()
for i in range(0,25):
  ax = plt.subplot(5,5,i+1)
  plt.imshow(toimage(x_train[i]))
  code = y_train[i]
  print("{}-->{}".format(code,code))
    
print("X_train: {},{}".format(x_train.shape,x_train.dtype))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(x_test.shape))


nTrain = x_train.shape[0]
nDimTrain = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
nTest = x_test.shape[0]
print("#Train: {},#Test:{}, nDim:{}".format(nTrain,nTest,nDimTrain))


x_train = x_train.reshape(nTrain,nDimTrain)
x_test = x_test.reshape(nTest,nDimTrain)
print("# reshape")
print("X_train: {}".format(x_train.shape))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))


y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print("X_train: {}".format(x_train.shape))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))



# Multi-layer Perceptron
model = Sequential()
model.add(Dense(units=50,activation="relu",input_shape=(nDimTrain,)))
model.add(Dense(units=10,activation="softmax"))
# Compile the model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x_train / 255.0, y_train,
              batch_size=128,
              shuffle=True,
              epochs=epochs,
              validation_data=(x_test / 255.0, y_test),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, y_test)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


# In[ ]:


import random
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

RANDOM_SEED = 1
SET_FIT_INTERCEPT = True

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_data = np.array(train)
test_data = np.array(test)

# drop the label (target) of train dataset
X_train_data = (train.values[:,1:]).astype(np.float32)

# get the label (target) of train dataset 
y_train_data = (train.values[:,0]).astype(np.int32)

# test dataset does not contain a label
X_test_data = (test.values).astype(np.float32)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size = .2, random_state=RANDOM_SEED)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape)

# examine the shape of the loaded data
print('train_data shape:', X_train.shape)
print('test_data shape:', X_test.shape, '\n')


X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)
test_data = test_data.reshape(-1,28,28,1)
print('response count\n', train.label.value_counts())

train_plot = train.drop('label',axis=1)
plt.figure(figsize=(14,12))
for digit_num in range(0,70):
    plt.subplot(7,10,digit_num+1)
    grid_data = train_plot.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

optimizers = ['Adam']
num_nodes = [256]


from sklearn.model_selection import ParameterGrid
parameters = {
'optimizer': optimizers,
'num_nodes' : num_nodes
}
parameterList = list(ParameterGrid(parameters))
num_nodes_list = []
optimizer_list = []
train_acc_list = []
test_acc_list = []
proc_list = []

#####################################################################################
optim = parameterList[0]['optimizer']
numNodes = parameterList[0]['num_nodes']
print("Optimizer: ", optim)
print("Number of Nodes: ", numNodes)

model1 = Sequential([
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',input_shape=(28,28,1)),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2, padding='same'),
    Flatten(),
    Dense(numNodes, activation='relu'),
    Dense(10, activation='softmax'),
    ])

model1.compile(
    optimizer=optim,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )

start = datetime.now()
model1.fit(
    X_train, # training data
    y_train, # training targets
    epochs=10,
    )
end = datetime.now()
processing_time = end-start

score_train,acc_train = model1.evaluate(X_train,y_train)
score_test,acc_test = model1.evaluate(X_test,y_test)

print("Training Accuracy: ", acc_train)
print("Test Accuracy: ", acc_test)
print("Processing Time: ", processing_time)
num_nodes_list.append(numNodes)
optimizer_list.append(optim)
train_acc_list.append(acc_train)
test_acc_list.append(acc_test)
proc_list.append(processing_time)

#####################################################################################



#####################################################################################
performance_df = pd.DataFrame(columns = ["Combination", 
                                         "Optimizer",
                                         "Number of Nodes",
                                         "Processing Time", 
                                         "Train Accuracy", 
                                         "Test Accuracy"])
performance_df['Combination'] = ParameterGrid(parameters)
performance_df['Optimizer'] = optimizer_list
performance_df['Number of Nodes'] = num_nodes_list
performance_df['Processing Time'] = proc_list
performance_df['Train Accuracy'] = train_acc_list
performance_df['Test Accuracy'] =  test_acc_list
performance_df = performance_df.sort_values(by='Test Accuracy',ascending=False)
performance_df

results = model1.predict(test_data)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
results.head(5)

np.savetxt('results.csv',
           np.c_[range(1,len(test)+1),results],
           delimiter=',',
           header='ImageId,Label',
           comments= '',
           fmt = '%d'
          )

