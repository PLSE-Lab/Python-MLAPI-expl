#!/usr/bin/env python
# coding: utf-8

# ### NN Digit recognizer

# In this notebook I updated the excellent work of [Poonam Ligade](http://www.kaggle.com/poonaml/deep-neural-network-keras-way) as a first view of neural nets in the Keras way version 2.1.0. I tried to simplify and avoid complexity to show just a few examples of Neural Network architectures on the MNIST dataset. 
# 
# For those who do not know it, as Tensorflow is a higher-level framework than python to build Neural Networks, Keras is a higher-level framework (API) than Tensorflow. Keras allow a higher level of abstraction to implement complex neural networks easy and quick.
# 
# I hope this will be usefull to anyone like me who wants to start using Neural Networks.
# 
# Any comment will be welcome!

# In[ ]:


import tensorflow as tf
tf.__version__


# ### Structure:
# 0. Initial configuration
# 1. Read data
# 2. Convert pandasDF to array
# 3. Example Visualization
# 4. Expand array dim for channels
# 5. Preproccessing images+
# 6. Simple NN - 1 layer
# 7. Complex NN - 5 layers
# 8. Simple Convolutional Neural Networks - 1 CNN layer
# 9. Complex Convolutional Neural Networks - 3 CNN layer
# 10. Data Augmentation + Complex Convolutional Neural Networks - X CNN layer

# ### 0. Initial Configuration

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # split train in train + validation
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, BatchNormalization, Convolution2D , MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# notebook parameters
train_path = '/kaggle/input/digit-recognizer/train.csv'
test_path = '/kaggle/input/digit-recognizer/test.csv'


# ### 1. Read datasets

# In[ ]:


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print('Dim of train: {}'.format(train.shape))
print('Dim of test: {}'.format(test.shape))


# ### 2. Convert pandasDF to arrays

# In[ ]:


X_train = train.iloc[:, 1:].values.astype('float32')  # all pixel values
Y_train = train.iloc[:, 0].values.astype('int32')  # only labels i.e targets digits
X_test = test.values.astype('float32')


# ### 3. Expand the array dim for channels

# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print('Dim of train: {}'.format(X_train.shape))
print('Dim od test: {}'.format(X_test.shape))


# ### 4. Visualization examples

# In[ ]:


j = 0
for i in range(6, 15):
    j += 1
    plt.subplot(330 + j)
    plt.imshow(X_train[i,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))
    plt.title(Y_train[i])


# ### 5. Preprocessing images

# In[ ]:


# ... feature standarization (later use)
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px


# In[ ]:


# ... one-hot-encoding of labels
Y_train = to_categorical(Y_train, num_classes=10)
num_classes = Y_train.shape[1]
print('Number of classes: {}'.format(num_classes))


# In[ ]:


# ... split the train data to train + validation (to monitor performance while training)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                  test_size=0.10, random_state=777)


# ### 6. Simple NN - 1 layer

# Lets first create a simple model from Keras Sequential layer:
# 
# 1. Lambda layer performs simple arithmetic operations like sum, average, exponentiation etc.
# 2. In 1st layer of the model we have to define input dimensions of our data in (rows,columns,colour channel) format. (In theano colour channel comes first)
# 3. Flatten will transform input into 1D array.
# 4. Dense is fully connected layer that means all neurons in previous layers will be connected to all neurons in fully connected layer. In the last layer we have to specify output dimensions/classes of the model. Here it's 10, since we have to output 10 different digit labels.

# In[ ]:


# ... define NN architecture
model = Sequential() # Initialize a Sequential object to define a NN layer by layer (sequentially)
model.add(Lambda(standardize, input_shape=(28,28,1))) # add a first layer which standardizes the input (grey-scale image of shape (28,28,1))
model.add(Flatten(data_format='channels_last')) # add a layer which transform the input of shape (28, 28, 1) to shape (n_x,)
model.add(Dense(10, activation='softmax')) # add a layer of 10 neurons that connect each one with all neurones in the previous layer
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)


# Before making network ready for training we have to make sure to add below things to compile forward and back-propagation:
# 
# 1. A loss function: to measure how good the network is at each step
# 2. An optimizer: to update network as it sees more data and reduce loss value
# 3. Metrics: to monitor performance of network over train and validation datasets

# In[ ]:


# ... define optimizer, loss function (and therefore a cost function) and metrics to monitor while training
model.compile(
    optimizer=RMSprop(lr=0.001, # learning rate (alpha)
                      rho=0.9, # momentum of order 2 (rho*dW + (1-rho)*dW**2)
                      momentum=0.0, # momentum of order 1 (rho*dW + (1-rho)*dW**2)
                      epsilon=1e-07, # term to avoid dividing by 0
                      centered=False, # if True standardize the gradients (high computational cost)
                      name='RMSprop'),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[ ]:


# ... fit NN (mini-batch approach of size 64 for 7 epochs with RMSprop optimizer)
history = model.fit(x=X_train,
                    y=Y_train,
                    batch_size=64,
                    epochs=20,
                    verbose=2,
                    validation_data=(X_val, Y_val))


# In[ ]:


# ... Visualize performance
history_dict = history.history
epochs = history.epoch
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(epochs, train_accuracy, c='b')
plt.plot(epochs, val_accuracy, c='r')
plt.title('learning curves')
plt.ylim(0.9,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()


# In[ ]:


# ... get real validation labels
real_val_class = []
for row in range(Y_val.shape[0]):
    prod = Y_val[row,:]*np.array(range(10))
    real_class = int(max(prod))
    real_val_class.append(int(real_class))


# In[ ]:


# ... validation error analysis
errors = pd.DataFrame({
    'real': real_val_class,
    'predict': model.predict_classes(X_val)
})

error_model_index = errors.loc[errors.real != errors.predict,:].index
error_model_index


# In[ ]:


i_check = error_model_index[0]
print('Real value: {}'.format(errors.loc[i_check,'real']))
print('Predict value: {}'.format(errors.loc[i_check,'predict']))

plt.figure(1)
plt.imshow(X_val[i_check,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))
plt.title(Y_val[i_check])


# In[ ]:


i_check = error_model_index[1]
print('Real value: {}'.format(errors.loc[i_check,'real']))
print('Predict value: {}'.format(errors.loc[i_check,'predict']))

plt.figure(1)
plt.imshow(X_val[i_check,:,:,:].reshape((28,28)), cmap=plt.get_cmap('gray'))
plt.title(Y_val[i_check])


# ### 7. Complex NN - 5 layers

# As we saw in the learning rates, there is not a big difference between Train performance and Validation performance (it means there is a little or no **variance error**). On the othe hand, the accuracy obtained in Validation is ~0.92 which is a bit far from the best possible score (~0.999 accuracy for other players) so there is a bit **bias error** that we can try to reduce:
# 
# * one way to achieve this is by increment the complexity of our model (try a bigger NN)
# * adding more data to train our model (look for more data or performe Data Augmentation)
# 
# In this section we will use a more complex architecture following the patters of classic neural networks: decrease the number of neurons per layer. We will perform Data Augmentation in the last section.

# In[ ]:


# ... define NN architecture
model_complex = Sequential() # Initialize a Sequential object to define a NN layer by layer (sequentially)
model_complex.add(Lambda(standardize,input_shape=(28,28,1))) # add a first layer which standardizes the input (grey-scale image of shape (28,28,1))
model_complex.add(Flatten(data_format='channels_last')) # add a layer which transform the input of shape (28, 28, 1) to shape (n_x,)
model_complex.add(Dense(512, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer
model_complex.add(Dense(256, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer
model_complex.add(Dense(128, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer
model_complex.add(Dense(64, activation='relu')) # add a layer of 512 neurons that connect each one with all neurones in the previous layer
model_complex.add(Dense(10, activation='softmax')) # add a layer of 10 neurons that connect each one with all neurones in the previous layer


# In[ ]:


model_complex.compile(optimizer='RMSprop', 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


# In[ ]:


history_complex = model_complex.fit(x=X_train,
                                    y=Y_train,
                                    batch_size=64,
                                    epochs=20,
                                    verbose=2,
                                    validation_data=(X_val, Y_val))


# In[ ]:


# ... Visualize performance
history_dict = history_complex.history
epochs = history_complex.epoch
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(epochs, train_accuracy, c='b')
plt.plot(epochs, val_accuracy, c='r')
plt.title('learning curves')
plt.ylim(0.9,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()


# There is an improvement in the performance.

# In[ ]:


# ... predictions
predictions = model_complex.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                          "Label": predictions})
submissions.to_csv("model_complex.csv", index=False, header=True)


# ### 8. Simple Convolutional Neural Networks - 1 CNN layer

# Now let's try a convolutional neural networks (CNN for short) which is Neural Network architectures which performe better on images by extracting usefull features in the first layers (convolutional layers).

# In[ ]:


# ... define architecture
cnn_model = Sequential()
cnn_model.add(Lambda(standardize, input_shape=(28,28,1)))
cnn_model.add(Convolution2D(64,(3,3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D())
cnn_model.add(Flatten())
cnn_model.add(Dense(124, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.9))
cnn_model.add(Dense(10, activation='softmax'))


# In[ ]:


# ... define optimizer, loss function and metrics to monitor
cnn_model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# In[ ]:


history_cnn = cnn_model.fit(x=X_train,
                            y=Y_train,
                            batch_size=256,
                            epochs=200,
                            verbose=2,
                            validation_data=(X_val, Y_val))


# In[ ]:


# ... Visualize performance
history_dict = history_cnn.history
epochs = history_cnn.epoch
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(epochs, train_accuracy, c='b')
plt.plot(epochs, val_accuracy, c='r')
plt.title('learning curves')
plt.ylim(0.9,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()


# In[ ]:


predictions = cnn_model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                          "Label": predictions})
submissions.to_csv("cnn_simple.csv", index=False, header=True)


# ### 9. Complex Convolutional Neural Networks - 3 CNN layers

# In[ ]:


# ... define architecture
cnn_adam_model = Sequential()
cnn_adam_model.add(Lambda(standardize, input_shape=(28,28,1)))
cnn_adam_model.add(Convolution2D(1024,(3,3), activation='relu'))
cnn_adam_model.add(BatchNormalization())
cnn_adam_model.add(MaxPooling2D())
cnn_adam_model.add(Convolution2D(512,(3,3), activation='relu'))
cnn_adam_model.add(BatchNormalization())
cnn_adam_model.add(MaxPooling2D())
cnn_adam_model.add(Convolution2D(256,(3,3), activation='relu'))
cnn_adam_model.add(BatchNormalization())
cnn_adam_model.add(MaxPooling2D())
cnn_adam_model.add(Flatten())
cnn_adam_model.add(Dense(512, activation='relu'))
cnn_adam_model.add(BatchNormalization())
cnn_adam_model.add(Dropout(0.9))
cnn_adam_model.add(Dense(124, activation='relu'))
cnn_adam_model.add(BatchNormalization())
cnn_adam_model.add(Dropout(0.9))
cnn_adam_model.add(Dense(10, activation='softmax'))


# In[ ]:


# ... define optimizer, loss function and metrics to monitor
cnn_adam_model.compile(optimizer=Adam(), 
                       loss='categorical_crossentropy',
                       metrics=['accuracy', 'mse'])


# In[ ]:


history_cnn_adam = cnn_adam_model.fit(x=X_train,
                                      y=Y_train,
                                      batch_size=124,
                                      epochs=50,
                                      verbose=2,
                                      validation_data=(X_val, Y_val))


# In[ ]:


# ... Visualize performance
history_dict = history_cnn_adam.history
epochs = history_cnn_adam.epoch
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.figure()
plt.plot(epochs, train_accuracy, c='b')
plt.plot(epochs, val_accuracy, c='r')
plt.title('learning curves')
plt.ylim(0.9,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()


# In[ ]:


predictions = cnn_adam_model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                          "Label": predictions})
submissions.to_csv("cnn_complex.csv", index=False, header=True)

