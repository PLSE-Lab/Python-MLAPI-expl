#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# We have learned about the neural network basics and experimented with different optimization functions, batch normalization, dropouts etc in the last note book. If you need a quick revision please take a look in https://www.kaggle.com/vishnurapps/beginners-guide-to-use-keras
# 
# Now we are going to check the importance of the depth of neural network and its influence in the output prediction. We will test with different depths, batch normalization and dropouts

# ## Importing Libraries

# In[ ]:


from keras.utils import np_utils 
from keras.datasets import mnist 
import seaborn as sns
from keras.initializers import RandomNormal
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
import time


# In[ ]:


# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4
# https://stackoverflow.com/a/14434334
# this function is used to update the plots for each epoch and error
def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[ ]:


# the data, shuffled and split between train and test sets 
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


type(X_train)


# - The X_tran and X_test are three dimentional numpy array. The 0th dimention is the number of images, 1st dimension is the width and 2nd dimension is the height

# In[ ]:


plt.imshow(X_train[218])


# - Using imshow to see one image in the dataset. The number in the 218th position is 6

# In[ ]:


print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d, %d)"%(X_train.shape[1], X_train.shape[2]))
print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d, %d)"%(X_test.shape[1], X_test.shape[2]))


# In[ ]:


# if you observe the input shape its 2 dimensional vector
# for each image we have a (28*28) vector
# we will convert the (28*28) vector into single dimensional vector of 1 * 784 

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) 


# In[ ]:


# after converting the input images from 3d to 2d vectors

print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d)"%(X_train.shape[1]))
print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d)"%(X_test.shape[1]))


# In[ ]:


# An example data point
print(X_train[0])


# In[ ]:


# if we observe the above matrix each cell is having a value between 0-255
# before we move to apply machine learning algorithms lets try to normalize the data
# X => (X - Xmin)/(Xmax-Xmin) = X/255

X_train = X_train/255
X_test = X_test/255


# In[ ]:


# example data point after normlizing
print(X_train[0])


# In[ ]:


# here we are having a class number for each image
print("Class label of first image :", y_train[0])

# lets convert this into a 10 dimensional vector
# ex: consider an image is 5 convert it into 5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# this conversion needed for MLPs 

Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)

print("After converting the output into a vector : ",Y_train[0])


# ## Experiment 1 (2 hidden layers 1024 512 neurons)

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense, Activation 


# In[ ]:


# some model parameters

output_dim = 10
input_dim = X_train.shape[1]

batch_size = 128 
nb_epoch = 20


# In[ ]:


model_2_relu = Sequential()
model_2_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_2_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_2_relu.add(Dense(output_dim, activation='softmax'))

print(model_2_relu.summary())


# In[ ]:


model_2_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_2_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_2_relu_score = model_2_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_2_relu_score[0]) 
print('Test accuracy:', model_2_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


model_2_relu.layers


# In[ ]:


w_after = model_2_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First two layers have weights in the form of normal distribution

# ## Experiment 2 (2 hidden layers 256 128 neurons)

# In[ ]:


model_2_relu_medium = Sequential()
model_2_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_2_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_2_relu_medium.add(Dense(output_dim, activation='softmax'))

print(model_2_relu_medium.summary())


# In[ ]:


model_2_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_2_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_2_relu_medium_score = model_2_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_2_relu_medium_score[0]) 
print('Test accuracy:', model_2_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_2_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First two layers have weights in the form of normal distribution
# - There is an increase in accuracy with reduction in number of neurons

# ## Experiment 3 (2 hidden layers 128 64 neurons)

# In[ ]:


model_2_relu_small = Sequential()
model_2_relu_small.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_2_relu_small.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_2_relu_small.add(Dense(output_dim, activation='softmax'))

print(model_2_relu_small.summary())


# In[ ]:


model_2_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_2_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_2_relu_small_score = model_2_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_2_relu_small_score[0]) 
print('Test accuracy:', model_2_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_2_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating. But the fluctuation has reduced with reduction in number of neurons
# - First two layers have weights in the form of normal distribution
# - There is an increase in accuracy with reduction in number of neurons

# ## Experiment 4 (2 hidden layers 1024  512 neurons + Batch Normalization)

# In[ ]:


from keras.layers.normalization import BatchNormalization

model_batch_relu = Sequential()

model_batch_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_batch_relu.add(BatchNormalization())

model_batch_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_batch_relu.add(BatchNormalization())

model_batch_relu.add(Dense(output_dim, activation='softmax'))


model_batch_relu.summary()


# In[ ]:


model_batch_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_batch_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_batch_relu_score = model_batch_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_batch_relu_score[0]) 
print('Test accuracy:', model_batch_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_batch_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is fluctuating. There is a significant gap between train and test loss
# - First two layers have weights in the form of normal distribution
# - Most of the weights of the output layer are positive

# ## Experiment 6 (2 hidden layers 256 128 neurons + Batch Normalization)

# In[ ]:


model_batch_relu_medium = Sequential()

model_batch_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_batch_relu_medium.add(BatchNormalization())

model_batch_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_batch_relu_medium.add(BatchNormalization())

model_batch_relu_medium.add(Dense(output_dim, activation='softmax'))


model_batch_relu_medium.summary()


# In[ ]:


model_batch_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_batch_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_batch_relu_medium_score = model_batch_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_batch_relu_medium_score[0]) 
print('Test accuracy:', model_batch_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_batch_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is fluctuating. There is a significant gap between train and test loss and gap is increasing
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive

# ## Experiment 6 (2 hidden layers 128 64 neurons + Batch Normalization))

# In[ ]:


model_batch_relu_small = Sequential()

model_batch_relu_small.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_batch_relu_small.add(BatchNormalization())

model_batch_relu_small.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_batch_relu_small.add(BatchNormalization())

model_batch_relu_small.add(Dense(output_dim, activation='softmax'))


model_batch_relu_small.summary()


# In[ ]:


model_batch_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_batch_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_batch_relu_small_score = model_batch_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_batch_relu_small_score[0]) 
print('Test accuracy:', model_batch_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_batch_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is fluctuating. There is a significant gap between train and test loss.
# - The fluctuation have reduced compared to models with more neurons
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive

# ## Experiment 7 (2 hidden layers 1024 512 neurons + Batch Normalization + Dropout))

# In[ ]:


# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

from keras.layers import Dropout

model_drop = Sequential()

model_drop.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))

model_drop.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))

model_drop.add(Dense(output_dim, activation='softmax'))


model_drop.summary()


# In[ ]:


model_drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_drop_score = model_drop.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_drop_score[0]) 
print('Test accuracy:', model_drop_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_drop.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing and become equal at the end of 20 epochs.
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss

# ## Experiment 8 (2 hidden layers 254 128 neurons + Batch Normalization + Dropout))

# In[ ]:


# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

from keras.layers import Dropout

model_drop_medium = Sequential()

model_drop_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_drop_medium.add(BatchNormalization())
model_drop_medium.add(Dropout(0.5))

model_drop_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_drop_medium.add(BatchNormalization())
model_drop_medium.add(Dropout(0.5))

model_drop_medium.add(Dense(output_dim, activation='softmax'))


model_drop_medium.summary()


# In[ ]:


model_drop_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_drop_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_drop_medium_score = model_drop_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_drop_medium_score[0]) 
print('Test accuracy:', model_drop_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_drop_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss

# ## Experiment 9 (2 hidden layers 128 64 neurons + Batch Normalization + Dropout))

# In[ ]:


# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

from keras.layers import Dropout

model_drop_small = Sequential()

model_drop_small.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_drop_small.add(BatchNormalization())
model_drop_small.add(Dropout(0.5))

model_drop_small.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_drop_small.add(BatchNormalization())
model_drop_small.add(Dropout(0.5))

model_drop_small.add(Dense(output_dim, activation='softmax'))


model_drop_small.summary()


# In[ ]:


model_drop_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_drop_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_drop_small_score = model_drop_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_drop_small_score[0]) 
print('Test accuracy:', model_drop_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_drop_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
out_w = w_after[4].flatten().reshape(-1,1)


fig = plt.figure(figsize=(10,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 3, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 3, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 3, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss
# - Accuracy was more for nn with more number of neurons in each layer

# ## Experiment 10 (3 hidden layers 1024 768 512 neurons)

# In[ ]:


model_3_relu = Sequential()
model_3_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_3_relu.add(Dense(output_dim, activation='softmax'))

print(model_3_relu.summary())


# In[ ]:


model_3_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_relu_score = model_3_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_relu_score[0]) 
print('Test accuracy:', model_3_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


model_3_relu.layers


# In[ ]:


w_after = model_3_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First three layers have weights in the form of normal distribution
# - Accuracy has increased with increase in number of hidden layers. Better than two layered nn.

# ## Experiment 11 (3 hidden layers 256 192 128 neurons)

# In[ ]:


model_3_relu_medium = Sequential()
model_3_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_3_relu_medium.add(Dense(output_dim, activation='softmax'))

print(model_3_relu_medium.summary())


# In[ ]:


model_3_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_relu_medium_score = model_3_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_relu_medium_score[0]) 
print('Test accuracy:', model_3_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First two layers have weights in the form of normal distribution
# - Accuracy has increased with increase in number of hidden layers. Better than two layered nn.

# ## Experiment 12 (3 hidden layers 128 96 64 neurons)

# In[ ]:


model_3_relu_small = Sequential()
model_3_relu_small.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu_small.add(Dense(96, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_3_relu_small.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_3_relu_small.add(Dense(output_dim, activation='softmax'))

print(model_3_relu_small.summary())


# In[ ]:


model_3_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_relu_small_score = model_3_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_relu_small_score[0]) 
print('Test accuracy:', model_3_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - Frequency of fluctuation has reduced
# - First two layers have weights in the form of normal distribution
# - Accuracy has increased with increase in number of hidden layers. Better than two layered nn.

# ## Experiment 13 (3 hidden layers 1024 768 512 neurons + batch normalization)

# In[ ]:


model_3_batch_relu = Sequential()

model_3_batch_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu.add(BatchNormalization())

model_3_batch_relu.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu.add(BatchNormalization())

model_3_batch_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_batch_relu.add(BatchNormalization())

model_3_batch_relu.add(Dense(output_dim, activation='softmax'))


model_3_batch_relu.summary()


# In[ ]:


model_3_batch_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_batch_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_batch_relu_score = model_3_batch_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_batch_relu_score[0]) 
print('Test accuracy:', model_3_batch_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_batch_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First and last layers have weights in the form of normal distribution

# ## Experiment 14 (3 hidden layers 256 192 128 neurons + batch normalization)

# In[ ]:


model_3_batch_relu_medium = Sequential()

model_3_batch_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu_medium.add(BatchNormalization())

model_3_batch_relu_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu_medium.add(BatchNormalization())

model_3_batch_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_batch_relu_medium.add(BatchNormalization())

model_3_batch_relu_medium.add(Dense(output_dim, activation='softmax'))


model_3_batch_relu_medium.summary()


# In[ ]:


model_3_batch_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_batch_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_batch_relu_medium_score = model_3_batch_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_batch_relu_medium_score[0]) 
print('Test accuracy:', model_3_batch_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_batch_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - Frequency of fluctuation has reduced
# - First layer have weights in the form of normal distribution

# ## Experiment 15 (3 hidden layers 128 96 64 neurons + batch normalization)

# In[ ]:


model_3_batch_relu_small = Sequential()

model_3_batch_relu_small.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu_small.add(BatchNormalization())

model_3_batch_relu_small.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_batch_relu_small.add(BatchNormalization())

model_3_batch_relu_small.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_batch_relu_small.add(BatchNormalization())

model_3_batch_relu_small.add(Dense(output_dim, activation='softmax'))


model_3_batch_relu_small.summary()


# In[ ]:


model_3_batch_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_batch_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_batch_relu_small_score = model_3_batch_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_batch_relu_small_score[0]) 
print('Test accuracy:', model_3_batch_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_batch_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - Frequency of fluctuation has reduced compared to prior model
# - First and last layers have weights in the form of normal distribution

# ## Experiment 16 (3 hidden layers 1024 768 512 neurons + batch normalization + dropouts)

# In[ ]:


from keras.layers import Dropout
model_3_drop = Sequential()

model_3_drop.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop.add(BatchNormalization())
model_3_drop.add(Dropout(0.5))

model_3_drop.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop.add(BatchNormalization())
model_3_drop.add(Dropout(0.5))

model_3_drop.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_drop.add(BatchNormalization())
model_3_drop.add(Dropout(0.5))

model_3_drop.add(Dense(output_dim, activation='softmax'))


model_3_drop.summary()


# In[ ]:


model_3_drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_drop_score = model_3_drop.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_drop_score[0]) 
print('Test accuracy:', model_3_drop_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_drop.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - First layer have weights in the form of normal distribution
# - Accuracy is more compared to two hidden layer NN.

# ## Experiment 17 (3 hidden layers 256 192 128 neurons + batch normalization + dropouts)

# In[ ]:


model_3_drop_medium = Sequential()

model_3_drop_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop_medium.add(BatchNormalization())
model_3_drop_medium.add(Dropout(0.5))

model_3_drop_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop_medium.add(BatchNormalization())
model_3_drop_medium.add(Dropout(0.5))

model_3_drop_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_drop_medium.add(BatchNormalization())
model_3_drop_medium.add(Dropout(0.5))

model_3_drop_medium.add(Dense(output_dim, activation='softmax'))


model_3_drop_medium.summary()


# In[ ]:


model_3_drop_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_drop_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_drop_medium_score = model_3_drop_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_drop_medium_score[0]) 
print('Test accuracy:', model_3_drop_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_drop_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - First layer have weights in the form of normal distribution
# - Accuracy is reduced from prior model. This means dropout works well if the number of neurons in each hidden layer is more.

# ## Experiment 18 (3 hidden layers 128 96 64 neurons + batch normalization + dropouts)

# In[ ]:


model_3_drop_small = Sequential()

model_3_drop_small.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop_small.add(BatchNormalization())
model_3_drop_small.add(Dropout(0.5))

model_3_drop_small.add(Dense(96, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_3_drop_small.add(BatchNormalization())
model_3_drop_small.add(Dropout(0.5))

model_3_drop_small.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_3_drop_small.add(BatchNormalization())
model_3_drop_small.add(Dropout(0.5))

model_3_drop_small.add(Dense(output_dim, activation='softmax'))


model_3_drop_small.summary()


# In[ ]:


model_3_drop_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_3_drop_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_3_drop_small_score = model_3_drop_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_3_drop_small_score[0]) 
print('Test accuracy:', model_3_drop_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_3_drop_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 4, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - Test loss is always less than train loss
# - First layer have weights in the form of normal distribution
# - Accuracy is becoming lower compared to two prior models. Means dropout works well if the number of neurons in hidden layer is more

# ## Experiment 19 (5 hidden layers 1024 896 768 640 512 neurons)

# In[ ]:


model_5_relu = Sequential()
model_5_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu.add(Dense(896, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu.add(Dense(640, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_5_relu.add(Dense(output_dim, activation='softmax'))

print(model_5_relu.summary())


# In[ ]:


model_5_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_relu_score = model_5_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_relu_score[0]) 
print('Test accuracy:', model_5_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First five layers have weights in the form of normal distribution
# - There is a slight decrease in the accuracy compared to three layer NN

# ## Experiment 20 (3 hidden layers 256 224 192 160 128 neurons)

# In[ ]:


model_5_relu_medium = Sequential()
model_5_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_medium.add(Dense(224, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_medium.add(Dense(160, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_5_relu_medium.add(Dense(output_dim, activation='softmax'))

print(model_5_relu_medium.summary())


# In[ ]:


model_5_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_relu_medium_score = model_5_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_relu_medium_score[0]) 
print('Test accuracy:', model_5_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - First two layers have weights in the form of normal distribution
# - There is a slight decrease in the accuracy compared to three layer NN

# ## Experiment 21 (5 hidden layers 128 112 96 80 64 neurons)

# In[ ]:


model_5_relu_small = Sequential()
model_5_relu_small.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_small.add(Dense(112, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_small.add(Dense(96, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_small.add(Dense(80, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))
model_5_relu_small.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )
model_5_relu_small.add(Dense(output_dim, activation='softmax'))

print(model_5_relu_small.summary())


# In[ ]:


model_5_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_relu_small_score = model_5_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_relu_small_score[0]) 
print('Test accuracy:', model_5_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but test loss is increasing and fluctuating
# - Frequency of fluctuation decresed
# - First layers have weights in the form of normal distribution
# - There is a slight decrease in the accuracy compared to three layer NN

# ## Experiment 22 (5 hidden layers 1024 896 768 640 512 neurons + batch normalization)

# In[ ]:


model_5_batch_relu = Sequential()

model_5_batch_relu.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu.add(BatchNormalization())

model_5_batch_relu.add(Dense(896, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu.add(BatchNormalization())

model_5_batch_relu.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu.add(BatchNormalization())

model_5_batch_relu.add(Dense(640, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu.add(BatchNormalization())

model_5_batch_relu.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_batch_relu.add(BatchNormalization())

model_5_batch_relu.add(Dense(output_dim, activation='softmax'))


model_5_batch_relu.summary()


# In[ ]:


model_5_batch_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_batch_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_batch_relu_score = model_5_batch_relu.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_batch_relu_score[0]) 
print('Test accuracy:', model_5_batch_relu_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_batch_relu.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but train loss is fluctuating. There is a significant gap between train and test loss
# - There is heavy fluctuation in the test loss
# - First layers have weights in the form of normal distribution
# - There is no significant inprovement in accuracy compared to other batch normalization models

# ## Experiment 23 (5 hidden layers 256 224 192 160 128 neurons + batch normalization)

# In[ ]:


model_5_batch_relu_medium = Sequential()

model_5_batch_relu_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_medium.add(BatchNormalization())

model_5_batch_relu_medium.add(Dense(224, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_medium.add(BatchNormalization())

model_5_batch_relu_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_medium.add(BatchNormalization())

model_5_batch_relu_medium.add(Dense(160, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_medium.add(BatchNormalization())

model_5_batch_relu_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_batch_relu_medium.add(BatchNormalization())

model_5_batch_relu_medium.add(Dense(output_dim, activation='softmax'))


model_5_batch_relu_medium.summary()


# In[ ]:


model_5_batch_relu_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_batch_relu_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_batch_relu_medium_score = model_5_batch_relu_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_batch_relu_medium_score[0]) 
print('Test accuracy:', model_5_batch_relu_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_batch_relu_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but train loss is fluctuating. There is a significant gap between train and test loss
# - There is heavy fluctuation in the test loss but compared to prior model, its less. 
# - First layers have weights in the form of normal distribution
# - There is no significant inprovement in accuracy compared to other batch normalization models

# ## Experiment 24 (3 hidden layers 128 112 96 80 64 neurons + batch normalization)

# In[ ]:


model_5_batch_relu_small = Sequential()

model_5_batch_relu_small.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_small.add(BatchNormalization())

model_5_batch_relu_small.add(Dense(112, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_small.add(BatchNormalization())

model_5_batch_relu_small.add(Dense(96, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_small.add(BatchNormalization())

model_5_batch_relu_small.add(Dense(80, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_batch_relu_small.add(BatchNormalization())

model_5_batch_relu_small.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_batch_relu_small.add(BatchNormalization())

model_5_batch_relu_small.add(Dense(output_dim, activation='softmax'))


model_5_batch_relu_small.summary()


# In[ ]:


model_5_batch_relu_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_batch_relu_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_batch_relu_small_score = model_5_batch_relu_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_batch_relu_small_score[0]) 
print('Test accuracy:', model_5_batch_relu_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_batch_relu_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Train loss is decreasing continously but train loss is fluctuating. There is a significant gap between train and test loss
# - There is heavy fluctuation in the test loss but compared to prior model, its less. 
# - Test loss seems to increase
# - First layers have weights in the form of normal distribution
# - There is no significant inprovement in accuracy compared to other batch normalization models

# ## Experiment 25 (3 hidden layers 1024 896 768 640 512 neurons + batch normalization + dropouts)

# In[ ]:


from keras.layers import Dropout
model_5_drop = Sequential()

model_5_drop.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop.add(BatchNormalization())
model_5_drop.add(Dropout(0.5))

model_5_drop.add(Dense(896, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop.add(BatchNormalization())
model_5_drop.add(Dropout(0.5))

model_5_drop.add(Dense(768, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop.add(BatchNormalization())
model_5_drop.add(Dropout(0.5))

model_5_drop.add(Dense(640, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop.add(BatchNormalization())
model_5_drop.add(Dropout(0.5))

model_5_drop.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_drop.add(BatchNormalization())
model_5_drop.add(Dropout(0.5))

model_5_drop.add(Dense(output_dim, activation='softmax'))


model_5_drop.summary()


# In[ ]:


model_5_drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_drop_score = model_5_drop.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_drop_score[0]) 
print('Test accuracy:', model_5_drop_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_drop.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing and become equal at the end of 20 epochs.
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss
# - Accuracy compared to prior dropout models is almost same

# ## Experiment 26 (5 hidden layers 256 224 192 160 128 neurons + batch normalization + dropouts)

# In[ ]:


model_5_drop_medium = Sequential()

model_5_drop_medium.add(Dense(256, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_medium.add(BatchNormalization())
model_5_drop_medium.add(Dropout(0.5))

model_5_drop_medium.add(Dense(224, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_medium.add(BatchNormalization())
model_5_drop_medium.add(Dropout(0.5))

model_5_drop_medium.add(Dense(192, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_medium.add(BatchNormalization())
model_5_drop_medium.add(Dropout(0.5))

model_5_drop_medium.add(Dense(160, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_medium.add(BatchNormalization())
model_5_drop_medium.add(Dropout(0.5))

model_5_drop_medium.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_drop_medium.add(BatchNormalization())
model_5_drop_medium.add(Dropout(0.5))

model_5_drop_medium.add(Dense(output_dim, activation='softmax'))


model_5_drop_medium.summary()


# In[ ]:


model_5_drop_medium.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_drop_medium.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_drop_medium_score = model_5_drop_medium.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_drop_medium_score[0]) 
print('Test accuracy:', model_5_drop_medium_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_drop_medium.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss
# - Accuracy was more for nn with more number of neurons in each layer

# ## Experiment 27 (5 hidden layers 128 112 96 80 64 neurons + batch normalization + dropouts)

# In[ ]:


model_5_drop_small = Sequential()

model_5_drop_small.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_small.add(BatchNormalization())
model_5_drop_small.add(Dropout(0.5))

model_5_drop_small.add(Dense(112, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_small.add(BatchNormalization())
model_5_drop_small.add(Dropout(0.5))

model_5_drop_small.add(Dense(96, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_small.add(BatchNormalization())
model_5_drop_small.add(Dropout(0.5))

model_5_drop_small.add(Dense(80, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))
model_5_drop_small.add(BatchNormalization())
model_5_drop_small.add(Dropout(0.5))

model_5_drop_small.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )
model_5_drop_small.add(BatchNormalization())
model_5_drop_small.add(Dropout(0.5))

model_5_drop_small.add(Dense(output_dim, activation='softmax'))


model_5_drop_small.summary()


# In[ ]:


model_5_drop_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_drop_small.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


# In[ ]:


model_5_drop_small_score = model_5_drop_small.evaluate(X_test, Y_test, verbose=1) 
print('Test score:', model_5_drop_small_score[0]) 
print('Test accuracy:', model_5_drop_small_score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# In[ ]:


w_after = model_5_drop_small.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
h4_w = w_after[6].flatten().reshape(-1,1)
h5_w = w_after[8].flatten().reshape(-1,1)
out_w = w_after[10].flatten().reshape(-1,1)


fig = plt.figure(figsize=(20,6))
plt.title("Weight matrices after model trained")
plt.subplot(1, 6, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 6, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 6, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 3 ')

plt.subplot(1, 6, 4)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 4 ')

plt.subplot(1, 6, 5)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='r')
plt.xlabel('Hidden Layer 5 ')

plt.subplot(1, 6, 6)
plt.title("Trained model Weights")
ax = sns.violinplot(y=out_w,color='y')
plt.xlabel('Output Layer ')
plt.show()


# - Both test and train loss are decresing.
# - First layer have weights in the form of normal distribution
# - Most of the weights of the output layer are positive
# - Test loss is always less than train loss
# - Accuracy was more for nn with more number of neurons in each layer

# ## Summary

# - The test loss with out using batch normalization or dropouts was heavly fluctuating.
# - This fluctuation seems to reduce with decrease in the number of neurons in in each layer.
# - Test loss was always greater than train loss.
# - The initial test loss was very high and in some models the test loss seems to increse. Especially the models having more number of neurons in each layer.
# - Similar observation was seen with batch normalization but the rate of change of test loss was less compared to models having no batch normalization.
# - Best performance was shown by models with dropouts. The test loss was always less than train loss. There was no fluctuation.
# - For models with dropouts the accuracy was more for models having more number of neurons in each hidden layer.

# ### Two hidden layers

# In[ ]:


from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Model No", "Hidden Layers", "Batch Normalization", "Dropouts", "test loss", "test accuracy"]
x.add_row([1,"1024-512", "No", "No", round(model_2_relu_score[0],4), round(model_2_relu_score[1],4)] )
x.add_row([2,"256-128", "No", "No", round(model_2_relu_medium_score[0],4), round(model_2_relu_medium_score[1],4)] )
x.add_row([3,"128-64", "No", "No", round(model_2_relu_small_score[0],4), round(model_2_relu_small_score[1],4)] )
x.add_row([4,"1024-512", "Yes", "No", round(model_batch_relu_score[0],4), round(model_batch_relu_score[1],4)] )
x.add_row([5,"256-128", "Yes", "No", round(model_batch_relu_medium_score[0],4), round(model_batch_relu_medium_score[1],4)] )
x.add_row([6,"128-64", "Yes", "No", round(model_batch_relu_small_score[0],4), round(model_batch_relu_small_score[1],4)] )
x.add_row([7,"1024-512", "Yes", "Yes", round(model_drop_score[0],4), round(model_drop_score[1],4)] )
x.add_row([8,"256-128", "Yes", "Yes", round(model_drop_medium_score[0],4), round(model_drop_medium_score[1],4)] )
x.add_row([9,"128-64", "Yes", "Yes", round(model_drop_small_score[0],4), round(model_drop_small_score[1],4)] )
print(x)


# ### Three Hidden Layer

# In[ ]:


x = PrettyTable()

x.field_names = ["Model No", "Hidden Layers", "Batch Normalization", "Dropouts", "test loss", "test accuracy"]

x.add_row([10,"1024-768-512", "No", "No", round(model_3_relu_score[0],4), round(model_3_relu_score[1],4)] )
x.add_row([11,"256-192-128", "No", "No", round(model_3_relu_medium_score[0],4), round(model_3_relu_medium_score[1],4)] )
x.add_row([12,"128-96-64", "No", "No", round(model_3_relu_small_score[0],4), round(model_3_relu_small_score[1],4)] )
x.add_row([13,"1024-768-512", "Yes", "No", round(model_3_batch_relu_score[0],4), round(model_3_batch_relu_score[1],4)] )
x.add_row([14,"256-192-128", "Yes", "No", round(model_3_batch_relu_medium_score[0],4), round(model_3_batch_relu_medium_score[1],4)] )
x.add_row([15,"128-96-64", "Yes", "No", round(model_3_batch_relu_small_score[0],4), round(model_3_batch_relu_small_score[1],4)] )
x.add_row([16,"1024-768-512", "Yes", "Yes", round(model_3_drop_score[0],4), round(model_3_drop_score[1],4)] )
x.add_row([17,"256-192-128", "Yes", "Yes", round(model_3_drop_medium_score[0],4), round(model_3_drop_medium_score[1],4)] )
x.add_row([18,"128-96-64", "Yes", "Yes", round(model_3_drop_small_score[0],4), round(model_3_drop_small_score[1],4)] )
print(x)


# ### Five Hidden Layer

# In[ ]:


x = PrettyTable()

x.field_names = ["Model No", "Hidden Layers", "Batch Normalization", "Dropouts", "test loss", "test accuracy"]
x.add_row([19,"1024-896-768-640-512", "No", "No", round(model_5_relu_score[0],4), round(model_5_relu_score[1],4)] )
x.add_row([20,"256-224-192-160-128", "No", "No", round(model_5_relu_medium_score[0],4), round(model_5_relu_medium_score[1],4)] )
x.add_row([21,"128-112-96-80-64", "No", "No", round(model_5_relu_small_score[0],4), round(model_5_relu_small_score[1],4)] )
x.add_row([22,"1024-896-768-640-512", "Yes", "No", round(model_5_batch_relu_score[0],4), round(model_5_batch_relu_score[1],4)] )
x.add_row([23,"256-224-192-160-128", "Yes", "No", round(model_5_batch_relu_medium_score[0],4), round(model_5_batch_relu_medium_score[1],4)] )
x.add_row([24,"128-112-96-80-64", "Yes", "No", round(model_5_batch_relu_small_score[0],4), round(model_5_batch_relu_small_score[1],4)] )
x.add_row([25,"1024-896-768-640-512", "Yes", "Yes", round(model_5_drop_score[0],4), round(model_5_drop_score[1],4)] )
x.add_row([26,"256-224-192-160-128", "Yes", "Yes", round(model_5_drop_medium_score[0],4), round(model_5_drop_medium_score[1],4)] )
x.add_row([27,"128-112-96-80-64", "Yes", "Yes", round(model_5_drop_small_score[0],4), round(model_5_drop_small_score[1],4)] )
print(x)


# In[ ]:




