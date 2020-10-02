#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# There are several parameters to a convolutional neural network. I am experimenting with some parameters of the CNN like kernel size of convolutional layer, maxpool layer, dropouts and number of neurons in each convolutional layer to see if there is any change in the performance of the model with a change in above mentioned parameters.
# 
# I have done similar experements in simple mlp  you can see them in the below link. Feel free to share your thoughts about these experements.
# 
# [https://www.kaggle.com/vishnurapps/experimenting-with-neural-networks](https://www.kaggle.com/vishnurapps/experimenting-with-neural-networks)
# 

# ## Importing libraries

# In[ ]:


# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

nb_epoch = 12


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


# ## Experiment 1 
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 2x2 + Drop out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n \nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Test loss seems to reduce eventually
# - At the end of 10 epochs test loss slightly increased and again reduced.

# ## Experiment 2 
# ### 2D Conv 5x5 Kernel o/p 32 + 2D Conv 5x5 Kernel o/p 64 + Max Pooling Kernel 2x2 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(5, 5),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (5, 5), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - There is a slight imporvement in accuracy and slight decrease in loss after increasing the convolution kernel size from 3x3 to 5x5

# ## Experiment 3
# ### 2D Conv 7x7 Kernel o/p 32 + 2D Conv 7x7 Kernel o/p 64 + Max Pooling Kernel 2x2 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(7, 7),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (7, 7), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - There is a slight decrease in loss after increasing the convolution kernel size from 5x5 to 7x7

# ## Experiment 4
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 3x3 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(3, 3)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increaseing the maxpool kernel from 2x2 to 3x3 has improved the accuracy and reduced loss significantly

# ## Experiment 5
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 5x5 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(5, 5)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increaseing the maxpool kernel from 3x3 to 5x5 has no advantage
# - There is no change in the loss or accuracy compared with the 2x2 maxpool kernel

# ## Experiment 5
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 2x2 + Droup out 0.4 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.4))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increased dropout rate after the first maxpool and it acted negetively.
# - The performance is bad compared to old model with 0.25 dropout rate

# ## Experiment 6
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 2x2 + Droup out 0.5 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.5))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increased dropout rate after the first maxpool and it acted negetively again
# - The performance is bad compared to old model with 0.25 dropout rate

# ## Experiment 7
# ### 2D Conv 3x3 Kernel o/p 32 + 2D Conv 3x3 Kernel o/p 64 + Max Pooling Kernel 2x2 + Droup out 0.6 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.6))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increased dropout rate after the first maxpool and it acted negetively
# - Loss and accuracy is worse than all previous models
# - The performance is bad compared to old model with 0.25 dropout rate

# ## Experiment 8
# ### 2D Conv 3x3 Kernel o/p 20 + 2D Conv 3x3 Kernel o/p 40 + Max Pooling Kernel 2x2 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(20, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(40, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Reduced the number of neurons in the first and second convolution layer. 
# - The performance has reduced compared to the first model.
# - Valiation loss is worse compared to all prior models.

# ## Experiment 9
# ### 2D Conv 3x3 Kernel o/p 40 + 2D Conv 3x3 Kernel o/p 20 + Max Pooling Kernel 2x2 + Droup out 0.25 + Flatten + Drop out 0.5

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(Conv2D(40, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(20, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nhistory = model.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))")


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,nb_epoch+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)


# - Increased the number of neurons in the first and reduced the number of neurons in the second convolution layer. 
# - The performance has improved slightly compared to the previous model..
# - Valiation loss is inproved compared to prior model.

# ## Conclusion

# - Increasing convolutional kernel size has shown an increase in accuray and better reducution in validation loss
# - Increasing maxpool kernel size has negative impact on loss and accuracy
# - Increasing the dropout rate wont simply improve the performance of the model. I has negative effect. Need to do hyper parameter tuning to find best value.
# - More nuber of neurons in convolution layers show better performance than less number of neurons
# 

# In[ ]:


from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Model description",  "test loss", "test accuracy"]
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 2x2 \n Dropout 0.25 \n Flatten \n Dense Dropout 0.5", 0.0302, .9904])
x.add_row(["Conv 2D 5x5 32\n Conv 2D 5x5 64\n Maxpool 2x2 \n Dropout 0.25 \n Flatten \n Dense Dropout 0.5", 0.0231, .9926])
x.add_row(["Conv 2D 7x7 32\n Conv 2D 7x7 64\n Maxpool 2x2 \n Dropout 0.25 \n Flatten \n Dense Dropout 0.5", 0.0227, .9937])
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 3x3 \n Dropout 0.25 \n Flatten \n Dense Dropout 0.5", 0.0275, .9924])
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 5x5 \n Dropout 0.25 \n Flatten \n Dense Dropout 0.5", 0.0284, .9925])
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 2x2 \n Dropout 0.40 \n Flatten \n Dense Dropout 0.5", 0.0294, .9908])
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 2x2 \n Dropout 0.50 \n Flatten \n Dense Dropout 0.5", 0.0305, .9908])
x.add_row(["Conv 2D 3x3 32\n Conv 2D 3x3 64\n Maxpool 2x2 \n Dropout 0.60 \n Flatten \n Dense Dropout 0.5", 0.0285, .9904])
x.add_row(["Conv 2D 3x3 20\n Conv 2D 3x3 40\n Maxpool 2x2 \n Dropout 0.60 \n Flatten \n Dense Dropout 0.5", 0.0285, .9904])
x.add_row(["Conv 2D 3x3 40\n Conv 2D 3x3 20\n Maxpool 2x2 \n Dropout 0.60 \n Flatten \n Dense Dropout 0.5", 0.0285, .9904])

print(x)


# In[ ]:




