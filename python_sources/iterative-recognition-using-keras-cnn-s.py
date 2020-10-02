#!/usr/bin/env python
# coding: utf-8

# # Purpose
# I'm trying to get familiar with the Kaggle kernels and explore model building. 
# # Approach
# I'll start with an simple CNN architecture and iterate to improve performance of the model. 
# # 1. Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Load and handle data
# The first column of the training data represents the output for the data, in which the rest of the columns represent the input. Kaggle does their testing on the test set. So in order to have my own guage on how well I'm going I divide the training set into two and have a validation set. Here I'm assuming the data sets come from the same distribution. Since we have a relatively small data I choose a 20% split.

# In[ ]:


# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

x_train = train.iloc[:,1:].values.astype('float64')
y_train = train.iloc[:,0].values.astype('int32')
test = test.values.astype('float64')

m = x_train.shape[0]

# Normalize and reshape
x_train = x_train / 255
x_train = x_train.reshape((m, 28, 28, 1))
test = test / 255
test = test.reshape((-1, 28, 28, 1))

#One-hot output representaion
y_train = to_categorical(y_train, num_classes = 10)

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

print("Number of examples: ", m)
print("Training input shape: ", x_train.shape)
print("Training output shape: ", y_train.shape)
print("Validate input shape: ", x_val.shape)
print("Validate output shape: ", y_val.shape)
print("Test input shape: ", test.shape)


# ## Here's how the data looks like (below the code to show them) 

# In[ ]:


# Plot consecutive images
def sample_images(x,offset=0, sample_num=10):
    for i in range(sample_num):
        plt.subplot(math.ceil(sample_num/5), 5, i+1)
        plt.imshow(x[offset + i][:, :, 0])
    plt.show()

# Plot an image given it's index
def sample_image(x, index):
    plt.imshow(x[index][:, :, 0])


#sample_image(x_train, 1)
sample_images(x_train)


# # 3. Hyperparameters
# I'll use epochs of 10 at each iteration and maybe a 30 for a final performance. I'll use a power of two for batches, 64 should be OK. (Hoping computer architecture helps performance)

# In[ ]:


epochs = 20
batch_size = 64
learning_rate = 0.001
activation = 'tanh'


# ## Some custom activation functions below

# In[ ]:


# Custom activations
def swish(x, beta=1):
    return (K.sigmoid(beta*x) * x)

def aria(x, alpha=1.25, beta=1):
    return ((K.sigmoid(beta*x)**alpha) * x)

get_custom_objects().update({'swish': swish, 'aria': aria})


# # 4. Data augmentation
# More relevant data is always welcome. Do not augment validation set as validation data should come from the same distributuion as the test set!<br/><br/> 

# In[ ]:


datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
datagen.fit(x_train)

batches = datagen.flow(x_train,y_train, batch_size=batch_size)
print("Number of training batches: ", len(batches))

first_batch = batches[0][0]
sample_images(first_batch, sample_num=25)


# You can see the smaples from the augmented training set above. 

# # 5. Models, Step by Step

# The first model I'll use will be a simple CNN. 

# In[ ]:


def model_1():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     activation=activation, input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model


# Model 1, after 10 epochs, achives (loss: 0.0414 - acc: 0.9880 - val_loss: 0.0843 - val_acc: 0.9762). <br/> 
# For this MNSIT example I'm not getting into avoidable bias and assuming the best we can do is 1. That gives as an underfitting of 0.012 and overfitting 0.0118. These are already good results in my opinion. But let's be greedy, this might be a tutorial competition but still a competition. Let's go for a bigger network. 

# In[ ]:


#LeNet-5 like network
def model_2():
    model = Sequential()
    
    # Layer1
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation=activation,
                     padding = 'same', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    # Layer2
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation=activation, 
                    padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2,2)))
    #Layer3
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    #Layer4
    model.add(Dense(84, activation=activation))
    #Layer5
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model


# With valid padding(stride = 2):    (loss: 0.0124 - acc: 0.9966 - val_loss: 0.0558 - val_acc: 0.9838) <br />
# With valid padding(stride = 1):     (loss: 0.0124 - acc: 0.9964 - val_loss: 0.0510 - val_acc: 0.9849)  <br />
# With same padding(stride = 2):   (loss: 0.0071 - acc: 0.9981 - val_loss: 0.0495 - val_acc: 0.9849)  <br />
# With same padding(stride = 1):    (loss: 0.0080 - acc: 0.9976 - val_loss: 0.0462 - val_acc: 0.9873)  <br />
# Same-valid padding(stride = 1):   (loss: 0.0073 - acc: 0.9980 - val_loss: 0.0533 - val_acc: 0.9839)  <br />
# If you ask me why I tried all these combinations while the last one is more like the actual LeNet5, Keras just makes it too easy to try with no need to worry about the shapes of weights.  <br /><br />
# Now handling overfitting is more worthwhile for this model. Let's regularize. 

# In[ ]:


# LeNet5 variation
def model_3():
    model = Sequential()
    
    # Layer1
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation=activation,
                     padding = 'same', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    # Layer2
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation=activation, 
                    padding = 'same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    #Layer3
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    model.add(Dropout(0.25))
    #Layer4
    model.add(Dense(84, activation=activation))
    model.add(Dropout(0.25))
    #Layer5
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model


# It didn't quite converge in 10 epochs now I'm using 20 with a book in front.<br/>
# layer3 dropout 0.5, layer4 dropout 0.25:<br/>
# (loss: 0.0287 - acc: 0.9907 - val_loss: 0.0518 - val_acc: 0.9854)  <br/>
# layer3 dropout 0.25, layer4 dropout 0.25:<br/>
# (loss: 0.0133 - acc: 0.9958 - val_loss: 0.0507 - val_acc: 0.9881)  <br/>
# 
# <br/> For me convolutional weights always seem more precious so I first apply dropout on dense layers. 

# ## Evaluate model 

# In[ ]:


optimizer = Adam(lr=learning_rate)
model = model_3()
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


fitted_model = model.fit_generator(generator=batches, validation_data = (x_val,y_val), 
                                   epochs = epochs, steps_per_epoch = m // batch_size)


# In[ ]:


# Evaluate model
# Loss plot
plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# # 6. Finishing
# Upon learning with the augmented data and having finalized the model, we can make use of the whole training set (including validation set). 

# In[ ]:


fitted_model = model.fit(np.concatenate((x_train, x_val), axis=0), 
                         np.concatenate((y_train, y_val), axis=0), 
                         batch_size=batch_size, epochs=epochs)


# In[ ]:


ypred = model.predict(test)
ypred = np.argmax(ypred,axis=1)


# # Final notes
# You can always loop between improving model, fighting overfitting and evaluating. Model_3 isn't the "end all and be all". <br/>

# In[ ]:


submissions = pd.DataFrame({"ImageId": list(range(1,len(ypred)+1)),
                         "Label": ypred})
submissions.to_csv("cnn_model.csv", index=False, header=True)

