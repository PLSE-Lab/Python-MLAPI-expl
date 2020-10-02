#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import stuff
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for plotting


# ### Difference between batch & epoch
# ![Batch vs Epoch](https://i.imgur.com/Zsb38ZL.png)
# Read more at [StackOverflow - Epoch vs Iteration when training neural networks](https://stackoverflow.com/a/31842945)

# In[ ]:


# How many images to send to GPU?
batch_size = 128

# How many target classes are in the dataset? (10 number, 0 - 9)
num_classes = 10

# Epoch = neural network have seen all training examples
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28


# In[ ]:


# loading the training dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# the data
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
x_test = test.values.astype('float32') # this is not a validation set. We don't have labels for this!


# In[ ]:


# preview the images
plt.figure(figsize=(12,10))
plot_cols, plot_rows = 10, 4
for i in range(40):
    plt.subplot(plot_rows, plot_cols, i+1)
    plt.imshow(x_train[i+50, :].reshape((28,28)),interpolation='nearest')
plt.show()


# In[ ]:


# Pixel values are from range 0 - 255. We must normalize to 0-1 (this is super important)!
x_train = x_train / 255.0
x_test = x_test / 255.0


# In[ ]:


# Different Keras backends can have different index ordering!
if K.image_data_format() == 'channels_first':
    # array indexing is x_train[image_index, color, y, x]
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # array indexing is x_train[image_index, y, x, color]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[ ]:


# convert class vectors to binary class matrices 
# this means that: (2 -> [0,0,1,0,0,0,0,0,0,0])
# in other words: we want network to put probability 100% (value 1.0) on index associated with number 2
y_train_matrix = keras.utils.to_categorical(y_train, num_classes)
y_train_matrix[0, :]


# In[ ]:


# split to train and validation set (remember, x_test is not validation set, but Kaggle test dataset without labels!)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train_matrix, test_size = 0.1, random_state=42)
print(X_train.shape, X_val.shape)


# In[ ]:


model = Sequential()

# https://keras.io/layers/core/
model.add(TODO ADD LAYERS (please use even number of neurons!))

# https://keras.io/losses/
# https://keras.io/optimizers/
model.compile(loss='TODO_CHOOSE_LOSS',
              optimizer=TODO_CHOOSE_OPTIMIZER,
              metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))

score = model.evaluate(X_val, Y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# show N-th layer neuron weights (this is going to be nice espeically for network without hidden layer!)
LAYER_INDEX = 1
weights = model.get_layer(name=Dense, index=LAYER_INDEX).get_weights()

x, y = 6, int(weights[0].shape[1] / 6)
for i in range(weights[0].shape[1]):  
    plt.subplot(y, x, i+1)
    plt.imshow(weights[0][:,i].reshape(28,28))
plt.show()


# In[ ]:


## Kaggle stuff

#get the predictions for the test data
predicted_classes = model.predict_classes(x_test)

# create submission file
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
# save results
submissions.to_csv("submission.csv", index=False, header=True)

# save network
model.save('my_awesome_model.h5')
json_string = model.to_json()

