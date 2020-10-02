#!/usr/bin/env python
# coding: utf-8

# # Simple CNN Classifier for notMNIST
# 
# This notebook uses Keras to create a convolutional neural network for classification of the images of letters in the dataset. We achieve about 94% testing accuracy after 40 epochs. The smaller dataset was chosen since its labeling error rate was lower.

# Firstly we import the libraries we'll need.

# In[ ]:


import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# Next we'll load up all of our images from the subdirectories, storing the labels in a list with one-hot encoding. 

# In[ ]:


X = []
labels = []
DATA_PATH = '../input/notMNIST_small/notMNIST_small'
# for each folder (holding a different set of letters)
for directory in os.listdir(DATA_PATH):
    # for each image
    for image in os.listdir(DATA_PATH + '/' + directory):
        # open image and load array data
        try:
            file_path = DATA_PATH + '/' + directory + '/' + image
            img = Image.open(file_path)
            img.load()
            img_data = np.asarray(img, dtype=np.int16)
            # add image to dataset
            X.append(img_data)
            # add label to labels
            labels.append(directory)
        except:
            None # do nothing if couldn't load file
N = len(X) # number of images
img_size = len(X[0]) # width of image
X = np.asarray(X).reshape(N, img_size, img_size,1) # add our single channel for processing purposes
labels = to_categorical(list(map(lambda x: ord(x)-ord('A'), labels)), 10) # convert to one-hot


# Then shuffle the data around (to get rid of alphabetization) and split the data into test and training sets. 

# In[ ]:


temp = list(zip(X, labels))
np.random.shuffle(temp)
X, labels = zip(*temp)
X, labels = np.asarray(X), np.asarray(labels)
PROP_TRAIN = 0.7 # proportion to use for training
NUM_TRAIN = int(N * PROP_TRAIN) # number to use for training
X_train, X_test = X[:NUM_TRAIN], X[NUM_TRAIN:]
labels_train, labels_test = labels[:NUM_TRAIN], labels[NUM_TRAIN:]


# Print out a couple of the images so we can see what we're working with. 

# In[ ]:


num_rows, num_cols = 3, 5
fig, axes = plt.subplots(num_rows, num_cols)
for i in range(num_rows):
    for j in range(num_cols):
        axes[i, j].imshow(X[num_cols*i + j, :, :, 0], cmap='gray')
        axes[i, j].axis('off')
fig.suptitle('Sample Images from Dataset')
plt.show()


# Now we can start to define our model with Keras. Two blocks of conv/pool are used and then connected to two fully connected layers. Selu activation is used to help circumvent need for batch normalization. 

# In[ ]:


shape = X[0].shape
img_in = Input(shape=shape, name='input')
x = Dropout(0.2, name='input_dropout')(img_in)

# conv block 1
x = Conv2D(16, (3,3), activation='selu', name='block1_conv1')(x)
x = Conv2D(32, (3,3), activation='selu', name='block1_conv2')(x)
x = MaxPooling2D((2,2), name='block1_pool')(x)
block_1 = Dropout(0.5, name='block1_dropout1')(x)

# conv block 2
x = Conv2D(32, (3,3), use_bias=False, activation='selu', name='block2_conv1')(block_1)
x = Conv2D(64, (3,3), use_bias=False, activation='selu', name='block2_conv2')(x)
x = MaxPooling2D((2,2), name='block2_pool')(x)
block_2 = Dropout(0.5, name='block2_dropout1')(x)

# dense block 3
x = Flatten(name='block3_flatten')(block_2)
x = Dense(128, activation='selu', name='block3_dense1')(x)
x = Dropout(0.5, name='block3_dropout1')(x)
x = Dense(128, activation='selu', name='block3_dense2')(x)
x = Dropout(0.5, name='block3_dropout2')(x)

# output layer
output = Dense(10, activation='softmax', name='output')(x)

# compile model
model = Model(img_in, output)
model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# Now we train the model for 40 epochs, checking accuracy on our test data each epoch. We also make a callback for logging to a CSV file each epoch so we can visualize our training. 

# In[ ]:


csv_logger = keras.callbacks.CSVLogger('training.csv', append=True)

# train model
model.fit(X_train, labels_train,
          epochs=40, batch_size=64,
          validation_data=[X_test, labels_test],
          callbacks=[csv_logger])


# Now we'll visualize the training and testing accuracy over time. The right-hand plot has values given by *log(1-a)* where *a* is the accuracy. 

# In[ ]:


data = np.genfromtxt('training.csv', delimiter=',')
data = data[1:][:,1:]

fig, axes = plt.subplots(1, 2)

# plot train and test accuracies
axes[0].plot(data[:,0]) # training accuracy
axes[0].plot(data[:,2]) # testing accuracy
axes[0].legend(['Training', 'Testing'])
axes[0].set_title('Accuracy Over Time')
axes[0].set_xlabel('epoch')
axes[0].set_ybound(0.0, 1.0)

# same plot zoomed into [0.85, 1.00]
axes[1].plot(np.log(1-data[:,0])) # training accuracy
axes[1].plot(np.log(1-data[:,2])) # testing accuracy
axes[1].legend(['Training', 'Testing'])
axes[1].set_title('Log-Inverse Accuracy')
axes[1].set_xlabel('epoch')
#axes[1].set_ybound(0.90,1.0)
plt.show()


# Finally we evaluate the model's accuracy on the testing data. 

# In[ ]:


score = model.evaluate(X_test, labels_test, verbose=False)
print('Loss: {}'.format(score[0]))
print('Accuracy: {}%'.format(np.round(10000*score[1])/100))


# In[ ]:




