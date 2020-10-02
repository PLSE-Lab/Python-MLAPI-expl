#!/usr/bin/env python
# coding: utf-8

# **Computer Vision is a field of study that seeks to develop computers to 'see' i.e visualize real world in the form of images, videos.**
# 
# In this notebook, we attempt to teach computers to read house numbers that were captured by Google street view cars. These house numbers come in all shapes and sizes, our model should be intelligent enough to remove the noise from the image that may have crept in and identify numbers accurately.
# 
# **Code Structure:**
# * Import packages, Visualize dataset
# * Pre-process the input to be fit into the model
# * Build Convolutional Neural Network
# * Use pre-trained weights for Transfer Learning
# * Check Model accuracy
# * Visualize Model predictions
# 
# *Special thanks to: *
# 
# *1. https://machinelearningmastery.com/what-is-computer-vision/ This is a treasure trove for everything on Machine Learning*
# 
# *2. https://github.com/BVLC/caffe/wiki/Model-Zoo List of pre-trained weights*

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


# ## Load the data

# In[ ]:


import h5py
import numpy as np

h5f = h5py.File('/kaggle/input/street-view-house-nos-h5-file/SVHN_single_grey1.h5', 'r')
h5w = h5py.File('/kaggle/input/cnn-mnist-weights-pretrained/cnn_mnist_weights.h5', 'r')
h5f


# ## Import train and test sets

# In[ ]:


X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]

h5f.close() #close this file


# **To understand the breadth and depth of the data, lets check shape of data.**

# In[ ]:


print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_train.shape)
print('y_test:', y_train.shape)


# ## Visualize the dataset

# **Showing the first 100 test images, we have to build a model that would classify these images accurately. **

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize = (10,10))

rows = 10
columns = 10
w = 10
h = 10

for i in range(1, rows * columns + 1):
    img = X_test[i]
    fig.add_subplot(rows, columns,i)
    plt.imshow(img, cmap = 'gray')
plt.show()


# In[ ]:


print(X_train.shape) #before reshape


# We will use **Transfer Learning** principles to classify these images. The model we will be using was trained on MNIST datatset, where the image shape were 28,28. While the images we have is 32,32 therefore we have to resize into a shape compatible with the model.
# 
# Use **OpenCV** to resize images.

# In[ ]:


import cv2
#Create zero array for X_train, X_test
X_train_res = np.zeros((X_train.shape[0], 28,28), dtype = np.float32) #create a zero array of size 28*28 same as MNIST
X_test_res = np.zeros((X_test.shape[0], 28, 28), dtype = np.float32)

for i in range(X_train.shape[0]):
    X_train_res[i,:,:] = cv2.resize(X_train[i], dsize = (28,28), interpolation = cv2.INTER_CUBIC)
    
for i in range(X_test.shape[0]):
    X_test_res[i,:,:] = cv2.resize(X_test[i], dsize = (28,28), interpolation = cv2.INTER_CUBIC)
    
print(X_train_res.shape)
print(X_test_res.shape)


# In[ ]:


img_rows, img_cols = 28, 28

X_train_CNN = X_train_res.reshape(X_train_res.shape[0], img_rows, img_cols, 1)
X_train_CNN.shape
X_test_CNN = X_test_res.reshape(X_test_res.shape[0], img_rows, img_cols, 1)
print(X_train_CNN.shape)

#Shape of 1 image would be as given below, this would be useful while creating models
input_shape  = (img_rows, img_cols, 1)
print(input_shape)


# We need to preprocess this i.e normalize the input. This ensures none of the columns would dominate the other.

# In[ ]:


X_train_CNN = X_train_CNN.astype('float32')
X_test_CNN =  X_test_CNN.astype('float32')

#Normalizing the input
X_train_CNN = X_train_CNN / 255.0
X_test_CNN = X_test_CNN / 255.0

print(X_train_CNN.shape)


# Now that we have normalized the input, lets check whether y_train are in the right format to be inserted into the model or not. As we observe below, it needs to be converted into One Hot Encoding vectors. Else it would lead to one of the column dominating the others.

# In[ ]:


y_train


# In[ ]:


#convert class vectors to binary class metrics
num_classes = 10 # since we will only classify nos between 0-9
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train[0]


# ## Build a Convolutional Neural Network

# In[ ]:


#Set model hyperparameters
num_classes = 10

#Define the layers of the model
model_CNN = Sequential()

#1. Conv Layer
model_CNN.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape, name = 'Conv1'))

#2. Conv Layer
model_CNN.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = input_shape, name = 'Conv2'))

#3. MaxPooling Layer
model_CNN.add(MaxPooling2D(pool_size = (2,2), name = 'Max1'))

#4. Dropout this prevents model from overfitting
model_CNN.add(Dropout(0.25, name = 'Drop1'))

#5. Flatten Layer
model_CNN.add(Flatten())

#6. Fully Connected Layer
model_CNN.add(Dense(128, activation = 'relu', name = 'Dense1'))

#7. Dropout
model_CNN.add(Dropout(0.5, name = 'Drop2'))

#8. Fully Connected Layer
model_CNN.add(Dense(num_classes, activation = 'softmax', name = 'Dense2'))

model_CNN.summary()


# **Freeze only the initial Convolutional layer weights and train dense FC layers.**
# 
# In transfer learning, we use some of the bottom layers as it is, while we tweak the top layers as required for a dataset. Here, we won't waste time in finding the best weights and we would only modify Dense layers.

# In[ ]:



for layer in model_CNN.layers:
    if('Dense' not in layer.name):
        layer.trainable = False
    else:
        layer.trainable = True
        
#Module to output colorful statements
from termcolor import colored

#Check which layers have been frozen
for layer in model_CNN.layers:
    print(colored(layer.name, 'blue'))
    print(colored(layer.trainable, 'red'))
    


# **Load pre-trained weights from MNIST CNN model**

# In[ ]:


model_CNN.load_weights('/kaggle/input/cnn-mnist-weights-pretrained/cnn_mnist_weights.h5')


# ## Set Optimizer, Loss, Metrics

# In[ ]:


from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

optimizer = Adam(lr = 0.001)
batch_size = 128
num_classes = 10
epochs = 12

model_CNN.compile(optimizer = optimizer, loss = categorical_crossentropy, metrics = ['accuracy'])


# ## Model Fitting

# In[ ]:


model_CNN.fit(X_train_CNN, y_train,
             batch_size = batch_size,
             epochs = epochs,
             verbose = 1,
             validation_data = (X_test_CNN, y_test))
             #callbacks = [tensorboard_callback, early_stopping, model_checkpoint])


# **CNN accuracy is 86% which is good but can be improved further when using different model structures. **
# 
# **Lets also evaluate test set.**

# In[ ]:


score = model_CNN.evaluate(X_test_CNN, y_test)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])


# ## Visualize some predictions

# In[ ]:


plt.figure(figsize = (2,2))
plt.imshow(X_test_CNN[30].reshape(28,28), cmap = 'gray') #image, reshape size, cmap
plt.show()
print(np.argmax(model_CNN.predict(X_test_CNN[30].reshape(1,28,28,1))))

plt.figure(figsize = (2,2))
plt.imshow(X_test_CNN[50].reshape(28,28), cmap = 'gray') #image, reshape size, cmap
plt.show()
print(np.argmax(model_CNN.predict(X_test_CNN[50].reshape(1,28,28,1))))

plt.figure(figsize = (2,2))
plt.imshow(X_test_CNN[78].reshape(28,28), cmap = 'gray') #image, reshape size, cmap
plt.show()
print(np.argmax(model_CNN.predict(X_test_CNN[78].reshape(1,28,28,1))))

plt.figure(figsize = (2,2))
plt.imshow(X_test_CNN[130].reshape(28,28), cmap = 'gray') #image, reshape size, cmap
plt.show()
print(np.argmax(model_CNN.predict(X_test_CNN[130].reshape(1,28,28,1))))


# ## Save the trained weights and model

# In[ ]:


model_CNN.save('./cnn_svhn.h5')
model_CNN.save_weights('./cnn_svhn_weights.h5')


# ***This kernel is a work in progress...***
