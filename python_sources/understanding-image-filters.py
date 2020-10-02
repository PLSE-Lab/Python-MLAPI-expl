#!/usr/bin/env python
# coding: utf-8

# # Understanding image filters
# In this kernel, I will try to explain how convolution filters work in image analysis. Why such a subject ? Because it is the basis of every CNN models. To do that, I will first explain how a filter is applied on an image and then show some classic filters.
# 
# 1. Convolution filters
# 2. Examples
# 3. Learned filters visualisation

# ## Convolution filters
# A convolution filter is applied on an image to output another image. This is how it works
# 
# * Superposition on the image (seen as a matrix of numbers).
# * Multiplication between filter elements and their corresponding image pixel
# * Sum of all the multiplication results
# * Creation of a new image with new pixel values
# ![How to apply a convolution filter](http://machinelearninguru.com/_images/topics/computer_vision/basics/convolution/1.JPG)

# ## Examples
# ### Utils

# In[ ]:


# Imports
import numpy as np
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt
import math
import random


# In[ ]:


#function to load data
def load_dataset():
    train_data = h5py.File('../input/happy-house-dataset/train_happy.h5', "r")
    x_train = np.array(train_data["train_set_x"][:])  
    return x_train

# Load the data
X_train = load_dataset()


# In[ ]:


# Small function to create a dummy image, basically 4 different squares
def create_dummy_image(h=64, w=64):
    img = np.zeros((h, w))
    img[:h//2,:w//2] = 255
    img[h//2:,:w//2] = 200
    img[h//2:,w//2:] = 100
    return img


# In[ ]:


# RGB to grayscale function --> filters can be applied channel per channel 
# but here we will focus only on grayscale images
def to_grayscale(img):
    h = len(img)
    w = len(img[0])
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i][j] = int(img[i][j][0] * 0.3 + img[i][j][1] * 0.59 + img[i][j][2] * 0.11)
    return new_img


# In[ ]:


# Let's look at some images
img1 = to_grayscale(X_train[0])
img2 = to_grayscale(X_train[1])
img3 = to_grayscale(X_train[42])
dummy_img = create_dummy_image()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
axes[0].imshow(img1, cmap='Greys_r')
axes[1].imshow(img2, cmap='Greys_r')
axes[2].imshow(img3, cmap='Greys_r')
axes[3].imshow(dummy_img, cmap='Greys_r')
plt.show()


# In[ ]:


# small function to apply a 2D filter on a 1-channel image
def apply_conv_filter(img, conv_filter):
    h = len(img)
    w = len(img[0])
    hf = len(conv_filter)
    wf = len(conv_filter[0])
    h_off = hf // 2
    w_off = wf // 2
    new_img = np.zeros((h, w))
    for i in range(h_off, h - h_off):
        for j in range(w_off, w - w_off):
            new_value = 0
            for k in range(hf):
                for l in range(wf):
                    new_value += img[i - h_off + k][j - w_off + l] * conv_filter[k][l]
            new_img[i][j] = abs(new_value)
    
    return new_img


# ### Edge detection filters
# Here, I will show some basic edge detection filters.
# 
# First, what is an edge ? It's a border between two uniform zone. On a grayscale image, it's a border between a high value zone and a low value one. By saying that, we can get the intuition of how we can detect these edges : find the place where there is a big shift in pixel values.

# In[ ]:


# Let's create basic gradient filters
grad_v = [[-1, 0, 1]]
grad_h = [[-1],
          [0],
          [1]]


# Here, I have defined two filters. One is a line, the other one a column.
# 
# The first one, *grad_v* is a line that will compare the pixel to our left to the pixel to our right. It can be seen as a gradient, that will highlight vertical edges.
# The second one, *grad_h* is a column. It will act as the first filter but for horizontal edges.

# In[ ]:


# Let's define a last one
grad = [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]]


# This last filter acts the same way than the two previous one, it's a gradient. But it tries to get all the edges, and not only the ones along one axis.
# 
# Let's look at their effect on some images.

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
axes[0][0].imshow(img1, cmap='Greys_r')
axes[1][0].imshow(apply_conv_filter(img1, grad_v), cmap='Greys_r')
axes[2][0].imshow(apply_conv_filter(img1, grad_h), cmap='Greys_r')
axes[3][0].imshow(apply_conv_filter(img1, grad), cmap='Greys_r')

axes[0][1].imshow(img2, cmap='Greys_r')
axes[1][1].imshow(apply_conv_filter(img2, grad_v), cmap='Greys_r')
axes[2][1].imshow(apply_conv_filter(img2, grad_h), cmap='Greys_r')
axes[3][1].imshow(apply_conv_filter(img2, grad), cmap='Greys_r')

axes[0][2].imshow(img3, cmap='Greys_r')
axes[1][2].imshow(apply_conv_filter(img3, grad_v), cmap='Greys_r')
axes[2][2].imshow(apply_conv_filter(img3, grad_h), cmap='Greys_r')
axes[3][2].imshow(apply_conv_filter(img3, grad), cmap='Greys_r')

axes[0][3].imshow(dummy_img, cmap='Greys_r')
axes[1][3].imshow(apply_conv_filter(dummy_img, grad_v), cmap='Greys_r')
axes[2][3].imshow(apply_conv_filter(dummy_img, grad_h), cmap='Greys_r')
axes[3][3].imshow(apply_conv_filter(dummy_img, grad), cmap='Greys_r')
plt.show()


# First, look at the dummy image. The vertical gradient filter only detects vertical edges. Moreover, the magnitude of the border in the *edge image* depends on the difference of pixel values along this border. This filter does exactly what it is supposed to do !!
# We can get the same type of conclusion for the two other filters.

# ### Blur filters
# Here we will look at the effect of two blur filters, the *box* one and the gaussian one. The purpose of using a blur filter is to, well..., blur an image. This can be done to reduce the noise of the image, to remove weird pixels, to smooth the image.

# The box filter aims at replacing the old pixel value by the mean of all the pixel of its neighbourhood. Here, it's a 3*3 square neighbourhood.
# The idea behind this is to smooth the image by taking into account multiple pixel values to reduce the impact of singular extreme points

# In[ ]:


# Box filter
box_filter = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

# Here, we want to normalize the filter so the pixel values of our output
# image remain in a valid range
box_filter = (1/9) * box_filter


# The gaussian blur works the sme way, except it uses a ponderation in the computation of the local mean around a pixel, to give more weights to closer pixels.

# In[ ]:


# Gaussian filter
gauss_filter = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])

# Here, we want to normalize the filter so the pixel values of our output
# image remain in a valid range
gauss_filter = (1/16) * box_filter


# Let's look at the effect of these two new filters.

# In[ ]:


# First, let's add some noise to the top-right corner of the dummy image
for i in range(32):
    for j in range(32):
        dummy_img[i][32+j] = random.randint(0, 25)

# and a singular point
dummy_img[42][42] = 255


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes[0][0].imshow(img1, cmap='Greys_r')
axes[1][0].imshow(apply_conv_filter(img1, box_filter), cmap='Greys_r')
axes[2][0].imshow(apply_conv_filter(img1, gauss_filter), cmap='Greys_r')

axes[0][1].imshow(img2, cmap='Greys_r')
axes[1][1].imshow(apply_conv_filter(img2, box_filter), cmap='Greys_r')
axes[2][1].imshow(apply_conv_filter(img2, gauss_filter), cmap='Greys_r')

axes[0][2].imshow(img3, cmap='Greys_r')
axes[1][2].imshow(apply_conv_filter(img3, box_filter), cmap='Greys_r')
axes[2][2].imshow(apply_conv_filter(img3, gauss_filter), cmap='Greys_r')

axes[0][3].imshow(dummy_img, cmap='Greys_r')
axes[1][3].imshow(apply_conv_filter(dummy_img, box_filter), cmap='Greys_r')
axes[2][3].imshow(apply_conv_filter(dummy_img, gauss_filter), cmap='Greys_r')
plt.show()


# We can see on the dummy image that almost all the noise of the top right corner has disappeared. Also, the singularity has been smoothed. But we also lost details on the edges. Usually, these filters are used before other filters, like the edge detection ones, to smooth the image and to avoid to detect noise as small edges.

# ## Learned filters visualisation
# Now, let's train a very small CNN on the MNIST dataset from which we will try to interpret the learned 2D filters.

# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
tf.set_random_seed(42)


# In[ ]:


# Settings
train_path = os.path.join('..', 'input', 'digit-recognizer', 'train.csv')
raw_train_df = pd.read_csv(train_path)

# CNN model settings
size = 28
lr = 0.002
num_classes = 10

# Training settings
# I changed this line
epochs = 50
batch_size = 128


# In[ ]:


# Utils
def parse_train_df(_train_df):
    labels = _train_df.iloc[:,0].values
    imgs = _train_df.iloc[:,1:].values
    imgs_2d = np.array([[[[float(imgs[index][i*28 + j]) / 255] for j in range(28)] for i in range(28)] for index in range(len(imgs))])
    processed_labels = [[0 for _ in range(10)] for i in range(len(labels))]
    for i in range(len(labels)):
        processed_labels[i][labels[i]] = 1
    return np.array(processed_labels), imgs_2d


# In[ ]:


# Data preprocessing
y_train_set, x_train_set = parse_train_df(raw_train_df)

x_train, x_val, y_train, y_val = train_test_split(x_train_set, y_train_set, test_size=0.20, random_state=42)


# One important thing to do here : visualize some input images. 28*28 grayscale images.

# In[ ]:


# Image visualization
n = 5
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x_train[i]+1)/2
    ax.imshow((-x_train[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
plt.show()


# ### CNN model definition
# Here, I will define and train a rather small CNN model. I decided to use 2 Conv2D layers and 1 Dense layer (before the output layer).

# In[ ]:


# CNN model
model = keras.Sequential()

model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(size, size, 1),
                 name='conv_1'
                ))
model.add(Conv2D(12, (3, 3), activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr),
              metrics=['accuracy'])

# Training
training_history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)


# Let's get the weights of the first layer. But first, let's look at the model summary.

# In[ ]:


model.summary()


# In the first layer, we have 60 parameters. Why 60 ?
# 
# The first layer is composed of 6 filters with a (3, 3) shape. So, each filter being composed of 9 parameters, this gives us 6*9=54 weights. The remaining 6 weights are 6 biases, one for each filter output.
# 
# In the next cell, I will fetch the weights of the first layer filters.
# 

# In[ ]:


w = model.layers[0].get_weights()
filters_1_raw = w[0]
biases_1_raw = w[1]

filters_1 = [np.zeros((3, 3)) for _ in range(6)]
for i in range(3):
    for j in range(3):
        for n in range(6):
            filters_1[n][i][j] = filters_1_raw[i][j][0][n]


# In[ ]:


# Let's look at these filters
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes[0][0].imshow(filters_1[0], cmap='Greys_r')
axes[1][0].imshow(filters_1[1], cmap='Greys_r')
axes[2][0].imshow(filters_1[2], cmap='Greys_r')

axes[0][1].imshow(filters_1[3], cmap='Greys_r')
axes[1][1].imshow(filters_1[4], cmap='Greys_r')
axes[2][1].imshow(filters_1[5], cmap='Greys_r')
plt.show()


# In[ ]:


# Here we select an image of the MNIST dataset
mnist_img = x_train[42, :, :, 0]
plt.imshow(mnist_img, cmap='Greys_r')


# Let's apply the first layer filters to our selected image.

# In[ ]:


layer_1_imgs = []
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 10))
for i in range(6):
    img = apply_conv_filter(mnist_img, filters_1[i])
    layer_1_imgs += [img]
    axes[i].imshow(img, cmap='Greys_r')
plt.show()


# The six images above are the output of our CNN first layer (without the effect of the biases). This is what our CNN *sees* after the first layer. It's hard to understand exactly what our model is doing here...
# We just need to understand that we used only 54 parameters to get this visualization. After this, our CNN will create 12 different filters for each one of the above images.

# Now, let's try to visualize some feature maps, for our convolutionnal layers.

# In[ ]:


layer_to_visualize = ['conv_1', 'conv_2']

layer_outputs = [layer.output for layer in model.layers if layer.name in layer_to_visualize]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
intermediate_activations = activation_model.predict(np.expand_dims(x_train[42], axis=0))

n_layer = len(layer_to_visualize)
n_img_per_layer = 6
layer_cpt = 0

fig, axes = plt.subplots(nrows=n_layer, ncols=n_img_per_layer, figsize=(15, 10))

for layer_name, layer_activation in zip(layer_to_visualize, intermediate_activations):
    
    for j in range(n_img_per_layer):
        axes[layer_cpt][j].imshow(layer_activation[0, :, :, j], cmap='Greys_r')
        axes[layer_cpt][j].set_title("{} - map {}".format(layer_name, j))
        
    layer_cpt += 1


# This gives us a visualization of 6 feature maps per layer. The more deep we are, the more the feature map will react to abstract features :
# 
# * First layer : edges, ...
# * Second layer : vertical edges, corners, ...

# Please feel free to ask anything, to comment or to upvote :)
