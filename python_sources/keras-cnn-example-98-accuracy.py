#!/usr/bin/env python
# coding: utf-8

# This notebook is basically an introduction to using keras's CNN for modeling the devanagiri data set.
# [Keras](http://keras.io/) is a wrapper library who's backend can be either Tensorflow, CNTK or Theano. It's very friendly for quick prototyping and testing ideas (of course, provided your model doesn't take days to compile as well :D ) 
# I use the [tensorflow-gpu](https://www.tensorflow.org/install/install_linux#gpu_support) backend for Keras. 
# 
# So let's get down to it.

# # Import relevant libraries

# In[17]:


# Standard useful data processing imports
import random
import numpy as np
import pandas as pd
# Visualisation imports
import matplotlib.pyplot as plt
import seaborn as sns
# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Keras Imports - CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


# Let's have a look at the data now

# In[18]:


# Import the data
dataset = pd.read_csv("../input/data.csv")
print(dataset.head())


# A row-major arrangement of 32 x 32 image data in each row with an additional column being the label at the end. Let's just keep them in suitable containers for further use and clear unwanted memory.

# In[19]:


x = dataset.values[:,:-1] / 255.0
y = dataset['character'].values
# Free memory
del dataset
n_classes = 46 # Number of classes


# Let's have a look at some of the images. Resize and plot the columns(pixels) in a row.

# In[20]:


# Now let's visualise a few random images
img_width_cols = 32
img_height_rows = 32
cutsomcmap = sns.dark_palette("white", as_cmap=True)
random_idxs = random.sample(range(1, len(y)), 4)
plt_dims = (15, 2.5)
f, axarr = plt.subplots(1, 4, figsize=plt_dims)
it = 0
for idx in random_idxs:
    image = x[idx, :].reshape((img_width_cols, img_height_rows)) * 255
    axarr[it].set_title(y[idx])
    axarr[it].axis('off')
    sns.heatmap(data=image.astype(np.uint8), cmap=cutsomcmap, ax=axarr[it])
    it = it+1


# At this points, we can split the data into training and validation (test) - an 80/20 split.
# 
# We can also encode our labels. 
# 
# To do that we first convert the string into Labels.
# 
# So all the "character_XX_YY" labels would be mapped to labels ranging from 0 to 45 ( Because we have 46 classes)
# 
# After that, we prepare this to be used by Keras by using its [to_categorical](https://keras.io/utils/#to_categorical) function.

# In[21]:


# Let's split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


# Now we shall build the CNN! 
# 
# Actually, no.
# 
# We'll have to do one final step of reshaping ALL the data before we go forward to build it.
# 
# Why do we need to reshape the data? Because each data point is in the form of a single row. In this CNN example, we're going to work with IMAGES. 
# 
# The CNN needs images to extract coarse and fine features so that it can train and classify them.
# 
# So a quick step - 

# In[22]:


im_shape = (img_height_rows, img_width_cols, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
x_test = x_test.reshape(x_test.shape[0], *im_shape)


# And NOW we make the model.
# 
# Let's first define a model. Here we shall use the keras [Sequential](https://keras.io/getting-started/sequential-model-guide/) model, which essentially involves us adding a layer one after the other....sequentially. 

# In[23]:


cnn = Sequential()


# The real fun part is defining the LAYERS.
# 
# The first layer, a.k.a the **input layer** requires a bit of attention in terms of the **shape** of the data it will be looking at.
# 
# So just for the first layer, we shall specify the input shape, i.e., the shape of the input image - rows, columns and number of channels.
# 
# Keras also has this neat API that joins the convolutional and activation layers into 1 API call.

# In[24]:


kernelSize = (3, 3)
ip_activation = 'relu'
ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)


# So a common trick here used when developing CNN architectures is to add two Convolution+Activation layers back to back BEFORE we proceed to the pooling layer for downsampling. 
# 
# 
# This is done so that the kernel size used at each layer can be small.
# 
# when multiple convolutional layers are added back to back, the overall effect of the multiple small kernels will be similar to the effect produced by a larger kernel, like having two 3x3 kernels instead of a 7x7 kernel. (Reference link coming soon!)

# In[25]:


# Add the next Convolutional+Activation layer
ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)

# Add the Pooling layer
pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)


# Let's do this again. i.e, One more *ConvAct + ConvAct + Pool* layer sequence. 

# In[26]:


ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)


# Here's an interesting part. Two common problems with most models is that they either underfit, or they overfit. With a simple CNN such as this, there is a high probability that your model would begin overfitting the data. i.e, it relies on the training data too much. 
# 
# There are multiple ways to address the problem of overfitting. This is a pretty neat [**site**](https://towardsdatascience.com/deep-learning-3-more-on-cnns-handling-overfitting-2bd5d99abe5d) which talks about this succintly.
# 
# We have 92000 data points to play with. I'm avoiding augmentation for the same reason.
# 
# For our case, we will use a simple Dropout layer to make sure the network does not depend on the training data too much.

# In[27]:


# Let's deactivate around 20% of neurons randomly for training
drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)


# we are done with the Convolutional layers, and will proceed to send this data to the Fully Connected ANN. To do this, our data must be a 1D vector and not a 2D image. So we **flatten** it.

# In[28]:


flat_layer_0 = Flatten()
cnn.add(Flatten())


# And proceed to the ANN - 

# In[29]:


# Now add the Dense layers
h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)
# Let's add one more before proceeding to the output layer
h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_1)


# Our problem is **classification**. So succeeding the final layer, we use a *softmax* activation function to classify our labels.

# In[30]:


op_activation = 'softmax'
output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)


# Almost done. Now we just need to define the optimizer and loss functions to minimize and compile the CNN

# In[31]:


opt = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)


# And now we train our model! 
# Note : If you have GPU support, set the epochs to 10 and you will be able to reproduce the 98% accuracy claim. Here I've set epochs=2 because of time constraints, and to illustrate what we can achieve.

# In[40]:


history = cnn.fit(x_train, y_train,
                  batch_size=32, epochs=2,
                  validation_data=(x_test, y_test))


# Let's see how well we did by looking at accuracy.

# In[42]:


scores = cnn.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Plot a graph of how the model learnt over its epochs - 

# In[43]:


# Accuracy
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()


# In[44]:


# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# And that would be all! 
# 
# A huge thanks to the Kaggle community for helping me learn!
# 
# And thank you for sticking through till the end! 
