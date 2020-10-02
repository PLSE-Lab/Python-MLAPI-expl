#!/usr/bin/env python
# coding: utf-8

# # Hand Digit recognition on MNIST dataset using CNN (keras library)
# 
# * **1. Introduction**
# * **2. Data preparation**
#     * 2.1 Load data
#     * 2.2 Check for null and missing values
#     * 2.3 Normalization
#     * 2.4 Reshape
#     * 2.5 Label encoding
#     * 2.6 Split training and valdiation set
# * **3. CNN**
#     * 3.1 Define the model
#     * 3.2 Set the optimizer
#     * 3.3 Data augmentation
# * **4. Evaluate the model**
#     * 4.1 Training and validation curves
# * **5. Prediction and submition**
#     * 5.1 Predict and Submit results

# # 1. Introduction
# 
# This is a 5 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend) which is very intuitive. Firstly, I will prepare the data (handwritten digits images) then i will focus on the CNN modeling and evaluation.
# 
# This Notebook follows three main parts:
# * The data preparation
# * The CNN modeling and evaluation
# * The results prediction and submission
# 
# 
# 
# 
# <img src="http://img1.imagilive.com/0717/mnist-sample.png" ></img>

# In[ ]:


# Import libraries for data preparation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
np.random.seed(2)


# # 2. Data preparation
# ## 2.1 Load data

# In[ ]:


# loading training data into pandas dataframe object
train_data = pd.read_csv("../input/train.csv")
train_data.head()


# In[ ]:


# loading test data into pandas dataframe object
test_data = pd.read_csv("../input/test.csv")
test_data.head()


# In[ ]:


# Getting training label from training data
Y_train = train_data['label']
X_train = train_data.drop(columns=['label'])
del train_data
print("Shape of training image data"+str(X_train.shape))
print("Shape of training label data"+str(Y_train.shape))


# In[ ]:


#Visualizing the training label
sns.countplot(Y_train)


# Training label is almost equally divided among all the 10 digits.

# ## 2.2 Check for null and missing values

# In[ ]:


# checking missing value in train data
X_train.isnull().any().any()


# In[ ]:


# checking missing value in test data
test_data.isnull().any().any()


# 
# There is no missing values in the train and test dataset. So we can safely go ahead.

# ## 2.3 Normalization

# We perform a grayscale normalization to reduce the effect of illumination's differences. 
# 
# Moreover the CNN converg faster on [0..1] data than on [0..255].

# In[ ]:


# performing grayscale normalization of test and train data
X_train = X_train/255.0
test_data = test_data/255.0 


# ## 2.3 Reshape

# In[ ]:


#reshaping train and test images to 28*28*1 pixels
X_train = X_train.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)


# Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices. 
# 
# Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices. 

# We can get a better sense for one of these examples by visualising the image and looking at the label.

# In[ ]:


# Some examples
index = np.random.randint(42000)
g = plt.imshow(X_train[index][:,:,0])
print("label : "+str(Y_train[index]))


# ## 2.5 Label encoding

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


# Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]).

# ## 2.6 Split training and valdiation set 

# In[ ]:


# Set the random seed
random_seed = 2


# In[ ]:


# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# I choosed to split the train set in two parts : a small fraction (10%) became the validation set which the model is evaluated and the rest (90%) is used to train the model.
# 
# Since we have 42 000 training images of balanced labels (see 2.1 Load data), a random split of the train set doesn't cause some labels to be over represented in the validation set. Be carefull with some unbalanced dataset a simple random split could cause inaccurate evaluation during the validation. 

# # 3. CNN
# ## 3.1 Define the model

# In[ ]:


## Importing keras libraries to model CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# ## CNN architecture
# 1. Conv2D (8 filters with 4*4 convolution) with Relu activation
# 2. MaxPool (2*2 pooling)
# 3. Conv2D (16 filters with 2*2 convolution) with Relu activation
# 4. MaxPool (2*2 pooling)
# 5. Flatten
# 6. Dense NN with Relu activation
# 7. Dense NN with Softmax
# 
# I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.
# 
# The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 8 filters for the first conv2D layers and 16 filters for second one. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.
# 
# The CNN can isolate features that are useful everywhere from these transformed images (feature maps).
# 
# The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important. 
# 
# Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.
# 
# Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting. 
# 
# 'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network. 
# 
# The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.
# 
# In the end, I used the features in two fully-connected (Dense) layers which is just a neural networks (NN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

# In[ ]:


# creating CNN model 
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (4,4), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
model.add(Conv2D(filters = 16, kernel_size = (2,2), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))


# ## 3.2 Set the optimizer
# 
# Once our layers are added to the model, we need to set up a score function, a loss function and an optimisation algorithm.
# 
# We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy".
# 
# The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss. 
# 
# I choosed RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.
# We could also have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than RMSprop.
# 
# The metric function "accuracy" is used is to evaluate the performance our model.
# This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

# In[ ]:


# defining an optimizer for the model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# train the model
batch_size = 100
epochs = 30
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), verbose = 2)


# **Basic CNN result :** The accuracy I have got here on validation set (with batch_size = 100 and epoch = 10) is 99.52. But the validation accuray is 98.71 which is less than training accuray. It means that the model is overfitting the training data. We need to try some methods to reduce overfitting.
# 
# **CNN with dropout:** After applying droput of 0.25 after both pooling layer, I am getting 99.1 accuracy in both training and validation set.  Now it looks pretty descent. But to increase overall training and validation set accuracy, we probably need more training data. Lets try out data augmentation menthod to generate more training data and re-train our model.

# ## 3.3 Data augmentation 

# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more. 
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

# In[ ]:


# datagen = ImageDataGenerator(rotation_range = 20,  # randomly rotate images in the range (degrees, 0 to 180)
#                             zoom_range = 0.2, # Randomly zoom image 
#                             width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
#                             height_shift_range=0.2)
# datagen.fit(X_train)


# For the data augmentation, i choosed to :
#    - Randomly rotate some training images by 10 degrees
#    - Randomly  Zoom by 10% some training images
#    - Randomly shift images horizontally by 10% of the width
#    - Randomly shift images vertically by 10% of the height
#    
# I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.
# 
# Once our model is ready, we fit the training dataset .

# In[ ]:


# Fit the model
# num_iteration = X_train.shape[0]/batch_size
# history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (X_val,Y_val),
#                               verbose = 2, steps_per_epoch = num_iteration)


# # 4. Evaluate the model
# ## 4.1 Training and validation curves

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# # 5. Prediction and submission
# ## 5.1 predict and submit results

# In[ ]:


# predict results
predictions = model.predict(test_data)
predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions, name = "Label")
image_id = pd.Series(range(1,28001),name = "ImageId")

submission = pd.concat([image_id,predictions],axis = 1)
submission.to_csv("submission.csv",index=False)

submission.head()

