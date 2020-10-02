#!/usr/bin/env python
# coding: utf-8

# ## The Street View House Numbers (SVHN) Dataset
# 
# SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.
# 
# - 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 0.
# - 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
# - Comes in two formats:
#   1. Original images with character level bounding boxes.
#   2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).
# 
# - The dataset that we will be using in this notebook contains 42000 training samples and 18000 testing samples

# Firstly, let's select TensorFlow version 2.x in colab

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow
tensorflow.__version__


# # Initialize the random number generator

# In[ ]:


import random
random.seed(0)


# # Ignore the warnings

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ### Load the dataset

# As we are using google colab, we need to mount the google drive to load the data file

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# Add path to the folder where your dataset is present

# In[ ]:


project_path = '/content/drive/My Drive/'


# Let's load the dataset now

# In[ ]:


import h5py

# Open the file as readonly
h5f = h5py.File(project_path + 'SVHN_single_grey1.h5', 'r')

# Load the training, test and validation se
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
# Close this file
h5f.close()


# ### Print the shape of training and testing data

# In[ ]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ### Let's visualize our dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

columns=10
rows=10

fig=plt.figure(figsize=(8, 8))

for i in range(1,columns*rows+1):
  img=X_train[i]
  fig.add_subplot(rows,columns,i)
  print(y_train[i],end='   ')
  if i % columns == 0:
    print ("")
  plt.imshow(img,cmap='gray')

plt.show()


# ### Resize all the train and test inputs to 28X28, to match with MNIST CNN model's input size
# 
# 

# In[ ]:


# Importing OpenCV module for the resizing function
import cv2
import numpy as np

# Create a resized dataset for training and testing inputs with corresponding size
# Here we are resizing it to 28X28 (same input size as MNIST)
X_train_resized=np.zeros((X_train.shape[0],28,28))
for i in range(X_train.shape[0]):
  #using cv2.resize to resize each train example to 28X28 size using Cubic interpolation
  X_train_resized[i,:,:]=cv2.resize(X_train[i],dsize=(28,28),interpolation=cv2.INTER_CUBIC)

X_test_resized = np.zeros((X_test.shape[0], 28, 28))
for i in range(X_test.shape[0]):
  #using cv2.resize to resize each test example to 28X28 size using Cubic interpolation
  X_test_resized[i,:,:] = cv2.resize(X_test[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
  
# We don't need the original dataset anynmore so we can clear up memory consumed by original dataset
del(X_train, X_test)


# ### Reshape train and test sets into compatible shapes
# - Sequential model in tensorflow.keras expects data to be in the format (n_e, n_h, n_w, n_c)
# - n_e= number of examples, n_h = height, n_w = width, n_c = number of channels
# - do not reshape labels

# In[ ]:


X_train = X_train_resized.reshape(X_train_resized.shape[0], 28, 28, 1)
X_test = X_test_resized.reshape(X_test_resized.shape[0], 28, 28, 1)


# We can delete X_train_resized and X_test_resized variables as we are going to use X_train and X_test variables going further

# In[ ]:


del(X_train_resized, X_test_resized)


# ### Normalize data
# - we must normalize our data as it is always required in neural network models
# - we can achieve this by dividing the RGB codes with 255 (which is the maximum RGB code minus the minimum RGB code)
# - normalize X_train and X_test
# - make sure that the values are float so that we can get decimal points after division

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


# ### Print shape of data and number of images
# - print shape of X_train
# - print number of images in X_train
# - print number of images in X_test

# In[ ]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("Images in X_train:", X_train.shape[0])
print("Images in X_test:", X_test.shape[0])


# ### One-hot encode the class vector
# - convert class vectors (integers) to binary class matrix
# - convert y_train and y_test
# - number of classes: 10
# - we are doing this to use categorical_crossentropy as loss

# In[ ]:


from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)


# In[ ]:


print("Label: ", y_train[2])
plt.imshow(X_train[2].reshape(28,28), cmap='gray')


# ### Building the CNN 
# - Define the layers of model with same size as the CNN used for MNIST Classification

# ### Initialize a sequential model
# - define a sequential model
# - add 2 convolutional layers
#     - no of filters in first layer: 32
#     - no of filters in second layer: 64
#     - kernel size: 3x3
#     - activation: "relu"
#     - input shape: (28, 28, 1) for first layer
# - add a max pooling layer of size 2x2
# - add a dropout layer
#     - dropout layers fight with the overfitting by disregarding some of the neurons while training
#     - use dropout rate 0.2
# - flatten the data
#     - add Flatten later
#     - flatten layers flatten 2D arrays to 1D array before building the fully connected layers
# - add 2 dense layers
#     - number of neurons in first layer: 128
#     - number of neurons in last layer: number of classes
#     - activation function in first layer: relu
#     - activation function in last layer: softmax
#     - we may experiment with any number of neurons for the first Dense layer; however, the final Dense layer must have neurons equal to the number of output classes

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

# Initialize the model
model = Sequential()

# Add a Convolutional Layer with 32 filters of size 3X3 and activation function as 'relu' 
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))

# Add a Convolutional Layer with 64 filters of size 3X3 and activation function as 'relu' 
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))

# Add a MaxPooling Layer of size 2X2 
model.add(MaxPooling2D(pool_size=(2, 2)))

# Apply Dropout with 0.2 probability 
model.add(Dropout(rate=0.2))

# Flatten the layer
model.add(Flatten())

# Add Fully Connected Layer with 128 units and activation function as 'relu'
model.add(Dense(128, activation="relu"))

#Add Fully Connected Layer with 10 units and activation function as 'softmax'
model.add(Dense(10, activation="softmax"))


# ### Make only dense layers trainable
# - freeze the initial convolutional layer weights and train only the dense (FC) layers
# - set trainalble = False for all layers other than Dense layers

# In[ ]:


for l in model.layers:
  print(l.name)


# In[ ]:


for l in model.layers:
  if 'dense' not in l.name:
    l.trainable=False
  if 'dense' in l.name:
    print(l.name + ' should be trained') 


# ### Load pre-trained weights from MNIST CNN model
# - load the file named `cnn_mnist_weights.h5`

# In[ ]:


model.load_weights(project_path + 'cnn_mnist_weights-1.h5')

