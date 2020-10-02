#!/usr/bin/env python
# coding: utf-8

# CNN is tried with Fashion Dataset of MNIST
# 
# 2019 - November

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="1"></a>
# ## Loading the Data Set
# * In this part we load and visualize the data.

# In[ ]:


data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv',dtype = np.float32)
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv',dtype = np.float32)


# In[ ]:


X_train = data_train.iloc[:,1:785]
X_test = data_test.iloc[:,1:785]


# In[ ]:


print('X_train', X_train.shape)
print('X_test', X_test.shape)


# In[ ]:


y_train = data_train.iloc[:,0]
y_test = data_test.iloc[:,0]


# In[ ]:


print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[ ]:


# visualize number of digits classes
plt.figure(figsize=(15,7))
g = sns.countplot(y_train, palette="icefire")
plt.title("Number of fashion classes")
y_train.value_counts()


# Labels
# 
# Each training and test example is assigned to one of the following labels:
# 
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag

# In[ ]:


def labels_of_fashions(argument): 
    switcher = { 
        0: "T-shirt/top", 
        1: "Trouser", 
        2: "Pullover", 
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    } 
    return switcher.get(argument, "nothing") 


# In[ ]:


# plot some samples
def sample_show(t):
    img = X_train.iloc[t].as_matrix()
    img = img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(labels_of_fashions(data_train.iloc[t,0]))
    plt.axis("off")
    plt.show()
    return


# In[ ]:


# plot some sample whichever you want 

sample_show(2600)


# In[ ]:


# plot some sample whichever you want 

sample_show(26001)


# <a id="2"></a>
# ## Normalization, Reshape and Label Encoding 
# * Normalization
#     * We perform a grayscale normalization to reduce the effect of illumination's differences.
#     * If we perform normalization, CNN works faster.
# * Reshape
#     * Train and test images (28 x 28) 
#     * We reshape all data to 28x28x1 3D matrices.
#     * Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel. 
# * Label Encoding  
#     * Encode labels to one hot vectors 
#         * 2 => [0,0,1,0,0,0,0,0,0,0]
#         * 4 => [0,0,0,0,1,0,0,0,0,0]

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0
print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)


# In[ ]:


# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)


# In[ ]:


# Label Encoding 
Y_train = to_categorical(y_train, num_classes = 10)
Y_test = to_categorical(y_test, num_classes = 10)


# In[ ]:


print("Y_train shape: ",Y_train.shape)
print("y_train shape: ",y_train.shape)
print("Y_test shape: ",Y_test.shape)
print("y_test shape: ",y_test.shape)


# <a id="3"></a>
# ## Train Test Split
# * We split the data into train and test sets.
# * test size is 10%.
# * train size is 90%.

# In[ ]:


# Split the train and the validation set for the fitting
# from sklearn.model_selection import train_test_split
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
# print("x_train shape",X_train.shape)
# print("x_test shape",X_val.shape)
# print("y_train shape",Y_train.shape)
# print("y_test shape",Y_val.shape)


# In[ ]:


# *****************
# if we do not want split the data, we can use these codes
# **********
Y_val = Y_test
X_val = X_test
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)


# In[ ]:


# Some examples
plt.imshow(X_train[200][:,:,0],cmap='gray')
plt.show()


# <a id="4"></a>
# ## Convolutional Neural Network 
# * CNN is used for image classification, object detection 
# * <a href="https://ibb.co/kV1j9p"><img src="https://preview.ibb.co/nRkBpp/gec2.jpg" alt="gec2" border="0"></a>

# <a id="5"></a>
# ### What is Convolution Operation?
# * We have some image and feature detector(3*3)
# * Feature detector does not need to be 3 by 3 matrix. It can be 5 by 5 or 7 by 7.
# * Feature detector = kernel = filter
# * Feauture detector detects features like edges or convex shapes. Example, if out input is dog, feature detector can detect features like ear or tail of the dog.
# * feature map = conv(input image, feature detector). Element wise multiplication of matrices.
# * feature map = convolved feature
# * Stride = navigating in input image.
# * We reduce the size of image. This is important bc code runs faster. However, we lost information. 
# * We create multiple feature maps bc we use multiple feature detectors(filters).
# * Lets look at gimp. Edge detect: [0,10,0],[10,-4,10],[0,10,0]
# * <a href="https://imgbb.com/"><img src="https://image.ibb.co/m4FQC9/gec.jpg" alt="gec" border="0"></a>
# * After having convolution layer we use ReLU to break up linearity. Increase nonlinearity. Because images are non linear.
# * <a href="https://ibb.co/mVZih9"><img src="https://preview.ibb.co/gbcQvU/RELU.jpg" alt="RELU" border="0"></a>

# <a id="6"></a>
# ### Same Padding
# * As we keep applying conv layers, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features.
# * input size and output size are same.
# * <a href="https://ibb.co/jUPkUp"><img src="https://preview.ibb.co/noH5Up/padding.jpg" alt="padding" border="0"></a>

# <a id="7"></a>
# ### Max Pooling
# * It makes down-sampling or sub-sampling (Reduces the number of parameters)
# * It makes the detection of features invariant to scale or orientation changes.
# * It reduce the amount of parameters and computation in the network, and hence to also control overfitting. 
# * <a href="https://ibb.co/ckTjN9"><img src="https://preview.ibb.co/gsNYFU/maxpool.jpg" alt="maxpool" border="0"></a>

# <a id="8"></a>
# ### Flattening
# * <a href="https://imgbb.com/"><img src="https://image.ibb.co/c7eVvU/flattenigng.jpg" alt="flattenigng" border="0"></a>

# <a id="9"></a>
# ### Full Connection
# * Neurons in a fully connected layer have connections to all activations in the previous layer
# * Artificial Neural Network
# * <a href="https://ibb.co/hsS14p"><img src="https://preview.ibb.co/evzsAU/fullyc.jpg" alt="fullyc" border="0"></a>

# <a id="10"></a>
# ## Implementing with Keras

# <a id="11"></a>
# ### Create Model
# * conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
# * Dropout: Dropout is a technique where randomly selected neurons are ignored during training
# * <a href="https://ibb.co/jGcvVU"><img src="https://preview.ibb.co/e7yPPp/dropout.jpg" alt="dropout" border="0"></a>

# In[ ]:


# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# <a id="12"></a>
# ### Define Optimizer   
# * Adam optimizer: Change the learning rate

# In[ ]:


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# <a id="13"></a>
# ### Compile Model
# * categorical crossentropy
# * We make binary cross entropy at previous parts and in machine learning tutorial
# * At this time we use categorical crossentropy. That means that we have multi class.
# * <a href="https://ibb.co/jm1bpp"><img src="https://preview.ibb.co/nN3ZaU/cce.jpg" alt="cce" border="0"></a>
# 

# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# <a id="14"></a>
# ### Epochs and Batch Size
# * Say you have a dataset of 10 examples (or samples). You have a **batch size** of 2, and you've specified you want the algorithm to run for 3 **epochs**. Therefore, in each epoch, you have 5 **batches** (10/2 = 5). Each batch gets passed through the algorithm, therefore you have 5 iterations **per epoch**.
# * reference: https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks

# In[ ]:


epochs = 10  # for better result increase the epochs
batch_size = 250


# <a id="15"></a>
# ### Data Augmentation
# * To avoid overfitting problem, we need to expand artificially our handwritten digit dataset
# * Alter the training data with small transformations to reproduce the variations of digit.
# * For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated.
# * <a href="https://ibb.co/k24CUp"><img src="https://preview.ibb.co/nMxXUp/augment.jpg" alt="augment" border="0"></a>

# In[ ]:


# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# <a id="16"></a>
# ### Fit the model

# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)


# <a id="17"></a>
# ### Evaluate the model
# * Test Loss visualization
# * Confusion matrix

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# <a id="18"></a>
# ## Conclusion
# * http://scs.ryerson.ca/~aharley/vis/conv/flat.html
# 

# ## Implementing with Pytorch Library

# Steps of CNN:
# 1. Import Libraries
# 2. Prepare Dataset
#     * Totally same with previous parts.
#     * We use same dataset so we only need train_loader and test_loader.
# 3. Convolutional layer:
#     * Create feature maps with filters(kernels).
#     * Padding: After applying filter, dimensions of original image decreases. However, we want to preserve as much as information about the original image. We can apply padding to increase dimension of feature map after convolutional layer.
#     * We use 2 convolutional layer.
#     * Number of feature map is out_channels = 16
#     * Filter(kernel) size is 5*5
# 4. Pooling layer:
#     * Prepares a condensed feature map from output of convolutional layer(feature map)
#     * 2 pooling layer that we will use max pooling.
#     * Pooling size is 2*2
# 5. Flattening: Flats the features map
# 6. Fully Connected Layer:
#     * Artificial Neural Network that we learnt at previous part.
#     * Or it can be only linear like logistic regression but at the end there is always softmax function.
#     * We will not use activation function in fully connected layer.
#     * You can think that our fully connected layer is logistic regression.
#     * We combine convolutional part and logistic regression to create our CNN model.
# 7. Instantiate Model Class
#     * create model
# 8. Instantiate Loss Class
#     * Cross entropy loss
#     * It also has softmax(logistic function) in it.
# 9. Instantiate Optimizer Class
#     * SGD Optimizer
# 10. Traning the Model
# 11. Prediction

# In[ ]:


# Import Libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


# Prepare Dataset
# load data

# data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv', dtype = np.float32)  # yukarda yapildi


# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = data_train.label.values
features_numpy = data_train.loc[:,data_train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[11].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[11]))
plt.savefig('graph.png')
plt.show()


# In[ ]:


# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

# batch_size, epoch and iteration
batch_size = 100
n_iters = 5000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
# Create ANN
model = CNNModel()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[ ]:


# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))


# In[ ]:


# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()

