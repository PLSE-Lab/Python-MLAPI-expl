#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this kernel, we will build a simple Convolutional Neural Network with Keras and explore the effect of different optimizer algorithms on our final prediction. 
# 
# 
#  ![](https://i.stack.imgur.com/HGTQU.gif)

# 1. Load Required Packages
# 2. Load our Data
# 3. Exploration of our Datasets
# 4. Engineering the Data
# 5. Building a simple CNN with Keras
# 6. SGD Algorithm
# 7. RMSprop Algorithm
# 8. Adagrad Algorithm
# 9. Adadelta Algorithm
# 10. Adam Algorithm
# 11. Adamax Algorithm
# 12. Nadam Algorithm
# 13. Final Notes

# # 1. Load Required Packages

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mping
import seaborn as sns
sns.set(style = 'white', context = 'notebook', palette = 'deep')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda
from keras.optimizers import RMSprop, SGD, Adagrad, Adam 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot
from math import pi
from math import cos
from math import floor
from keras.callbacks import Callback
from keras import backend
from numpy import argmax
from subprocess import check_output
from keras.layers import Convolution2D, MaxPooling2D
import keras


# # 2. Load our Data

# In[9]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

## Let's make sure our data is loaded 
print(check_output(["ls", "../input"]).decode("utf8"))


# # 3. Data Exploration

# In[3]:


## Let's see what our training data looks like.
print("Training Shape:", train.shape)
train.head()


# In[4]:


## Let's see what our testing data looks like.

print("Testing Shape: ", test.shape)
test.head()


# In[5]:


## Variable for our pixel data
trainX = (train.iloc[:,1:].values).astype('float32')

## Variable for our targets digits. 
trainY = train.iloc[:,0].values.astype("int32")

## Variable for our testing pixels
testX = test.values.astype('float32')

## For the convinience of our Random Access Memory, let's give him some space.
del train 


# In[6]:


## Let's see what our variables looks like now
trainX


# In[7]:


## Here's our digits category that our model needs to be good at recognizing. 
trainY


# In[10]:


## Let's visualize the count of each value

Y = train["label"]

Y.value_counts()

plot = sns.countplot(Y)


# In[11]:


## Let's visualize some digits

trainX = trainX.reshape(trainX.shape[0], 28, 28)

for i in range(6,9):
    plt.subplot(330 + (i+1))
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    plt.title(trainY[i]);


# # 4. Data Engineering

# Since Keras models are very picky and only expect our data in a specific format, we need to do some changement to our data.
# 
# - Reshaping ---> (Keras expect an additional colour dimension)
# - Standardize our features ---> (Centre the data around the zero mean)
# - One hot encode our target labels ---> [0,0,0,0,0,0,0,0,1,0]
# - Data Split ---> (data to train on - data to test on)
# - Data augmentation ---> Because we love data!

# In[12]:


## Reshape Training
trainX = trainX.reshape(-1,28,28,1)
trainX.shape


# In[13]:


## Reshape Testing
testX = testX.reshape(-1,28,28,1)
testX.shape


# In[14]:


## Standardization

meanX = trainX.mean().astype(np.float32)
std_X = trainX.std().astype(np.float32)

def standardization(x):
    return (x-meanX)/std_X


# In[15]:


## One Hot Encoding

trainY = to_categorical(trainY, num_classes = 10)
classes = trainY.shape[1]
classes


# In[16]:


## Data Split

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.10, random_state=2)


# In[17]:


trainX.shape


# In[18]:


testX.shape


# In[19]:


trainY.shape


# In[20]:


testY.shape


# In[21]:


g = plt.imshow(trainX[0][:,:,0])


# In order to avoid the problem of overfitting, we will artificially increased our initial training data. That way, our model will have the chance to see a greater sets of different training digits and eventually having a better generalization when faced with the testing data. 

# In[22]:


## For the love of data, let's create some more!

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(trainX)


# # 5. Build Model

# 1. Model Architechture ---> Very simple model.
# 2. Loss Function ---> Need to know how our model is performing.
# 3. Optimization Algorithm ---> Need to update our hyperparameters in order to improve our model predictions.
# 4. Training Time!
# 

# We will use a very simple model in order to minimize our training time. 
# 
# Conv2D layers can be seen as an image transformer with the main purpose of learning different features within the images. 
# 
# As the main goal of our network is to take images as input and succesffuly having a small sets of prediction as output, we must have layers performing some sort of downsampling. The MaxPool2D layers filled this purpose with brilliance and also brings a reduction in computational cost. 
# 
# In order to avoid having a network that overfit our training data, we add some dropout. In other words, our network will randomly sets a weight of zero to some nodes during training, forcing our data too see a different "network architechture" at each epoch. 
# 
# Finally to bring our predictions to life, we have a Flatten layers transforming all the features into a 1D vector and Dense layers that do the necessary works to output the probability of each class.

# In[23]:


## Architechture

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[52]:


## Model parameters

epochs = 25 ## We keep it low for minimum training time. Increase if better performance is needed.
batch_size = 86
verbose = 2
step_per_epoch = trainX.shape[0] // batch_size
loss = 'categorical_crossentropy'
learning_rate = 0.1


# Let's now explore different optimization algorithms. A good way to think about the optimization problem is a ball that is stuck in a mountain. Our goal is to find the most efficient way to roll down the ball completely at the bottom. With a very smooth and gentle slope, one might say that it's a relatively easy problem to solve. Unfortunately, in most cases, the slope is extremely noisy and the ball as no compass to know if it is located completely at the bottom or only stuck in a local minimum. Here's below an example showing the kind of environment our ball needs to navigate in order to find the global minimum.

# ![](https://www.mcs.anl.gov/~more/dfo/images/noisy_quadratic_surf.png)

# With the ball analogy in mind, let's explore some algorithms. 

# ## SGD Momentum Optimization

# In regular Stochastic Gradient Descent algorithm, our ball just takes small regular steps down the slope without increasing its speeds. One might say it can be a safe method to make sure sure you make it alive at the bottom, but our goal is to get down there as fast as possible. An alternative strategy is to brings some momentum in our journey. Momentum algorithm cares a great deal about what previous gradients steps were. At each iteration, the ball will subtract the local gradient from the momentum vector (multiplied by the learning rate), and it will move (updating the weights) by simply adding the momentum vector. In other words, the gradient is used as an acceleration, not as a speed. For example, if I have a momentum parameter of 0.9, then the terminal velocity of the ball will be 10 times the gradient times the learning rate, so it will end going 10 times faster than a regular stochastic method. One drawback of this method is that we add one additional hyperparameter to tune in our network.

# In[25]:


Momentum_opti = SGD(lr= learning_rate, momentum = 0.9, nesterov = False)


# ## Nesterov Gradient Optimization

# The idea behind Nesterov gradient descent is to have the ball measuring the gradient of the slope not exactly where the current position is, but slightly ahead in the same direction of the momentum. Overall the iteration steps, those small forward measurements add up and ends up being significantly faster than regular momentum optimization.

# In[26]:


Nesterov_opti = SGD(lr = learning_rate, momentum = 0.9, nesterov = True)


# ## Adagrad Optimization

# With Adagrad, our ball will adjust his speed depending on the gradient of the slope. In other words, it will decay the learning rate but will do so faster for steep dimensions and slower for gentler slopes. This type of optimization will helps our ball to point the resulting updates more directly toward to global minimum. One drawback of Adagrad is that our ball will sometimes end up stopping before reaching his final destination. 

# In[27]:


Adagrad_opti = Adagrad(lr = learning_rate, epsilon = None, decay = 0.9)


# ## RMSProp Optimization

# One alternative in order to solve the problem of early stopping in the previous algorithm is to implement RMSProp optimization. This method resolves the issue by accumulating only the gradients of the most recent iterations instead of all the gradients encounter since the beginning of training. One simple drawback is that we add an additional hyperparameter to our model, but in some cases, it is worthed in order to avoid early stopping. 

# In[28]:


RMSProp_opti = RMSprop(lr = learning_rate, rho = 0.9, decay = 0.9, epsilon = 1e-10)


# ## Adam Optimization

# Adam optimization essentially combines the ideas behind RMSProp and Momentum.

# In[29]:


Adam_opti = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-10, decay = 0.9, amsgrad = False)


# ## Training Time!

# In[53]:


## Change optimizer variable in order to test training with another optimizer.
model.compile(optimizer = Nesterov_opti, loss = loss, metrics = ["accuracy"])


# In[54]:


## Training Time!

history = model.fit_generator(datagen.flow(trainX, trainY, batch_size = batch_size), epochs = epochs, validation_data = (testX, testY), verbose = verbose,
                             steps_per_epoch = step_per_epoch)
                             


# In[55]:


_, train_acc = model.evaluate(trainX, trainY, verbose = 0)


# In[56]:


_, test_acc = model.evaluate(testX, testY, verbose = 0)


# In[57]:


print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# In[61]:


predictions = model.predict_classes(testX, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)


# # 10. Conclusion

# As you probably know, each problem in deep learning is different and require different approaches. With that in mind, feel free to experiment with different optimizers in order to observe the effect on your result. 
# 
# Here's below great ressources I used in order to build this Kernel. 
# 
# * --> https://machinelearningmastery.com/
# * --> https://keras.io/optimizers/
# * --> https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
# 
# As I am still new in the field of deep learning, I am always looking for new information and feedback. Feel free to comment and upvote if you find the kernel useful. 
# 
# Cheers Kagglers friends!
# 
