#!/usr/bin/env python
# coding: utf-8

# # MNIST CNN Classifier

# ## Introduction
# This notebook shows the step by step development of a Convolutional Neural Network (CNN) digit classifier for the MNIST dataset using the Keras API with Tensorflow as a backend.

# ## Importing Packages
# We'll start by importing the packages we need.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from PIL import Image

sns.set(style='white', context='notebook', palette='deep')


# # Loading The Dataset
# Our dataset is divided into 2 subsets in 2 csv files, the training data is in `train.csv`and the test data is in `test.csv`. We'll use pandas read_csv method to read the csv files into dataframes.

# In[ ]:


# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Let's seperate the images from the labels in out training subset.[](http://)

# In[ ]:


# Get labels
Y_train = train["label"]

# Drop 'label' column to get only the images
X_train = train.drop(labels = ["label"],axis = 1).values

# Get test subset values
X_test = test.values

# free some space
del train
del test


# # Data Exploration
# To get a better sense of our dataset, let's explore it a little.
# 
# Let's start by printing the shapes of our X, Y and test data.

# In[ ]:


print('X_train\'s shape: ' + str(X_train.shape))
print('Y_train\'s shape: ' + str(Y_train.shape))
print()
print('X_test\'s shape: ' + str(X_test.shape))


# As we can see we have 42,000 training examples each consisting of a 28x28px image flattened into a 784px vector and its corresponding label (i.e. the handwritten digit in the image).
# Our test data has 28,000 images of the same shape as our training images but without the labels, since we'll be generating the predictions to submit in the [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition.
# 
# Since we're developing a CNN we need our input be a 3D image not a 1D array of pixels. Let's reshape our X_train and X_test to 3D matrices.

# In[ ]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print('X_train\'s shape: ' + str(X_train.shape))
print('X_test\'s shape: ' + str(X_test.shape))


# Let's randomly plot a few of the images and their corresponding labels to better visualize the nature of our dataset.

# In[ ]:


indices = np.random.randint(0, 42000, 16)

plt.figure(figsize=(20,5))
i = 0
for index in indices:
    ax = plt.subplot(2, 8, i + 1)
    ax.set_title('Label: ' + str(Y_train[index]))
    ax.imshow(X_train[index][:,:,0])
    i += 1


# We now have a general idea of the shape of our data, but let's take a look at the number of examples we have for each label to make sure we're not underrepresenting any of the digits.

# In[ ]:


print(Y_train.value_counts())

sns.countplot(Y_train)


# ## Preprocessing
# Before building our model we need to make sure our data is in the right format and apply any sort of processing that can help our model make the best use of our data.
# 
# ### Normalization
# Normalizing our data can lead to noticeable improvements in training our model. Since we're dealing with greyscale images it's sufficient to divide the pixel values with 255 to get a range from 0 to 1. This helps in reducing the effect of illumination differences.

# In[ ]:


# Normalizing the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# ### Label Encoding
# Our Y_train is currently an array of digits from 0 to 9 corresponding to the labels of each of our X_train examples. This will be an issue since we'll be using the  [softmax](https://en.wikipedia.org/wiki/Softmax_function) function which outputs n probability vectors in which the i-th element in the j-th vector corresponds corresponds to the probability of the i-th training example belonging to the j-th class.
# 
# We can slove this by using the to_categorical function from [keras.utils.np_utils](https://keras.io/utils/) to turn our Y_train into 10 one hot vectors that are compaitable with the output of the softmax function.

# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)

print('Y_train\'s shape: ' + str(Y_train.shape))


# ### Validation Set
# To evaluate our model during training and to avoid overfitting  we'll split our data into a 90% training subset and a 10% validation subset. The validation subset will be used to evaluate our model's performance on unseen data at the end of every epoch.

# In[ ]:


# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)


# ## Building The Model
# We'll be using [keras](https://keras.io/), a high-level neural networks API capable of running on top of TensorFlow, to create our CNN.
# 
# Architecture of the network:
# - 2 Conv layers with 32 5x5 filters with Same padding and a ReLU activation followed by a Max Pooling layer of size 2x2
# - 2 Conv layers with 64 3x3 filters with Same padding and a ReLU activation followed by a Max Pooling layer of size 2x2 and a stride of 2
# - A Fully Connected layer with 256 neurons and a ReLU activation with a dropout rate of 0.3
# - A Softmax output layer ouputting our 10 probability vectors

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))


# We'll use the **RMSprop** with the default values as it seems to work well with our model and gives better results than other optimizers like the **adam** optimizer.
# 
# For the loss function we'll be using the **categorical_crossentropy** function since our output categorical format.

# In[ ]:


# Compile the model
model.compile(optimizer='RMSprop' , loss='categorical_crossentropy', metrics=['accuracy'])


# We'll use an annealing learning rate to increase the speed at which our optimizer converges. Using the keras ReduceLROnPlateau callback we'll reduce our learning rate by half if our accuracy doesn't improve for 3 epochs.

# In[ ]:


# learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001)


# A very useful step that helps prevent overfitting is data augmentation. By adding different types of noise randomly to our data we can improve the generalizability of our model making it better equipped to deal with unseen data with unexpected distortions.

# In[ ]:


# Data augmentation to prevent overfitting

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

datagen.fit(X_train)


# ## Fitting The Model
# Now that we've built the model we just need to choose 2 more hyperparameters for the fitting process, the number of epochs and the batch size.
# 
# One epoch is when every example in our dataset is passed forward and backward through the neural network once. Epochs are split into batches of training examples that get trained at the same time.
# 
# We chose 30 epochs as it seems to be the sweet spot between a bearable training time and acceptable performance from our trained model. Increasing the number of epochs could potentially improve the performance depending on the model, but usually it plateaus at some point.
# 
# As for the batch size, the smaller it is the less memory we need to use for each pass since we're using less samples, but that comes at the cost of a less accurate estimation of the gradient. Also a larger batch size results in a faster training process for our network as we need to do fewer passes and updates.
# 
# We chose 100 for our batch size as it's neither too small nor too big and it divides our dataset evenly, avoiding the issue of an incomplete last batch.

# In[ ]:


epochs = 30
batch_size = 100

model.fit_generator(
    datagen.flow(X_train,Y_train, batch_size=batch_size),
    epochs = epochs,
    validation_data = (X_val,Y_val),
    verbose = 1,
    callbacks=[learning_rate_reduction])


# ## Model Evaluation
# Let's check how well our model performes on both the training data and the validation data.

# In[ ]:


(loss_train, accuracy_train) = model.evaluate(X_train, Y_train)
print('Performance on train data:')
print('Loss: ' + str(loss_train))
print('Accuracy: ' + str(accuracy_train*100) + '%')

(loss_val, accuracy_val) = model.evaluate(X_val, Y_val)
print('Performance on validation data:')
print('Loss: ' + str(loss_val))
print('Accuracy: ' + str(accuracy_val*100) + '%')


# ## Predicting The Test Data
# Now that we've trained our model, we can use it to predict our test data.

# In[ ]:


# predict results
results = model.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
# insert the predictions into a pandas series
results = pd.Series(results,name="Label")


# Let's add the ImageId column to comply with the output format and output our prediction to csv file.

# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_predictions_cnn_datagen.csv",index=False)

