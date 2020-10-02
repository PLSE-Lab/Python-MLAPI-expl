#!/usr/bin/env python
# coding: utf-8

# Content:
# * [Introduction](#1):
# * [Import Data Set](#2):
# * [Preparing Images](#3):
# * [Normalize the Data](#4):
# * [Label Encoding](#5):
# * [Implementing with Keras](#6):
# * [Set the Optimizer and Annealer](#7):
# * [Compile Model](#8):
# * [Data Augmentation](#9):
# * [Epochs and Batch Size](#10):
# * [Fit the Model](#11):
# * [Evaluate the model](#12):
# * [Predict Test Data](#13):
# * [Conclusion](#14):

# <a id="1"></a> <br>
# # INTRODUCTION
# * In this kernel, we will be working on Humpback Whale Identification Dataset (Implementing with Keras).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# <a id="2"></a> <br>
# ## Import Dataset

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


# Display the content of data
train.info()


# In[ ]:


# shape gives number of rows and columns in a tuple
train.shape


# In[ ]:


train.Id.describe()


# In[ ]:


train.head(10)


# In[ ]:


train.tail(10)


# In[ ]:


# put labels into y_train variable
y_train = train["Id"]
# Drop 'Id' column
X_train = train.drop(labels = ["Id"], axis = 1)
y_train.head()


# In[ ]:


# Indicates sum of values in our data
train.isnull().sum().sum()


# <a id="3"></a> <br>
# ## Preparing Images

# In[ ]:


from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def prepareImages(train, shape, path):
    
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0
    
    for fig in train['Image']:
        
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+path+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        x_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return x_train


# In[ ]:


x_train = prepareImages(train, train.shape[0], "train")


# <a id="4"></a> <br>
# ### Normalize the Data

# In[ ]:


x_train = x_train / 255.0
print("x_train shape: ",x_train.shape)


# #### Let's look at some samples

# In[ ]:


# Some examples(first one)
plt.imshow(x_train[0][:,:,0], cmap="gray")
plt.title(plt.title(train.iloc[0,0]))
plt.axis("off")
plt.show()


# In[ ]:


# Some examples(last one)
plt.imshow(x_train[25360][:,:,0], cmap="gray")
plt.title(plt.title(train.iloc[25360,0]))
plt.axis("off")
plt.show()


# In[ ]:


# Some examples(55th)
plt.imshow(x_train[55][:,:,0], cmap="gray")
plt.title(plt.title(train.iloc[55,0]))
plt.axis("off")
plt.show()


# <a id="5"></a> <br>
# ## Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[ ]:


y_train = label_encoder.fit_transform(y_train)


# In[ ]:


#let's look at first 10 values
y_train[0:10]  # => new_whale :)


# In[ ]:


y_train.shape


# In[ ]:


# convert to one-hot-encoding(one hot vectors)
# we have 5005 class look at from=> train.Id.describe()

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 5005)


# In[ ]:


#converted
print(y_train.shape)
y_train #let's look at vectors


# <a id="6"></a> <br>
# ## Implementing with Keras

# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (100,100,3)))
model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y_train.shape[1], activation = "softmax"))


# In[ ]:


model.summary()


# <a id="7"></a> <br>
# ### Set the Optimizer and Annealer
# * Adam optimizer: Changes the learning rate during training

# In[ ]:


# Define the optimizer
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)


# In[ ]:


# # Define the optimizer
# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# <a id="8"></a> <br>
# ### Compile Model

# In[ ]:


model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


# <a id="9"></a>
# ### Data Augmentation

# In[ ]:


# if you want to use Data Augmentation,Activate it.


# In[ ]:


# # With data augmentation to prevent overfitting

# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images


# datagen.fit(x_train)


# For the data augmentation, i choosed to :
# 
# * ** Randomly rotate some training images by 10 degrees **
# * ** Randomly zoom by 10% some training images **
# * ** Randomly shift images horizontally by 10% of the width **
# * ** Randomly shift images vertically by 10% of the height **
# 

# <a id="10"></a> <br>
# ### Epochs and Batch Size

# In[ ]:


epochs = 100  # for better result increase the epochs
batch_size = 1000


# <a id="11"></a>
# ### Fit the Model

# In[ ]:


#if you don't want to use data augmentation ,Use this code.
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[learning_rate_reduction])


# ### We get good accuracy, but we need to play with the hyperparameters to get better results from the submission result.

# ### If you want to use data augmentation,Enable below code and disable above code.

# In[ ]:


# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                               epochs=100, verbose = 2, 
#                               steps_per_epoch=x_train.shape[0] // batch_size,
#                               callbacks=[learning_rate_reduction]) 


# ** * we get good accuracy, but we need to play with the hyperparameters to get better results from the submission result. **

# <a id="12"></a> <br>
# ### Evaluate the model
# * Validation and Loss visualization

# In[ ]:


# Plot the loss curve for training
plt.plot(history.history['loss'], color='r', label="Train Loss")
plt.title("Train Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# Plot the accuracy curve for training
plt.plot(history.history['acc'], color='g', label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


print('Train accuracy of the model: ',history.history['acc'][-1])


# In[ ]:


print('Train loss of the model: ',history.history['loss'][-1])


# <a id="13"></a> <br>
# ## Predict Test Data

# In[ ]:


test = os.listdir("../input/test/")
print(len(test))


# In[ ]:


col = ['Image']
test_data = pd.DataFrame(test, columns=col)
test_data['Id'] = ''


# In[ ]:


x_test = prepareImages(test_data, test_data.shape[0], "test")
x_test /= 255


# In[ ]:


predictions = model.predict(np.array(x_test), verbose=1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_data.head(10)
test_data.to_csv('submission_3.csv', index=False)


# ## To be continued...

# <a id="14"></a> <br>
# # Conclusion
# * If you like it, please upvote.
# * If you have any question, I will be appreciate to hear it.
