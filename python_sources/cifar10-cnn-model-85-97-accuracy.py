#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10 Dataset CNN Model achieving 85.97% accuracy with regularization and data augmentation

# This notebook is the result of a series of experiments I conducted on the CIFAR-10 dataset to understand hyperparameter tuning of a Convolutional Neural Network.  It explains the model with the final parameters that achieved the highest results. This model secures a 85.97% accuracy on unseen test data.
# 
# # CIFAR-10 dataset
# The CIFAR-10 dataset contains 60,000 color images of dimension 32 X 32 in 3 channels divided into 10 classes. The training data has 50,000 images and the test data has 10,000. You can read more about the dataset here: https://www.cs.toronto.edu/~kriz/cifar.html 
# This is a mulyi-label image classification problem with 10 labels. The data is equally split between the labels.

# ### Import the required libraries

# In[ ]:


from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from PIL import Image


# ### Load the dataset from the keras library and split into train and test set
# This is the easiest way to load the CIFAR-10 dataset. You can also download the files from the link in the introduction, but requires a lot more steps to bring it to a usable state. 

# In[ ]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Training set shape:', X_train.shape)
print('Test set shape:', X_test.shape)


# ### Normalize the train and test data
# Converting to float and dividing each instance by 255 so that all the image pixels are between 0 and 1

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# ### One Hot Encode the labels
# 
# The labels are currently vectors stored as a list with 10 values, all are zero except the correct index for that label will be a 1. 
# Example: 
# - Airplane --> [1,0,0,0,0,0,0,0,0,0] 
# - Automobile --> [0,1,0,0,0,0,0,0,0,0]
# - Bird --> [0,0,1,0,0,0,0,0,0,0]
# 
# We want to split them into separate columns by one hot encoding them

# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# ### Splitting the train set for a validation set
# 
# We will further split the training set to create a validation set to test model results on. We want to make sure that we don't touch the test set till we're happy with our model and are ready to make predictions off the test set. 

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


# ### Define the CNN Model
# 
# The layout of this model is similar to AlexNet designed by Alex Krizhevsky but with different number of filters, kernel_size etc. 
# 
# We create a Sequential model and start adding layers one by one. The first Conv2D layers are preceeded by MaxPooling2D and Dropout layer. Then 3 Conv2D layers are stacked followed by again a pooling and dropout layer followed by 2 fully connected Dense layers leading to an output layer. The kernel_size and pool_sie are the same through out the network. 
# 
# The filters double in the size with every layer starting from 128 going up to 512 and coming back down to 256 in the fifth layer. Similar values for neurons were used in the fully connected layer. These settings gave me the best accuracy though it can be computationally expensive. With the help of a GPU on the Kaggle platform, I was able to train this model in approximately an hour.
# I went with the standard activation 'relu' and 'same'padding'
# 
# To stop my model from overfitting, I used the l2 kernel_regularizer and also added dropout layers. This reduced overfitting while also icreasing accuracy by a few percent. I again exprimented with a varity of dropout values to land on this one, using lower dropout of 0.3 for the conv layers and a higher 0.5 for the fully connected layers.

# In[ ]:


def cnn_model():
    
    model = Sequential()
    
    # First Conv layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4), input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # Second Conv layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # Third, fourth, fifth convolution layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # Fully Connected layers
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    return model


# ### Data Augmentation
# 
# Augmenting the training data and introducing random variations of images like rotating them by 15 dgrees, changing width and height etc. made the model generalize better and reduce overfitting while also increasing the accuracy by a bit. It does increase the training time due to the added variations, but is definitely worthit training on a GPU. Don't even think of training this model on a CPU, it will take days.

# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False)

datagen.fit(X_train)


# ### Intitialize the model

# In[ ]:


model = cnn_model()


# ### Compile the model
# Pretty standard settings for the loss, optimizer and metrics functions. I did play around with the learning rate a little bit but the idea was to keep it fairly low to let it slowly converge.

# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer=Adam(lr=0.0003, decay=1e-6),
             metrics=['accuracy'])


# ### Fit the model
# 
# I first tried a lower batch_size of 32 which increase the amount of time for every epoch and also converged to a higher accuracy very slowly. Batch_size of 64 converged the model much faster and also slowly increases the model accuracy by a bit at the end.
# 
# Started with 100 epochs but increased it to 125 as the model was slowly converging still at 100 epochs. Overfitting was not a concern as I had applied strong regluarization. If you look at the output below, the model is beginning to achieve 80% accuracy on the validation set around 30-35 epochs, and the convergence after that is very slow.

# In[ ]:


history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64),
                    steps_per_epoch = len(X_train) // 64, 
                    epochs = 125, 
                    validation_data= (X_valid, y_valid),
                    verbose=1)


# ### Plotting the train and val accuracy and loss

# In[ ]:


pd.DataFrame(history.history).plot()


# ### Evaluating model on the test set

# In[ ]:


scores = model.evaluate(X_test, y_test)


# ### Make predictions

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)
errors = y_pred - y_true != 0


# ### Print Classification Report
# This gives us a breaksdown of scores per label. We can see from the report below that our model has learned classifiying automobiles, ships and truck with a 90% precision and recall, and around 75-90% on all the other categories

# In[ ]:


print(classification_report(y_true, y_pred))


# ### Check the predictions

# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(12,12))
axes = axes.ravel()

for i in np.arange(25):
    axes[i].imshow(X_test[i])
    axes[i].set_title('True: %s \nPredict: %s' % (labels[y_true[i]], labels[y_pred[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# ### Check the wrong predictions

# In[ ]:


fig, axes = plt.subplots(5, 5, figsize=(12,12))
axes = axes.ravel()

miss_pred = np.where(y_pred != y_true)[0]
for i in np.arange(25):
    axes[i].imshow(X_test[miss_pred[i]])
    axes[i].set_title('True: %s \nPredict: %s' % (labels[y_true[miss_pred[i]]], labels[y_pred[miss_pred[i]]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# We can see from the pictures above that our model is perofrming really well, all the misclassified images can very easily be misclassified by a human as well. They are very similar to the mislcassfied prediction. 

# ### Saving the model
# Always save the model and weights so that we can use this trained model and experiment with different parameters to recreate it.

# In[ ]:


model.save('cifar10_cnn.h5')


# In[ ]:




