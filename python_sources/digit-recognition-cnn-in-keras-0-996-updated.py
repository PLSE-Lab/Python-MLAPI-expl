#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# This dataset has 42000 training samples ranging over ten classes (0-9) of hand written digits. The test set consists of 28000 unlabelled samples that need to be labelled from 0-9. The samples are listed as a list of 784 values which can be reshaped into a 28x28 matrix in order to give an image. Such a problem is usually solved using a Convolutional Neural Network (CNN) which preserves the spatial information within images. In a traditional feed forward neural network, an image would have to be flattened i.e. in this case we would use the 784 values along with a bias value as an input to the network. However, in CNNs small squares of information from the 28x28 image scene are convolved with kernels/filters and this information is then used to build a model. Such a structure is robust to distortions such as those caused due to shifts and translations in the scene.
# 
# I tried a shallow baseline Convolutional Neural Network followed by a deeper model to check the level of improvement. The score of > 0.994 is achieved using the larger model structure so the baseline_model function call can be commented if only the main model needs to be run.
# 
# Update: The baseline_model in this kernel is currently commented.
# 
# * Data Preprocessing
#  * Loading the data
#  * Missing values and class imbalance
#  * Visualizing the images
#  * Creating the validation data set
# * Defining the models
#  * Baseline model
#  * Deeper CNN model
#  * Learning rate scheduler
#  * Data augmentation
# * Training the model
# * Evaluating the models
#  * Plotting accuracy and loss
# * Generating the submission file
# 
# 
# 
# 

# Importing the relevant libraries.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint


# ## Data Preprocessing
# Next, the MNIST data is loaded using the Pandas library. It can be seen from the data frames that there are 42000 training examples, 28000 samples in the test data and 784 pixel values. 

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y_train = train['label']
X_train = train.drop(labels = ["label"], axis = 1)

print(X_train.shape)
print(test.shape)
train.head()


# Check for missing values and class imbalance. Using the isnull() function, it can be observed that there are no null or missing values in this dataset because the output array is empty. The value_counts() function can be used to determine the unique counts of a class. This function arranges the counts in descending order so from the output it can be seen that class with digit 1 occurs most frequently with a count of 4684. From the bar plot it can be seen that the frequency with which each digit occurs is fairly consistent.

# In[ ]:


print(np.where(pd.isnull(X_train)))
print(np.where(pd.isnull(y_train)))
print(np.where(pd.isnull(test)))

counts = y_train.value_counts()
print(counts)


# In[ ]:


plt.figure()
sns.barplot(counts.index, counts.values)
plt.title('Frequency of each digit class')
plt.xlabel('Digits')
plt.ylabel('Number of occurrences')
plt.show()


# Reshape the list of pixel values into a 28x28 matrix for use within the CNN and to display images. Images in the MNIST are in gray scale so rather than the three RGB colour channels there is only one channel.

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# Display some of the training images to check what the handwritten characters look like. It can be seen that there is variation in the handwritten digits, e.g. the rotated 1.

# In[ ]:


# display the first four training images to gain an understanding
# of the data
fig, axis = plt.subplots(2,2)
axis[0,0].imshow(X_train[0,:,:,0], cmap = plt.get_cmap('gray'))
# axis[0,0].set_title("First Image")
axis[0,0].axis("off")

axis[0,1].imshow(X_train[1,:,:,0], cmap = plt.get_cmap('gray'))
# axis[0,1].set_title("Second Image")
axis[0,1].axis("off")

axis[1,0].imshow(X_train[2,:,:,0], cmap = plt.get_cmap('gray'))
# axis[1,0].set_title("Third Image")
axis[1,0].axis("off")

axis[1,1].imshow(X_train[3,:,:,0], cmap = plt.get_cmap('gray'))
# axis[1,1].set_title("Fourth Image")
axis[1,1].axis("off")

plt.suptitle("Some training examples")
plt.show()


# Next the training and test data is normalized so that pixels are in between 0 to 1 rather than 0 to 255 and the labels of the training data are one hot encoded. One hot encoding maps numbered classes (0-9) to binary vectors e.g. class with digit 1 will be mapped to [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].

# In[ ]:


X_train.astype('float32')
test.astype('float32')
X_train = X_train / 255.0
test = test / 255.0

# one hot encode the labels of the training data
y_train = to_categorical(y_train)
print(y_train.shape)


# Split the data into training and validation set for fitting a model using the sklearn train_test_split. In order to build a robust model, we need a validation set to test the model's accuracy as it learns information about the data. For this kernel, the data is split into a 90/10 split, i.e., 90% of data is used for testing and 10% data is used for validation. The random seed is also set so that the splits can be reproduced if required.

# In[ ]:


random_seed = 42
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size = 0.10,
                                                  random_state = random_seed)


# ## Creating the model
# A baseline model and a deeper model are defined next and compiled with the categorical cross entropy loss function and adam as the optimizer. Convolutional layers are used along with the relu activation function followed by batch normalization and maxpooling. The baseline model consists of one convolutional layer Conv2D as opposed to four convolutional layers in the deeper CNN. 

# In[ ]:


def baseline_cnn(num_classes=10):
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',input_shape = (28,28,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# In[ ]:


def cnn(num_classes=10):
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',input_shape = (28,28,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.20))


    model.add(Conv2D(filters = 64, kernel_size = (2,2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (2,2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


#  Define the learning rate annealer to reduce the learning rate when training the model.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',
                                            patience = 5,
                                            verbose = 1,
                                            factor = 0.8,
                                            min_lr = 0.00001)


# In order to improve the performance of the CNN, we augment the data using ImageDataGenerator to create additional images when training the model. Some examples of the augmented images are also shown.

# In[ ]:


datagen = ImageDataGenerator(rotation_range = 20,
                             zoom_range = 0.2,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             fill_mode = 'constant',
                             cval = 0.0)
datagen.fit(X_train)
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size = 16):
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_batch[i,:,:,0], cmap = plt.get_cmap('gray'))
        plt.axis('off')
    plt.show()
    break


# ## Training the model
# Next, we can call the baseline model and run it with data augmentation for 100 epochs to evaluate the accuracy. We can then run the deeper CNN and compare the performance of the two models created.

# In[ ]:


batch_size = 86
epochs = 100

# Comment baseline_model and history_baseline if only the larger CNN needs to be run
#baseline_model = baseline_cnn(y_train.shape[1])
#history_baseline = baseline_model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
#                    validation_data=(X_val, y_val),
#                    epochs = epochs,
#                    steps_per_epoch = X_train.shape[0] // batch_size,
#                    callbacks = [learning_rate_reduction],
#                    verbose=2)


# In[ ]:


model = cnn(y_train.shape[1])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                    validation_data=(X_val, y_val),
                    epochs = epochs,
                    steps_per_epoch = X_train.shape[0] // batch_size,
                    callbacks = [learning_rate_reduction],
                    verbose=2)


# ## Evaluating the model
# Plot the accuracy and loss of the models for the 100 epochs.

# In[ ]:


try:
    baseline_model
except NameError: 
    baseline_model = None

if baseline_model is None:
    fig, axis = plt.subplots(1,2,figsize=[10,5])
    axis[0].plot(history.history['acc'])
    axis[0].plot(history.history['val_acc'])
    axis[0].set_title('Model Accuracy')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('accuracy')
    axis[0].legend(['acc','val_acc'], loc='lower right')

    axis[1].plot(history.history['loss'])
    axis[1].plot(history.history['val_loss'])
    axis[1].set_title('Model Loss')
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('accuracy')
    axis[1].legend(['acc','val_acc'], loc='lower right')
else:
    fig, axis = plt.subplots(2,2,figsize=[15,10])
    axis[0,0].plot(history_baseline.history['acc'])
    axis[0,0].plot(history_baseline.history['val_acc'])
    axis[0,0].set_title('Baseline model accuracy')
    axis[0,0].set_xlabel('epoch')
    axis[0,0].set_ylabel('accuracy')
    axis[0,0].legend(['acc','val_acc'], loc='lower right')

    axis[0,1].plot(history_baseline.history['loss'])
    axis[0,1].plot(history_baseline.history['val_loss'])
    axis[0,1].set_title('Baseline model Loss')
    axis[0,1].set_xlabel('epoch')
    axis[0,1].set_ylabel('accuracy')
    axis[0,1].legend(['acc','val_acc'], loc='lower right')

    axis[1,0].plot(history.history['acc'])
    axis[1,0].plot(history.history['val_acc'])
    axis[1,0].set_title('Model Accuracy')
    axis[1,0].set_xlabel('epoch')
    axis[1,0].set_ylabel('accuracy')
    axis[1,0].legend(['acc','val_acc'], loc='lower right')

    axis[1,1].plot(history.history['loss'])
    axis[1,1].plot(history.history['val_loss'])
    axis[1,1].set_title('Model Loss')
    axis[1,1].set_xlabel('epoch')
    axis[1,1].set_ylabel('accuracy')
    axis[1,1].legend(['acc','val_acc'], loc='lower right')

    plt.show()


# ## Submitting the results

# Generate the csv file for submission to Kaggle using the larger CNN model.

# In[ ]:


prediction = model.predict(test)
prediction = np.argmax(prediction, axis =1)
dataframe = pd.read_csv("../input/sample_submission.csv") 
list_of_images = dataframe.ImageId.values
name = 'submission'
submission = pd.DataFrame({'ImageId':list_of_images})
submission['Label'] = prediction
submission.to_csv(f'{name}.csv',index=False)

