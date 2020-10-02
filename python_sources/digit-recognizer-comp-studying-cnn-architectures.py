#!/usr/bin/env python
# coding: utf-8

# # Kaggle competitions: Digit recognizer
# 
# The aim of this notebook is to develop a convolutional neural network (CNN) to recognize handwritten digits. Since CNN's architecture is critical for the model's performance, we will analyze different variations in order to discover which setups better fit to our dataset.
# 
# In order to find the best architecture of the CNN, we will perform different "experiments" to determine which combinations of layers and parameters give better results. This procedure is extensive and requires some hours to finish, but it's completely independent on external results (i.e. other kernels or studies) and it is based on the trial and error process to be followed in any real project in which there are no previous references.
# 
# 
# **TABLE OF CONTENTS**:
# 
# 1. [Load the data](#section1)
# 2. [Data preparation](#section2)
# 3. [Initial CNN model](#section3)
# 4. [Experiment 1. Size of the convolution kernels](#section4)
# 5. [Experiment 2. Number of convolution layers](#section5)
# 6. [Experiment 3. Number of convolution nodes](#section6)
# 7. [Experiment 4. Dropout percentage](#section7)
# 8. [Experiment 5. Dense layer size](#section8)
# 9. [Experiment 6. Data augmentation](#section9)
# 10. [Experiment 7. Batch normalization](#section10)
# 11. [Experiment 8. Replacement of large kernel layers by two smaller ones](#section11)
# 12. [Experiment 9. Replacement of max pooling by convolutions with strides](#section12)
# 13. [Final model and submission](#section13)
# 
# **Disclaimer**: This kernel has been strongly inspired by https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist, which is a very rich and extensive review of an ensemble of 15 CNNs for the digit recognizer competition. I highly recommend to take a look on it for a state of the art review of this competition.

# In[ ]:


# Fundamental libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# General ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import operator
import time

# Neural networks libraries
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# ## Load the data <a id="section1"></a>
# 
# Since digit images are grey, we will deal with a single channel (on the contrary, a coloured image has 3 channels, one for each RGB). Let's look at the data structure:

# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")

print("Original data structure:")
display(train.head())


# It's always a good practice to analyze classes distribution (10 digits, from 0 to 9) in order to ensure if all of them are equally distributed:

# In[ ]:


fig = sns.countplot(train['label'], alpha=0.75).set_title("Digit counts")
plt.xlabel("Digits")
plt.ylabel("Counts")
plt.savefig('digit_counts.png')
plt.show()


# Looks like all classes are pretty well balanced, so that stratified train/test split won't be necessary. 
# 
# Finally, let's check if there are missing values (for example, due to corrupted pixels):

# In[ ]:


train.isna().sum().sort_values(ascending=False)


# ## Data preparation <a id="section2"></a>
# 
# Data is now loaded, and we verified that all classes are more ore less evenly distributed and there are no missing values. Hence, we are ready to prepare our data.
# 
# What we will do:
# * Define the data size: 28x28 pixels
# * Extract target column
# * Normalize values. From 0 to 1, instead of the common 0-255 pixel values
# * Reshape datasets. Take  into account that there is a single channel
# * One hot encode the target column

# In[ ]:


img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, test):
    y = raw["label"]
    x = raw.drop(labels = ["label"],axis = 1) 
    
    x = x/255.
    x = x.values.reshape(-1, img_rows,img_cols,1)
    
    test = test/255.
    test = test.values.reshape(-1,img_rows,img_cols,1)
    
    return x, y, test

X_train, Y_train, X_test = prep_data(train, test)
Y_train = to_categorical(Y_train, num_classes)

print("Data preparation correctly finished")


# ## Initial CNN model <a id="section3"></a>
# 
# Everythin is ready to create our CNN model, and hence obtain our first results. This step serves as a starting point to verify that the previous procedures have been succesfull, and that we can get a reasonably good accuracy. 
# 
# Hence, this initial architecture is very simple and consists on:
# 1. Input layer
# 2. Convolutional layer with 32 filters, 4x4 size and relu activation function
# 3. MaxPool layer
# 4. Dense layer (256)
# 5. Output layer
# 
# Note: CNN uses max pooling to replace output with a max summary to reduce data size and processing time.

# In[ ]:


batch_size = 64

model_1 = Sequential()
model_1.add(Conv2D(filters=16, kernel_size=(4,4),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_1.add(MaxPool2D())
model_1.add(Flatten())
model_1.add(Dense(256, activation='relu'))
model_1.add(Dense(num_classes, activation='softmax'))

print("CNN ready to compile")


# Compile the model and fit to training data:

# In[ ]:


model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history = model_1.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=20,
          validation_split = 0.1)

print("Fitting finished")


# Nice, we got an impressing 98.83% accuracy on the validation set with our simple model. Let's see the its evolution with each epoch:

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_1 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('initial_cnn.png')
plt.show()


# Trainig seems to have reached an accuracy plateaux. Hence, we just need to predict the test dataset, get the maximum probability results and submit the file.

# In[ ]:


# predict results
results = model_1.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step3.csv",index=False)


# Done, we are ready to upload the "submit_step3.csv" and obtain our first official score. 
# 
# Best score at this stage: 0.9883.

# ## Experiment 1. Size of the convolution kernels <a id="section4"></a>
# 
# Our first attempt to predict images with digits has proven to be reasonably successful. However, there's always room for improvement, and the following steps will focus on discovering how the network's architecture impacts the model's accuracy.
# 
# We will start by measuring the model accuracy for different kernel sizes:
# * **Model_kernel_1**. Conv2D (16,3x3,relu) + MaxPool + Dense256 + output
# * **Model_kernel_2**. Conv2D (16,4x4,relu) + MaxPool + Dense256 + output
# * **Model_kernel_3**. Conv2D (16,5x5,relu) + MaxPool + Dense256 + output
# * **Model_kernel_4**. Conv2D (16,6x6,relu) + MaxPool + Dense256 + output
# 
# Create the models:

# In[ ]:


# Model_kernel_1: 3x3
model_kernel_1 = Sequential()
model_kernel_1.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                     activation='relu', 
                     input_shape=(img_rows, img_cols, 1)))
model_kernel_1.add(MaxPool2D(padding='same'))
model_kernel_1.add(Flatten())
model_kernel_1.add(Dense(256, activation='relu'))
model_kernel_1.add(Dense(num_classes, activation='softmax'))
    
# Model_kernel_2: 4x4
model_kernel_2 = Sequential()
model_kernel_2.add(Conv2D(filters=16, kernel_size=(4,4), padding='same',
                     activation='relu', 
                     input_shape=(img_rows, img_cols, 1)))
model_kernel_2.add(MaxPool2D(padding='same'))
model_kernel_2.add(Flatten())
model_kernel_2.add(Dense(256, activation='relu'))
model_kernel_2.add(Dense(num_classes, activation='softmax'))
    
# Model_kernel_3: 5x5
model_kernel_3 = Sequential()
model_kernel_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',
                     activation='relu', 
                     input_shape=(img_rows, img_cols, 1)))
model_kernel_3.add(MaxPool2D(padding='same'))
model_kernel_3.add(Flatten())
model_kernel_3.add(Dense(256, activation='relu'))
model_kernel_3.add(Dense(num_classes, activation='softmax'))

# Model_kernel_4: 6x6
model_kernel_4 = Sequential()
model_kernel_4.add(Conv2D(filters=16, kernel_size=(6,6), padding='same',
                     activation='relu', 
                     input_shape=(img_rows, img_cols, 1)))
model_kernel_4.add(MaxPool2D(padding='same'))
model_kernel_4.add(Flatten())
model_kernel_4.add(Dense(256, activation='relu'))
model_kernel_4.add(Dense(num_classes, activation='softmax')) 


# Then compile the models several times (3 repetitions), so that we can gather some statistics and average the results:

# In[ ]:


ts = time.time()

n_reps = 10
n_epochs = 20

# Keep track of the history evolution for all repetitions of the CNNs
history_kernel_1, history_kernel_val_1 = [0]*n_epochs, [0]*n_epochs
history_kernel_2, history_kernel_val_2 = [0]*n_epochs, [0]*n_epochs
history_kernel_3, history_kernel_val_3 = [0]*n_epochs, [0]*n_epochs
history_kernel_4, history_kernel_val_4 = [0]*n_epochs, [0]*n_epochs


for rep in range(n_reps):

    # Compile model_kernel_1
    model_kernel_1.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    model_kernel_1_history_rep = model_kernel_1.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_split = 0.1, 
            verbose=0)
    history_kernel_1 = tuple(map(operator.add, history_kernel_1, model_kernel_1_history_rep.history['accuracy']))
    history_kernel_val_1 = tuple(map(operator.add, history_kernel_val_1, model_kernel_1_history_rep.history['val_accuracy']))

    # Compile model_kernel_2
    model_kernel_2.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    model_kernel_2_history_rep = model_kernel_2.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_split = 0.1, 
            verbose=0)
    history_kernel_2 = tuple(map(operator.add, history_kernel_2, model_kernel_2_history_rep.history['accuracy']))
    history_kernel_val_2 = tuple(map(operator.add, history_kernel_val_2, model_kernel_2_history_rep.history['val_accuracy']))
    
    # Compile model_kernel_3
    model_kernel_3.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    model_kernel_3_history_rep = model_kernel_3.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_split = 0.1, 
            verbose=0)
    history_kernel_3 = tuple(map(operator.add, history_kernel_3, model_kernel_3_history_rep.history['accuracy']))
    history_kernel_val_3 = tuple(map(operator.add, history_kernel_val_3, model_kernel_3_history_rep.history['val_accuracy']))
    
    # Compile model_kernel_4
    model_kernel_4.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    model_kernel_4_history_rep = model_kernel_4.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_split = 0.1, 
            verbose=0)
    history_kernel_4 = tuple(map(operator.add, history_kernel_4, model_kernel_4_history_rep.history['accuracy']))
    history_kernel_val_4 = tuple(map(operator.add, history_kernel_val_4, model_kernel_4_history_rep.history['val_accuracy']))    
    
# Average historic data for each CNN (train and valuation)
history_kernel_1 = [x/n_reps for x in list(history_kernel_1)] 
history_kernel_2 = [x/n_reps for x in list(history_kernel_2)]
history_kernel_3 = [x/n_reps for x in list(history_kernel_3)]
history_kernel_4 = [x/n_reps for x in list(history_kernel_4)]
history_kernel_val_1 = [x/n_reps for x in list(history_kernel_val_1)]
history_kernel_val_2 = [x/n_reps for x in list(history_kernel_val_2)]
history_kernel_val_3 = [x/n_reps for x in list(history_kernel_val_3)]
history_kernel_val_4 = [x/n_reps for x in list(history_kernel_val_4)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Finally, plot the accuracy results:

# In[ ]:


# Plot the results
plt.plot(history_kernel_val_1)
plt.plot(history_kernel_val_2)
plt.plot(history_kernel_val_3)
plt.plot(history_kernel_val_4)
plt.title('Model accuracy for different convolution kernel sizes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.ylim(0.95,1)
plt.xlim(0,n_epochs)
plt.legend(['3x3', '4x4', '5x5', '6x6'], loc='upper left')
plt.savefig('convolution_kernel_size.png')
plt.show()


# **Conclusion**: Both kernels of 5x5 and 6x6 accomplish the best accuracy. Hence, from now on, our convolution layers will be 5x5 to optimally capture image patterns while still being computatnionally efficient.
# 
# Best score at this stage: 0.9902.

# ## Experiment 2. Number of convolution layers <a id="section5"></a>
# 
# Now that we know the optimal kernel size, let's study how the number of convolution layers affect the model's performance, to see if a larger number increases the accuracy:
# * **Model_layers_1**. Conv2D (32,5x5,relu) + MaxPool + Dense256 + output
# * **Model_layers_2**. Conv2D (16,5x5,relu) + MaxPool + Conv2D(32,5x5,relu) + MaxPool + Dense256 + output
# * **Model_layers_3**. Conv2D (16,5x5,relu) + MaxPool + Conv2D(32,5x5,relu) + MaxPool + Conv2D(64,5x5,relu) + MaxPool + Dense256 + output
# 
# Define the models:

# In[ ]:


# Model_layers_1: 1 Conv2d layer, same as our initial model (model_1) 

# Model_layers_2: 2 Conv2D layers
model_layers_2 = Sequential()
model_layers_2.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_layers_2.add(MaxPool2D(padding='same'))
model_layers_2.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
model_layers_2.add(MaxPool2D(padding='same'))
model_layers_2.add(Flatten())
model_layers_2.add(Dense(256, activation='relu'))
model_layers_2.add(Dense(num_classes, activation='softmax'))

# Model_layers_3: 3 Conv2D layers
model_layers_3 = Sequential()
model_layers_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_layers_3.add(MaxPool2D())
model_layers_3.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
model_layers_3.add(MaxPool2D(padding='same'))
model_layers_3.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
model_layers_3.add(MaxPool2D(padding='same'))
model_layers_3.add(Flatten())
model_layers_3.add(Dense(256, activation='relu'))
model_layers_3.add(Dense(num_classes, activation='softmax'))


# Compile 3 times and get statistics:

# In[ ]:


n_reps = 5
n_epochs = 20

# Keep track of the history evolution for all repetitions of the CNNs
history_layers_1, history_layers_val_1 = [0]*n_epochs, [0]*n_epochs
history_layers_2, history_layers_val_2 = [0]*n_epochs, [0]*n_epochs
history_layers_3, history_layers_val_3 = [0]*n_epochs, [0]*n_epochs

ts = time.time()

for rep in range(n_reps):

    # Compite model_1
    model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    history_layers_1_rep = model_1.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_split = 0.1, 
          verbose=0)
    
    history_layers_1 = tuple(map(operator.add, history_layers_1, history_layers_1_rep.history['accuracy']))
    history_layers_val_1 = tuple(map(operator.add, history_layers_val_1, history_layers_1_rep.history['val_accuracy']))
    

    # Compile model_2
    model_layers_2.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_layers_2_rep = model_layers_2.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_layers_2 = tuple(map(operator.add, history_layers_2, history_layers_2_rep.history['accuracy']))
    history_layers_val_2 = tuple(map(operator.add, history_layers_val_2, history_layers_2_rep.history['val_accuracy']))

    
    # Compile model_3
    model_layers_3.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_layers_3_rep = model_layers_3.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_layers_3 = tuple(map(operator.add, history_layers_3, history_layers_3_rep.history['accuracy']))
    history_layers_val_3 = tuple(map(operator.add, history_layers_val_3, history_layers_3_rep.history['val_accuracy']))
    
# Average historic data for each CNN (train and valuation)
history_layers_1 = [x/n_reps for x in list(history_layers_1)] 
history_layers_2 = [x/n_reps for x in list(history_layers_2)]
history_layers_3 = [x/n_reps for x in list(history_layers_3)]
history_layers_val_1 = [x/n_reps for x in list(history_layers_val_1)]
history_layers_val_2 = [x/n_reps for x in list(history_layers_val_2)]
history_layers_val_3 = [x/n_reps for x in list(history_layers_val_3)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_layers_val_1)
plt.plot(history_layers_val_2)
plt.plot(history_layers_val_3)
plt.title('Model accuracy for different number of Conv layers')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.98,1)
plt.xlim(0,20)
plt.legend(['1 layer', '2 layers', '3 layers'], loc='upper left')
plt.savefig('number_of_layers.png')
plt.show()


# **Conclusion**: The best performance is accomplished with two convolutional layers. Hence, from now on we will will work with this architecture. 
# 
# Best score at this stage: 0.9917.

# ## Experiment 3. Number of convolution nodes <a id="section6"></a>
# 
# It turns out that 2 convolutional layers is the magic number for our dataset, however we haven't modified the number of filters for each layer (beyond  increasing the number of nodes for each succesive layer). Let's analyze this.
# 
# In this section we will review 7 models:
# * **Model_size_1**. Conv2D (8,5x5,relu) + MaxPool + Conv2D (16,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_2**. Conv2D (16,5x5,relu) + MaxPool + Conv2D (32,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_3**. Conv2D (32,5x5,relu) + MaxPool + Conv2D (32,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_4**. Conv2D (24,5x5,relu) + MaxPool + Conv2D (48,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_5**. Conv2D (32,5x5,relu) + MaxPool + Conv2D (64,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_6**. Conv2D (48,5x5,relu) + MaxPool + Conv2D (96,5x5,relu) + MaxPool + Dense256 + output
# * **Model_size_7**. Conv2D (64,5x5,relu) + MaxPool + Conv2D (128,5x5,relu) + MaxPool + Dense256 + output
# 
# Define the models:

# In[ ]:


# Model_size_1: 8-16
model_size_1 = Sequential()
model_size_1.add(Conv2D(filters=16, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_1.add(MaxPool2D())
model_size_1.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model_size_1.add(MaxPool2D(padding='same'))
model_size_1.add(Flatten())
model_size_1.add(Dense(256, activation='relu'))
model_size_1.add(Dense(num_classes, activation='softmax'))

# Model_size_2: 16-32
model_size_2 = Sequential()
model_size_2.add(Conv2D(filters=16, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_2.add(MaxPool2D())
model_size_2.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model_size_2.add(MaxPool2D(padding='same'))
model_size_2.add(Flatten())
model_size_2.add(Dense(256, activation='relu'))
model_size_2.add(Dense(num_classes, activation='softmax'))

# Model_size_3: 32-32
model_size_3 = Sequential()
model_size_3.add(Conv2D(filters=16, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_3.add(MaxPool2D())
model_size_3.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model_size_3.add(MaxPool2D(padding='same'))
model_size_3.add(Flatten())
model_size_3.add(Dense(256, activation='relu'))
model_size_3.add(Dense(num_classes, activation='softmax'))

# Model_size_4: 24-48
model_size_4 = Sequential()
model_size_4.add(Conv2D(filters=24, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_4.add(MaxPool2D())
model_size_4.add(Conv2D(filters=48, kernel_size=(5,5), activation='relu'))
model_size_4.add(MaxPool2D(padding='same'))
model_size_4.add(Flatten())
model_size_4.add(Dense(256, activation='relu'))
model_size_4.add(Dense(num_classes, activation='softmax'))

# Model_size_5: 32-64
model_size_5 = Sequential()
model_size_5.add(Conv2D(filters=32, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_5.add(MaxPool2D())
model_size_5.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model_size_5.add(MaxPool2D(padding='same'))
model_size_5.add(Flatten())
model_size_5.add(Dense(256, activation='relu'))
model_size_5.add(Dense(num_classes, activation='softmax'))

# Model_size_6: 48-96
model_size_6 = Sequential()
model_size_6.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_size_6.add(MaxPool2D())
model_size_6.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_size_6.add(MaxPool2D(padding='same'))
model_size_6.add(Flatten())
model_size_6.add(Dense(256, activation='relu'))
model_size_6.add(Dense(num_classes, activation='softmax'))


# Compile 3 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 3
n_epochs = 20

# Keep track of the history evolution for all repetitions of the CNNs
history_size_1, history_size_val_1 = [0]*n_epochs, [0]*n_epochs
history_size_2, history_size_val_2 = [0]*n_epochs, [0]*n_epochs
history_size_3, history_size_val_3 = [0]*n_epochs, [0]*n_epochs
history_size_4, history_size_val_4 = [0]*n_epochs, [0]*n_epochs
history_size_5, history_size_val_5 = [0]*n_epochs, [0]*n_epochs
history_size_6, history_size_val_6 = [0]*n_epochs, [0]*n_epochs


for rep in range(n_reps):

    # Compite model_1
    model_size_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    history_size_1_rep = model_size_1.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_split = 0.1, 
          verbose=0)
    
    history_size_1 = tuple(map(operator.add, history_size_1, history_size_1_rep.history['accuracy']))
    history_size_val_1 = tuple(map(operator.add, history_size_val_1, history_size_1_rep.history['val_accuracy']))
    

    # Compile model_2
    model_size_2.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_size_2_rep = model_size_2.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_size_2 = tuple(map(operator.add, history_size_2, history_size_2_rep.history['accuracy']))
    history_size_val_2 = tuple(map(operator.add, history_size_val_2, history_size_2_rep.history['val_accuracy']))

    
    # Compile model_3
    model_size_3.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_size_3_rep = model_size_3.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_size_3 = tuple(map(operator.add, history_size_3, history_size_3_rep.history['accuracy']))
    history_size_val_3 = tuple(map(operator.add, history_size_val_3, history_size_3_rep.history['val_accuracy']))
    
    # Compile model_4
    model_size_4.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_size_4_rep = model_size_4.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_size_4 = tuple(map(operator.add, history_size_4, history_size_4_rep.history['accuracy']))
    history_size_val_4 = tuple(map(operator.add, history_size_val_4, history_size_4_rep.history['val_accuracy']))
    
    # Compile model_5
    model_size_5.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_size_5_rep = model_size_5.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_size_5 = tuple(map(operator.add, history_size_5, history_size_5_rep.history['accuracy']))
    history_size_val_5 = tuple(map(operator.add, history_size_val_5, history_size_5_rep.history['val_accuracy']))
    
    # Compile model_6
    model_size_6.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_size_6_rep = model_size_6.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_size_6 = tuple(map(operator.add, history_size_6, history_size_6_rep.history['accuracy']))
    history_size_val_6 = tuple(map(operator.add, history_size_val_6, history_size_6_rep.history['val_accuracy']))
    
    
# Average historic data for each CNN (train and valuation)
history_size_1 = [x/n_reps for x in list(history_size_1)] 
history_size_2 = [x/n_reps for x in list(history_size_2)]
history_size_3 = [x/n_reps for x in list(history_size_3)]
history_size_4 = [x/n_reps for x in list(history_size_4)] 
history_size_5 = [x/n_reps for x in list(history_size_5)]
history_size_6 = [x/n_reps for x in list(history_size_6)]
history_size_val_1 = [x/n_reps for x in list(history_size_val_1)]
history_size_val_2 = [x/n_reps for x in list(history_size_val_2)]
history_size_val_3 = [x/n_reps for x in list(history_size_val_3)]
history_size_val_4 = [x/n_reps for x in list(history_size_val_4)]
history_size_val_5 = [x/n_reps for x in list(history_size_val_5)]
history_size_val_6 = [x/n_reps for x in list(history_size_val_6)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_size_val_1)
plt.plot(history_size_val_2)
plt.plot(history_size_val_3)
plt.plot(history_size_val_4)
plt.plot(history_size_val_5)
plt.plot(history_size_val_6)
plt.title('Model accuracy for different Conv sizes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.98,1)
plt.xlim(0,n_epochs)
plt.legend(['8-16', '16-32', '32-32', '24-48', '32-64', '48-96', '64,128'], loc='upper left')
plt.savefig('convolution_size.png')
plt.show()


# **Conclusion**: A combination of 48 and 96 nodes give the best performance for our CNN model. 
# 
# Best score at this stage: 0.9938.

# Submit best results:

# In[ ]:


# predict results
results = model_size_6.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step6.csv",index=False)


# ## Experiment 4. Dropout percentage <a id="section7"></a>
# 
# Neural networks may focus on certain paths between layers, hence being more prone to overfitting. One way to deal with this is to switch off some nodes randomly, so that particular paths are not prefered. In this section, we will include droput layers after each Conv2D and analyze the effect of the droput percentage.
# 
# Droput poercentages:
# * **Model_dropout_1**. Conv2D (48,5x5,relu) + MaxPool + Conv2D (96,5x5,relu) + MaxPool + Dense256 + output
# * **Model_dropout_2**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.2) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.2) + Dense256 + output
# * **Model_dropout_3**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense256 + output
# * **Model_dropout_4**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.6) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.6) + Dense256 + output
# * **Model_dropout_5**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.8) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.8) + Dense256 + output
# 
# Define models:

# In[ ]:


# Model_dropout_1: No dropout, same as model_size_6

# Model_dropout_2: 20% dropout
model_dropout_2 = Sequential()
model_dropout_2.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dropout_2.add(MaxPool2D())
model_dropout_2.add(Dropout(0.2))
model_dropout_2.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dropout_2.add(MaxPool2D(padding='same'))
model_dropout_2.add(Dropout(0.2))
model_dropout_2.add(Flatten())
model_dropout_2.add(Dense(256, activation='relu'))
model_dropout_2.add(Dense(num_classes, activation='softmax'))

# Model_dropout_3: 40% dropout
model_dropout_3 = Sequential()
model_dropout_3.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dropout_3.add(MaxPool2D())
model_dropout_3.add(Dropout(0.4))
model_dropout_3.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dropout_3.add(MaxPool2D(padding='same'))
model_dropout_3.add(Dropout(0.4))
model_dropout_3.add(Flatten())
model_dropout_3.add(Dense(256, activation='relu'))
model_dropout_3.add(Dense(num_classes, activation='softmax'))

# Model_dropout_4: 60% dropout
model_dropout_4 = Sequential()
model_dropout_4.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dropout_4.add(MaxPool2D())
model_dropout_4.add(Dropout(0.6))
model_dropout_4.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dropout_4.add(MaxPool2D(padding='same'))
model_dropout_4.add(Dropout(0.6))
model_dropout_4.add(Flatten())
model_dropout_4.add(Dense(256, activation='relu'))
model_dropout_4.add(Dense(num_classes, activation='softmax'))

# Model_dropout_5: 80% dropout
model_dropout_5 = Sequential()
model_dropout_5.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dropout_5.add(MaxPool2D())
model_dropout_5.add(Dropout(0.8))
model_dropout_5.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dropout_5.add(MaxPool2D(padding='same'))
model_dropout_5.add(Dropout(0.8))
model_dropout_5.add(Flatten())
model_dropout_5.add(Dense(256, activation='relu'))
model_dropout_5.add(Dense(num_classes, activation='softmax'))


# Compile 3 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 3
n_epochs = 20

# Keep track of the history evolution for all repetitions of the CNNs
history_dropout_1, history_dropout_val_1 = [0]*n_epochs, [0]*n_epochs
history_dropout_2, history_dropout_val_2 = [0]*n_epochs, [0]*n_epochs
history_dropout_3, history_dropout_val_3 = [0]*n_epochs, [0]*n_epochs
history_dropout_4, history_dropout_val_4 = [0]*n_epochs, [0]*n_epochs
history_dropout_5, history_dropout_val_5 = [0]*n_epochs, [0]*n_epochs


for rep in range(n_reps):

    # Model_1 was previously computed in Step 6

    # Compile model_2
    model_dropout_2.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dropout_2_rep = model_dropout_2.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dropout_2 = tuple(map(operator.add, history_dropout_2, history_dropout_2_rep.history['accuracy']))
    history_dropout_val_2 = tuple(map(operator.add, history_dropout_val_2, history_dropout_2_rep.history['val_accuracy']))

    
    # Compile model_3
    model_dropout_3.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dropout_3_rep = model_dropout_3.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dropout_3 = tuple(map(operator.add, history_dropout_3, history_dropout_3_rep.history['accuracy']))
    history_dropout_val_3 = tuple(map(operator.add, history_dropout_val_3, history_dropout_3_rep.history['val_accuracy']))
    
    # Compile model_4
    model_dropout_4.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dropout_4_rep = model_dropout_4.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dropout_4 = tuple(map(operator.add, history_dropout_4, history_dropout_4_rep.history['accuracy']))
    history_dropout_val_4 = tuple(map(operator.add, history_dropout_val_4, history_dropout_4_rep.history['val_accuracy']))
    
    # Compile model_5
    model_dropout_5.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dropout_5_rep = model_dropout_5.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dropout_5 = tuple(map(operator.add, history_dropout_5, history_dropout_5_rep.history['accuracy']))
    history_dropout_val_5 = tuple(map(operator.add, history_dropout_val_5, history_dropout_5_rep.history['val_accuracy']))
       
    
# Average historic data for each CNN (train and valuation)
history_dropout_2 = [x/n_reps for x in list(history_dropout_2)]
history_dropout_3 = [x/n_reps for x in list(history_dropout_3)]
history_dropout_4 = [x/n_reps for x in list(history_dropout_4)] 
history_dropout_5 = [x/n_reps for x in list(history_dropout_5)]
history_dropout_val_2 = [x/n_reps for x in list(history_dropout_val_2)]
history_dropout_val_3 = [x/n_reps for x in list(history_dropout_val_3)]
history_dropout_val_4 = [x/n_reps for x in list(history_dropout_val_4)]
history_dropout_val_5 = [x/n_reps for x in list(history_dropout_val_5)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_size_val_1)
plt.plot(history_dropout_val_2)
plt.plot(history_dropout_val_3)
plt.plot(history_dropout_val_4)
plt.plot(history_dropout_val_5)
plt.title('Model accuracy for different dropouts')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.98,1)
plt.xlim(0,n_epochs)
plt.legend(['0% dropout', '20% dropout', '40% dropout', '60% dropout', '80% dropout'], loc='upper left')
plt.savefig('dropout.png')
plt.show()


# **Conclusion**: Both a 40% and a 60% dropout present the higher accuracies. I choose to use a final 40% dropout.
# 
# Best score at this stage: 0.9939

# In[ ]:


# predict results
results = model_dropout_3.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step7.csv",index=False)


# ## Experiment 5. Dense layer size <a id="section8"></a>
# 
# At this point we have modified all but the size of dense layers. Hence, let's complete the analysis by studying different nodes on them.
# 
# Dense layers models:
# * **Model_dense_1**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense64 + output
# * **Model_dense_2**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense128 + output
# * **Model_dense_3**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense256 + output
# * **Model_dense_4**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense512 + output
# * **Model_dense_5**. Conv2D (48,5x5,relu) + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + MaxPool + Dropout(0.4) + Dense1024 + output
# 
# Define the models,

# In[ ]:


# Model_dense_64: 20% dropout
model_dense_1 = Sequential()
model_dense_1.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dense_1.add(MaxPool2D())
model_dense_1.add(Dropout(0.4))
model_dense_1.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dense_1.add(MaxPool2D(padding='same'))
model_dense_1.add(Dropout(0.4))
model_dense_1.add(Flatten())
model_dense_1.add(Dense(64, activation='relu'))
model_dense_1.add(Dense(num_classes, activation='softmax'))

# Model_dense_128: 128 nodes dense layer
model_dense_2 = Sequential()
model_dense_2.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dense_2.add(MaxPool2D())
model_dense_2.add(Dropout(0.4))
model_dense_2.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dense_2.add(MaxPool2D(padding='same'))
model_dense_2.add(Dropout(0.4))
model_dense_2.add(Flatten())
model_dense_2.add(Dense(128, activation='relu'))
model_dense_2.add(Dense(num_classes, activation='softmax'))

# Model_dense_3: 256 nodes dense layer. Same as model from Step 7.

# Model_dense_4: 512 nodes dense layer
model_dense_4 = Sequential()
model_dense_4.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dense_4.add(MaxPool2D())
model_dense_4.add(Dropout(0.4))
model_dense_4.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dense_4.add(MaxPool2D(padding='same'))
model_dense_4.add(Dropout(0.4))
model_dense_4.add(Flatten())
model_dense_4.add(Dense(512, activation='relu'))
model_dense_4.add(Dense(num_classes, activation='softmax'))

# Model_dense_5: 1024 nodes dense layer
model_dense_5 = Sequential()
model_dense_5.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_dense_5.add(MaxPool2D())
model_dense_5.add(Dropout(0.4))
model_dense_5.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_dense_5.add(MaxPool2D(padding='same'))
model_dense_5.add(Dropout(0.4))
model_dense_5.add(Flatten())
model_dense_5.add(Dense(1024, activation='relu'))
model_dense_5.add(Dense(num_classes, activation='softmax'))


# Compile 3 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 3
n_epochs = 20

# Keep track of the history evolution for all repetitions of the CNNs
history_dense_1, history_dense_val_1 = [0]*n_epochs, [0]*n_epochs
history_dense_2, history_dense_val_2 = [0]*n_epochs, [0]*n_epochs
history_dense_4, history_dense_val_4 = [0]*n_epochs, [0]*n_epochs
history_dense_5, history_dense_val_5 = [0]*n_epochs, [0]*n_epochs


for rep in range(n_reps):

    # Compile model_dense_1
    model_dense_1.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dense_1_rep = model_dense_1.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dense_1 = tuple(map(operator.add, history_dense_1, history_dense_1_rep.history['accuracy']))
    history_dense_val_1 = tuple(map(operator.add, history_dense_val_1, history_dense_1_rep.history['val_accuracy']))

    # Compile model_dense_2
    model_dense_2.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dense_2_rep = model_dense_2.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dense_2 = tuple(map(operator.add, history_dense_2, history_dense_2_rep.history['accuracy']))
    history_dense_val_2 = tuple(map(operator.add, history_dense_val_2, history_dense_2_rep.history['val_accuracy']))
    
    # Model with 256 dense nodes was compiled in Step 7.
    
    # Compile model_dense_4
    model_dense_4.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dense_4_rep = model_dense_4.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dense_4 = tuple(map(operator.add, history_dense_4, history_dense_4_rep.history['accuracy']))
    history_dense_val_4 = tuple(map(operator.add, history_dense_val_4, history_dense_4_rep.history['val_accuracy']))
    
    # Compile model_dense_5
    model_dense_5.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history_dense_5_rep = model_dense_5.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_split = 0.1, 
              verbose=0)
    
    history_dense_5 = tuple(map(operator.add, history_dense_5, history_dense_5_rep.history['accuracy']))
    history_dense_val_5 = tuple(map(operator.add, history_dense_val_5, history_dense_5_rep.history['val_accuracy']))
       
    
# Average historic data for each CNN (train and valuation)
history_dense_1 = [x/n_reps for x in list(history_dense_2)]
history_dense_2 = [x/n_reps for x in list(history_dense_2)]
history_dense_4 = [x/n_reps for x in list(history_dense_4)] 
history_dense_5 = [x/n_reps for x in list(history_dense_5)]
history_dense_val_1 = [x/n_reps for x in list(history_dense_val_2)]
history_dense_val_2 = [x/n_reps for x in list(history_dense_val_2)]
history_dense_val_4 = [x/n_reps for x in list(history_dense_val_4)]
history_dense_val_5 = [x/n_reps for x in list(history_dense_val_5)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_dense_val_1)
plt.plot(history_dense_val_2)
plt.plot(history_dropout_val_3)
plt.plot(history_dense_val_4)
plt.plot(history_dense_val_5)
plt.title('Model accuracy for different number of dense nodes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.99,1)
plt.xlim(0,n_epochs)
plt.legend(['64 dense nodes', '128 dense nodes', '256 dense nodes', '512 dense nodes', '1024 dense nodes'], loc='upper left')
plt.savefig('dense_nodes.png')
plt.show()


# **Conclusion**: All models present very similar results. Hence, we will use a 128 nodes dense layer (high performance yet not the most computational consuming option).
# 
# Best score at this stage: 0.9930 (which is lower than before, but this is due to instabilities)

# In[ ]:


# predict results
results = model_dropout_3.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step8.csv",index=False)


# With this, we have covered all the basic modifications for the CNN architecture, accomplishing a reasonably high score. However, there are techniques (a bit more advanced than just tunning our CNN parameters) that may improve the prediction score of the model. The following steps are designed to be the key difference to obtain more competitive results.

# ## Experiment 6. Data augmentation <a id="section9"></a>
# 
# This technique increases the total number of images to train the model. The general idea is to generate slightly modified versions of the original images by rotating, zooming or shifting them. This modified images help the model to generalize patterns, so that it performs better in the test dataset.
# 
# See the nice Data Augmentation topic from the Deep Learning course for more details: https://www.kaggle.com/dansbecker/data-augmentation.
# 
# First, we generate the augmented images:

# In[ ]:


X_train_validation, X_val_validation, Y_train_validation, Y_val_validation = train_test_split(X_train, Y_train, test_size = 0.2)

# Generate augmented additional data
data_generator_with_aug = ImageDataGenerator(width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 10,
                                   zoom_range = 0.1)
data_generator_no_aug = ImageDataGenerator()

train_generator = data_generator_with_aug.flow(X_train_validation, Y_train_validation, batch_size=64)
validation_generator = data_generator_no_aug.flow(X_train_validation, Y_train_validation, batch_size=64)


# In[ ]:


# Model for augmented data (same as dropout_3)
model_augmentation = Sequential()
model_augmentation.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_augmentation.add(MaxPool2D())
model_augmentation.add(Dropout(0.4))
model_augmentation.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_augmentation.add(MaxPool2D(padding='same'))
model_augmentation.add(Dropout(0.4))
model_augmentation.add(Flatten())
model_augmentation.add(Dense(256, activation='relu'))
model_augmentation.add(Dense(num_classes, activation='softmax'))


# Compile 10 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 10
n_epochs = 20

# Use the model with better score and include augmented data. Repeat n_reps times for averaging
history_augmentation, history_augmentation_val = [0]*n_epochs, [0]*n_epochs

for rep in range(n_reps):
    # Compile the model
    model_augmentation.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Fit the model
    history_augmentation_rep = model_augmentation.fit_generator(train_generator,
                                                         epochs = n_epochs, 
                                                         steps_per_epoch = X_train_validation.shape[0]//64,
                                                         validation_data = validation_generator,  
                                                         verbose=0)
    history_augmentation = tuple(map(operator.add, history_augmentation, history_augmentation_rep.history['accuracy']))
    history_augmentation_val = tuple(map(operator.add, history_augmentation_val, history_augmentation_rep.history['val_accuracy']))

history_augmentation = [x/n_reps for x in list(history_augmentation)]
history_augmentation_val = [x/n_reps for x in list(history_augmentation_val)]  
    
print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_augmentation_val)
plt.plot(history_dropout_val_3)
plt.title('Model accuracy for data augmentation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.99,1)
plt.xlim(0,n_epochs)
plt.legend(['with augmentation', 'without augmentation'], loc='upper left')
plt.savefig('augmentation.png')
plt.show()


# **Conclusion**: as expected, data augmentation has helped the CNN to generalize patterns and recognise more digits. The performance has significantly improved.
# 

# In[ ]:


# predict results
results = model_dropout_3.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step9.csv",index=False)


# ## Experiment 7. Batch normalization <a id="section10"></a>
# 
# Batch normalization is a technique to improve the performance, speed and stability of neural networks. It essentially normalises the inputs of a layer by scaling the activations. See https://arxiv.org/pdf/1502.03167v3.pdf for in depth details.
# 
# Batch normalization model:
# * **Model_batch_norm**. Conv2D (48,5x5,relu) + BatchNorm + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + BatchNorm + MaxPool + Dropout(0.4) + Dense64 + BatchNorm + output
# 
# Let's define the model adding batch normalization after each convolution or dense layer (except the input/output layers):

# In[ ]:


# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer
model_batch_norm = Sequential()
model_batch_norm.add(Conv2D(filters=48, kernel_size=(5,5),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_batch_norm.add(BatchNormalization())
model_batch_norm.add(MaxPool2D())
model_batch_norm.add(Dropout(0.4))
model_batch_norm.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))
model_batch_norm.add(BatchNormalization())
model_batch_norm.add(MaxPool2D(padding='same'))
model_batch_norm.add(Dropout(0.4))
model_batch_norm.add(Flatten())
model_batch_norm.add(Dense(256, activation='relu'))
model_batch_norm.add(BatchNormalization())
model_batch_norm.add(Dense(num_classes, activation='softmax'))


# Compile 10 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 10
n_epochs = 20

# Use the model with better score and include augmented data. Repeat n_reps times for averaging
history_batch_norm, history_batch_norm_val = [0]*n_epochs, [0]*n_epochs

for rep in range(n_reps):
    # Compile the model
    model_batch_norm.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Fit the model
    history_batch_norm_rep = model_batch_norm.fit_generator(train_generator,
                                                         epochs = n_epochs, 
                                                         steps_per_epoch = X_train_validation.shape[0]//64,
                                                         validation_data = validation_generator,  
                                                         verbose=0)
    history_batch_norm = tuple(map(operator.add, history_batch_norm, history_batch_norm_rep.history['accuracy']))
    history_batch_norm_val = tuple(map(operator.add, history_batch_norm_val, history_batch_norm_rep.history['val_accuracy']))
    
history_batch_norm = [x/n_reps for x in list(history_batch_norm)]
history_batch_norm_val = [x/n_reps for x in list(history_batch_norm_val)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_batch_norm_val)
plt.plot(history_augmentation_val)
plt.title('Model accuracy for batch normalization')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.99,1)
plt.xlim(0,n_epochs)
plt.legend(['with batch normalization', 'without batch normalization'], loc='upper left')
plt.savefig('batch_normalization.png')
plt.show()


# **Conclusion**: the obtained results are somewhat confusing. In one hand, batch normalization seems to increase the instability of the results. This should be studied more exhaustively with other datasets and I let this as a *to do* task for myself, but please feel free to add some comments about this in the kernel discussion. On the other hand, the accuracy curve seems to reach a plateaux for large epochs when batch normalization is not applied, but there's an increasing tendency in the other case. My final decision has been to keep batch normalization, since we will increase the model's complexity in the following steps and this method has demonstrated to enhance CNNs efficiency in numerous studies.

# In[ ]:


# predict results
results = model_batch_norm.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step10.csv",index=False)


# ## Experiment 8. Replacement of large kernel layers by two smaller ones <a id="section10"></a>
# 
# There is evidence pointing out that convolution layers with large kernel sizes can be replaced by two (or more) convolution layers with smaller kernel size. This seems to speed up and improve the performance of CNN for computer vision, since two convolutions are able to better detect  non-linearities in data. See https://arxiv.org/pdf/1512.00567v1.pdf for an in depth study of this technique.
# 
# Replace large kernels model:
# * **Model_batch_norm**. Conv2D (48,3x3,relu) + BatchNorm + Conv2D (48,3x3,relu) + BatchNorm + MaxPool + Dropout(0.4) + Conv2D (96,5x5,relu) + BatchNorm + Conv2D (96,5x5,relu) + BatchNorm + MaxPool + Dropout(0.4) + Dense64 + BatchNorm + output
# 
# Define the model:

# In[ ]:


# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer
model_smaller_kernels = Sequential()
model_smaller_kernels.add(Conv2D(filters=48, kernel_size=(3,3),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_smaller_kernels.add(BatchNormalization())
model_smaller_kernels.add(Conv2D(filters=46, kernel_size=(3,3), activation='relu'))
model_smaller_kernels.add(BatchNormalization())
model_smaller_kernels.add(MaxPool2D())
model_smaller_kernels.add(Dropout(0.4))
model_smaller_kernels.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
model_smaller_kernels.add(BatchNormalization())
model_smaller_kernels.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
model_smaller_kernels.add(BatchNormalization())
model_smaller_kernels.add(MaxPool2D(padding='same'))
model_smaller_kernels.add(Dropout(0.4))
model_smaller_kernels.add(Flatten())
model_smaller_kernels.add(Dense(256, activation='relu'))
model_smaller_kernels.add(BatchNormalization())
model_smaller_kernels.add(Dense(num_classes, activation='softmax'))


# Compile 10 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 10
n_epochs = 20

# Use the model with better score and include augmented data. Repeat n_reps times for averaging
history_smaller_kernels, history_smaller_kernels_val = [0]*n_epochs, [0]*n_epochs

for rep in range(n_reps):
    # Compile the model
    model_smaller_kernels.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Fit the model
    history_smaller_kernels_rep = model_smaller_kernels.fit_generator(train_generator,
                                                         epochs = n_epochs, 
                                                         steps_per_epoch = X_train_validation.shape[0]//64,
                                                         validation_data = validation_generator,  
                                                         verbose=0)
    history_smaller_kernels = tuple(map(operator.add, history_smaller_kernels, history_smaller_kernels_rep.history['accuracy']))
    history_smaller_kernels_val = tuple(map(operator.add, history_smaller_kernels_val, history_smaller_kernels_rep.history['val_accuracy']))
    
history_smaller_kernels = [x/n_reps for x in list(history_smaller_kernels)]
history_smaller_kernels_val = [x/n_reps for x in list(history_smaller_kernels_val)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_smaller_kernels_val)
plt.plot(history_batch_norm_val)
plt.title('Model accuracy replacing Conv2D(5x5) by Conv2D(3x3)+Conv2D(3x3)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.99,1)
plt.xlim(0,n_epochs)
plt.legend(['3x3+3x3', '5x5'], loc='upper left')
plt.savefig('replace_big_convs.png')
plt.show()


# **Conclusion**: performance is clearly higher when Conv2D(5x5) layers are replaced by two Conv2D(3x3) layers. The non-linearities detected in the second case seem to be key in order to generalize digit recognition in difficult cases.

# In[ ]:


# predict results
results = model_smaller_kernels.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step11.csv",index=False)


# ## Experiment 9. Replacement of max pooling by convolutions with strides <a id="section11"></a>
# 
# There are other notable network architecture innovations which have yielded competitive results in image classification. One of these is to  replace max-pooling with a convolutional layer with increased stride, which yields competitive or state-of-the-art performance on several image recognition datasets. It has been stated that Conv2D(strides=2) layers does not just substitute MaxPooling functionality, but they also add the capability to learn from data.
# 
# See https://doi.org/10.1016/j.neucom.2018.07.079 for an extensive study of this method.
# 
# Replace pooling by convolutions model:
# * **Model_pool_conv**. Conv2D (48,3x3,relu) + BatchNorm + Conv2D (48,3x3,relu) + BatchNorm + Conv2D (48,3x3,relu,2strides) + Dropout(0.4) + Conv2D (96,5x5,relu) + BatchNorm + Conv2D (96,5x5,relu) + BatchNorm + Conv2D (48,3x3,relu,2strides) + Dropout(0.4) + Dense64 + BatchNorm + output
# 
# Define the model:

# In[ ]:


# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer
model_pool_conv = Sequential()
model_pool_conv.add(Conv2D(filters=48, kernel_size=(3,3),
                 activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
model_pool_conv.add(BatchNormalization())
model_pool_conv.add(Conv2D(filters=46, kernel_size=(3,3), activation='relu'))
model_pool_conv.add(BatchNormalization())
model_pool_conv.add(Conv2D(filters=46, kernel_size=(5,5), activation='relu', strides=2, padding='same'))
model_pool_conv.add(Dropout(0.4))
model_pool_conv.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
model_pool_conv.add(BatchNormalization())
model_pool_conv.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
model_pool_conv.add(BatchNormalization())
model_pool_conv.add(Conv2D(filters=46, kernel_size=(5,5), activation='relu', strides=2, padding='same'))
model_pool_conv.add(Dropout(0.4))
model_pool_conv.add(Flatten())
model_pool_conv.add(Dense(256, activation='relu'))
model_pool_conv.add(BatchNormalization())
model_pool_conv.add(Dense(num_classes, activation='softmax'))


# Compile 10 times and get statistics:

# In[ ]:


ts = time.time()

n_reps = 10
n_epochs = 20

# Use the model with better score and include augmented data. Repeat n_reps times for averaging
history_pool_conv, history_pool_conv_val = [0]*n_epochs, [0]*n_epochs

for rep in range(n_reps):
    # Compile the model
    model_pool_conv.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Fit the model
    history_pool_conv_rep = model_pool_conv.fit_generator(train_generator,
                                                         epochs = n_epochs, 
                                                         steps_per_epoch = X_train_validation.shape[0]//64,
                                                         validation_data = validation_generator,  
                                                         verbose=0)
    history_pool_conv = tuple(map(operator.add, history_pool_conv, history_pool_conv_rep.history['accuracy']))
    history_pool_conv_val = tuple(map(operator.add, history_pool_conv_val, history_pool_conv_rep.history['val_accuracy']))
    
history_pool_conv = [x/n_reps for x in list(history_pool_conv)]
history_pool_conv_val = [x/n_reps for x in list(history_pool_conv_val)]

print ("Time spent, " + str(time.time() - ts) + " s")


# Plot the model's performance:

# In[ ]:


# Plot the results
plt.plot(history_pool_conv_val)
plt.plot(history_smaller_kernels_val)
plt.title('Model accuracy replacing MaxPool by Conv2D with strides')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.99,1)
plt.xlim(0,n_epochs)
plt.legend(['Conv2D with strides', 'MaxPool'], loc='upper left')
plt.savefig('replace_maxpool_by_conv.png')
plt.show()


# **Conclusion**: this case in similar to batch normalization. The general behavior looks better when MaxPooling is not replaced by convolutional layers, but the long term tendency is better for convolutions with strides. Given that our objective is to increase the final CNN accuracy, I decided to keep this replacement (which I verified to be better through submissions).

# In[ ]:


# predict results
results = model_pool_conv.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step12.csv",index=False)


# ## Final model and submission <a id="section12"></a>
# 
# It has been a long journey through the intricacies of a CNN model, from simply modifying a few layer parameters to using advanced techniques as data augmentation. With this, we have been able to analyze which is the better architecture for the model, and how this affects the performance. 
# 
# The definitive CNN architecture is:
# * **Definitive model**: Conv2D (48,3x3,relu) + BatchNorm + Conv2D (48,3x3,relu) + BatchNorm + Conv2D (48,3x3,relu,2strides) + Dropout(0.4) + Conv2D (96,5x5,relu) + BatchNorm + Conv2D (96,5x5,relu) + BatchNorm + Conv2D (48,3x3,relu,2strides) + Dropout(0.4) + Dense64 + BatchNorm + output
# 
# Since this model is exactly the same we used in the previous step, we don't need to define it again. Let's train it for a large enough number of epochs:

# In[ ]:


ts = time.time()

n_reps = 1
n_epochs = 40

# Use the model with better score and include augmented data. Repeat n_reps times for averaging
history_definitive, history_definitive_val = [0]*n_epochs, [0]*n_epochs

# Callback function (early stopping)
callback_fcn = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+n_epochs))

for rep in range(n_reps):
    # Compile the model
    model_pool_conv.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Fit the model
    history_definitive_rep = model_pool_conv.fit_generator(train_generator,
                                                         epochs = n_epochs, 
                                                         steps_per_epoch = X_train_validation.shape[0]//64,
                                                         validation_data = validation_generator,  
                                                         callbacks=[callback_fcn],
                                                         verbose=0)
    history_definitive = tuple(map(operator.add, history_definitive, history_definitive_rep.history['accuracy']))
    history_definitive_val = tuple(map(operator.add, history_definitive_val, history_definitive_rep.history['val_accuracy']))
    
history_definitive = [x/n_reps for x in list(history_definitive)]
history_definitive_val = [x/n_reps for x in list(history_definitive_val)]

print ("Time spent, " + str(time.time() - ts) + " s")


# And finally submit the results,

# In[ ]:


# predict results
results = model_pool_conv.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit_step13.csv",index=False)


# This model reached a score of 0.99557, which is close to the top 10% models. Other kernels argued that, given the most difficult digits (some of them almost impossible to recognise by humans), the best results a single CNN could reach is around 99.8. Again, I recommend to see https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist for a reference. Hence, we can consider our results as *quite good*, and without doubt, it has been an enjoyable learning experience for me as my first image detection model.
# 
# Feel free to add any comments, suggestions or questions in the kernel's discussion, all types of feedback are always very welcome. Hope you enjoyed this work and see you around!
