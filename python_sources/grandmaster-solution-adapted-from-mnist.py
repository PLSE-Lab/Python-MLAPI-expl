#!/usr/bin/env python
# coding: utf-8

# # Adapted from MNIST competition
# This is an implementation of Kaggle Grandmaster Chris Deotte's [notebook](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist) in the original MNIST competition. It has been slightly modified to fit the time requirements of this competition, and to take advantage of the holdout set here for further validation.

# In[ ]:


# LOAD LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# ## Load Kaggle's training images

# In[ ]:


# LOAD THE DATA
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
holdout = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


# In[ ]:


# PREPARE DATA FOR NEURAL NETWORK
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
# prepare the test data set by removing the id
X_test.drop(labels=['id'], axis=1, inplace=True)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

len_test = len(X_test)


# In[ ]:


X_holdout = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
# prepare the holdout data set
y_holdout = X_holdout['label']
X_holdout.drop(labels=['label'], axis=1, inplace=True)

#prepare holdout data
X_holdout = X_holdout.astype('float32') / 255.
X_holdout = X_holdout.values.reshape(X_holdout.shape[0], 28, 28, 1).astype('float32')


# ### View some of the images

# In[ ]:


import matplotlib.pyplot as plt
# PREVIEW IMAGES
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# ### Generate more images!!
# by randomly rotating, scaling, and shifting Kaggle's training images.

# In[ ]:


# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# In[ ]:


# PREVIEW AUGMENTED IMAGES
X_train3 = X_train[9,].reshape((1,28,28,1))
Y_train3 = Y_train[9,].reshape((1,10))
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()
    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
    if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))
    if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# ## Build Multiple Convolutional Neural Networks!
# The time limit on running gpu kernels in this competition means that we will run less than the 15 kernels the original kernel had in the MNIST competition.

# In[ ]:


# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 4
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# # Architectural highlights
# ![LeNet5](http://playagricola.com/Kaggle/LeNet5.png)
# The CNNs in this kernel follow [LeNet5's][1] design (pictured above) with the following improvements:  
# * Two stacked 3x3 filters replace the single 5x5 filters. These become nonlinear 5x5 convolutions
# * A convolution with stride 2 replaces pooling layers. These become learnable pooling layers.
# * ReLU activation replaces sigmoid.
# * Batch normalization is added
# * Dropout is added
# * More feature maps (channels) are added
# * An ensemble of 15 CNNs with bagging is used  
#   
# Experiments [(here)][2] show that each of these changes improve classification accuracy.
# 
# [1]:http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
# [2]:https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

# # Train Multiple CNNs

# In[ ]:


# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
epochs = 45
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))


# # Ensemble Multiple CNN predictions and submit

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (X_test.shape[0],10) )
results_holdout = np.zeros( (X_holdout.shape[0],10) )
for j in range(nets):
    results = results + model[j].predict(X_test)
    results_holdout = results_holdout + model[j].predict(X_holdout)

results_holdout = np.argmax(results_holdout,axis = 1)
holdout_accuracy = accuracy_score(y_holdout, results_holdout)
print( " Holdout Accuracy = %3.4f"% (holdout_accuracy))

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="label")
submission = pd.concat([pd.Series(range(0,len_test),name = "id"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# In[ ]:


# PREVIEW PREDICTIONS
plt.figure(figsize=(15,6))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("predict=%d" % results[i],y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()


# # Final Thoughts
# As the original kernel shows, there is an upper limit beyond which it isn't possible to classify accurately. For the other MNIST competition this is around 99.8%. Validation scores for this kernel have been around that level but the private leaderboard scores have been lower. Further, the public leaderboard scores for this kernel are below some other publicly available kernels in this competition. It is possible that this is random fluctuation, however, the consistency with which other models have scored slightly higher makes this seem unlikely. 
# 
# The code here is clearly a good solution to this competition and is based on the excellent results this same approach achieved in the other MNIST competition. I have created discussions [here](https://www.kaggle.com/c/Kannada-MNIST/discussion/111460) and [here](https://www.kaggle.com/c/Kannada-MNIST/discussion/111462) that explore why the accuracy on the public leaderboard may be lower than the validation accuracy for this competition.
