#!/usr/bin/env python
# coding: utf-8

# **INTRO**
# 
# MNIST dataset has 70000 28x28 grayscale images and a single label for each image. 
# 
# In this kaggle competition the original MNIST dataset splitted into a train dataset with 42000 images and labels and a test dataset with 28000 images only. 
# There are no labels in kaggle's test dataset. 
# 
# So in this competition for training you should use kaggle's train dataset with 42000 images and not the original dataset with 60000 or even 70000 images as some competitors do.
# If you find submission with kaggle score higher than 0.998 in the Leaderboard, it is very likely that more than 42k images were used for training. 
# See details about it in section 'How much more accuracy is possible?' in [this fine kernel](http://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist) by Chris Deotte.
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# LOAD THE DATA
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

print (train.shape)
print (test.shape)


# In[ ]:


# PREPARE DATA FOR FEEDING YOUR CNN
from keras.utils.np_utils import to_categorical
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


# **The model** 
# 
# There are many structures you may find out. This model below is very much inspired by [Chris Deotte](https://www.kaggle.com/cdeotte)'s [25 Million Images! MNIST kernel](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist). I highly recommend reading his kernels and also those ones he refers to.
# 
# This model below contains 7 convolutional layers and 1 dense layer in one net
# and it is an ensemble-of-nets type solution which I found very useful to improve accuracy in MNIST competition.
# 
# I kept C4 and dense layers fixed and played a bit with the order of top 6 layers and found C3-C3-C5-C5-C7-C7-C4-FC10 structure the best. Feel free to permutate top 6 layers by changing kernel sizes below and see changes in Trainable parameters printed below. For kernel sizes you may use other than 3,3,5,5,7,7 of course but you may need to change other parameters in the model.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.initializers import he_normal
from keras.optimizers import Adam

nets = 15
model = [0] *nets

#Conv layer kernel sizes for the top 6 layers
ks1=3
ks2=3
ks3=5
ks4=5
ks5=7
ks6=7
init = he_normal(seed=82)

for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=ks1, activation='relu', kernel_initializer = init, input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    #model[j].add(Dropout(0.4))
    model[j].add(Conv2D(32, kernel_size=ks2, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    #model[j].add(Dropout(0.4))
    model[j].add(Conv2D(32, kernel_size=ks3, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=ks4, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    #model[j].add(Dropout(0.4))
    model[j].add(Conv2D(64, kernel_size=ks5, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    #model[j].add(Dropout(0.4))
    model[j].add(Conv2D(64, kernel_size=ks6, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size=4, activation='relu', kernel_initializer = init ))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    optA = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model[j].compile(optimizer=optA, loss="categorical_crossentropy", metrics=["accuracy"])

model[0].summary()


# **Regularization and optimizatio**n
# 
# In the model there are hundreds of thousands of parameters. 
# To find the best set ... or one of the best sets of parameters in a reasonable time - there are several tricks you may apply.
# Without going into details these tricks are dropout, batch normalization, ADAM. These are already there, defined in the model.
# 
# There is one more trick in this kernel which is data augmentation. This is a very useful trick and it does a little change in the original image. Using this trick you alter the original image just a little and you do this change in a random way. So while training you take the original image then alter a bit and you do the training. 
# Two things which I learned from this.
# 1. Original image never used as input for the network during training.
# 2. Practically you never train with same exact image.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.10, width_shift_range=0.1, height_shift_range=0.1)


# **Controlling the process of training**
# 

# In[ ]:


from keras.callbacks import ReduceLROnPlateau, EarlyStopping

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.31623, min_delta=1e-4)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=10, verbose=1, mode='max', restore_best_weights=True)


# And here comes the training.
# Before the actual training there is a further split of the MNIST training dataset into the actual training set and a validation set. This split is randomly done for each network. So very probably we use up all images for training.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

history = [0] * nets
epochs = 50
for j in range(nets):
    rs = 10*j + 1
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, train_size = 0.9, random_state = rs)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64), epochs = epochs, steps_per_epoch = X_train2.shape[0]//64, validation_data = (X_val2,Y_val2), callbacks=[learning_rate_reduction, early_stopping], verbose=0)
    #pred = model[j].predict_classes(X_val2)
    #Y_val0 = np.argmax(Y_val2,axis=1) 
    maxpos = history[j].history['val_acc'].index(max(history[j].history['val_acc']))
    print("N{0:d}:: Max val_acc={1:.5f} at Epoch {2:d} Min_tr_loss={3:.5f} Min_val_loss={4:.5f}".format(j+1,max(history[j].history['val_acc']),maxpos+1, min(history[j].history['loss']), min(history[j].history['val_loss']) ))


# **Summarize ensemble results and create submission file**

# In[ ]:


results = np.zeros( (X_test.shape[0],10) ) 
for j in range(nets):
    results = results + model[j].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST_32C3C3C564C5C7C7128C4FC10_50e_LR10_RLROP_ES_BN_DA_DO40_TR90_N15_ADAM_xx.csv",index=False)

print(submission.shape)

