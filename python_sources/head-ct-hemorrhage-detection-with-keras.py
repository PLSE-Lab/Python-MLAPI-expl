#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


labels_df = pd.read_csv('../input/labels.csv')
labels = np.array(labels_df[' hemorrhage'].tolist())

files = sorted(glob.glob('../input/head_ct/head_ct/*.png'))
images = np.array([cv2.imread(path) for path in files])


# # Initial data exploration

# In[ ]:


labels_df[' hemorrhage'].hist(bins=2)


# There is the same amount of data for both cases.

# Images are not the same sizes! We need to find the optimal size, but before we have to explore it.

# In[ ]:


images_df = pd.DataFrame(images, columns=['image'])


# In[ ]:


images_df['width'] = images_df['image'].apply(lambda x: x.shape[0])
images_df['height'] = images_df['image'].apply(lambda x: x.shape[1])


# In[ ]:


images_df[['height', 'width']].hist(bins=20)


# In[ ]:


images_df[['height', 'width']].describe()


# Before we will create and train model, we need to make all images the same sizes.
# 
# The tradeoff is simple here - lesser images would be faster to train, there would be a lot of examples so lesser chance of overfitting, but it is a clear loss of information. If the error would be still big, we will need to consider to use a bigger size, and either stretch little images (and lose quality significantly) or drop them entirely (and risk overfitting).
# 
# For now we will go the simplest path - resizing to the smallest size (and even smaller - 128 insted of 134).

# In[ ]:


images = np.array([cv2.resize(image, (128, 128)) for image in images])


# In[ ]:


plt.imshow(images[0])


# In[ ]:


plt.imshow(images[100])


# The quality of images seems to be acceptable.

# # Adding flipped images

# We could also improve the dataset by adding flipped images. It doesn't matter from what side we will look at the CT scan, brain hemorrhage can and should be diagnosed just as well. By adding flipped images to dataset, we can greatly increase the accuracy of model.

# In[ ]:


plt.figure(figsize=(12, 12))
for i, flip in enumerate([None, -1, 0, 1]):
    plt.subplot(221 + i)
    if flip is None:
        plt.imshow(images[0])
    else:
        plt.imshow(cv2.flip(images[0], flip))


# Now, we don't want those flipped images in our test set just to be sure model didn't create any preferences for upside down and flipped images, so the dataset expansion should take place after split into train and test sets.
# 
# Fortunately, there is ImageDataGenerator for the purposes of flipping and rotating images.

# Split data into train, validation and test subsets.

# In[ ]:


print(labels)


# In[ ]:


# since data is strictly true until index 100 and then strictly false,
# we can take random 90 entries from frist half and then random 90 from the second half
# to have evenly distributed train and test sets
indicies = np.random.permutation(100)
train_true_idx, test_true_idx = indicies[:90], indicies[90:]
train_false_idx, test_false_idx = indicies[:90] + 100, indicies[90:] + 100
train_idx, test_idx = np.append(train_true_idx, train_false_idx), np.append(test_true_idx, test_false_idx)

train_validationX, train_validationY = images[train_idx], labels[train_idx]
testX, testY = images[test_idx], labels[test_idx]

print(train_validationX.shape, testX.shape)
print(train_validationY.shape, testY.shape)


# In[ ]:


# now to split train and validation sets
tr_len = train_validationX.shape[0]
train_val_split = int(tr_len*0.9)
indicies = np.random.permutation(tr_len)
train_idx, validation_idx = indicies[:train_val_split], indicies[train_val_split:]

trainX, trainY = train_validationX[train_idx], train_validationY[train_idx]
validationX, validationY = train_validationX[validation_idx], train_validationY[validation_idx]

print(trainX.shape, validationX.shape)
print(trainY.shape, validationY.shape)


# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import math


# # Image augmentation

# In[ ]:


train_image_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.,
    zoom_range=0.05,
    rotation_range=180,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
validation_image_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.,
    zoom_range=0.05,
    rotation_range=90,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0)


# In[ ]:


plt.figure(figsize=(12, 12))
for X_batch, y_batch in train_image_data.flow(trainX, trainY, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i])
    plt.show()
    break


# # Building the model

# In[ ]:


def check_accuracy(model, setX, actual, print_images=True):
    predicted = np.array([int(x[0] > 0.5) for x in model.predict(setX)])
    if print_images:
        rows = math.ceil(len(predicted)/10.)
        plt.figure(figsize=(20, 3 * rows))
        for i in range(len(predicted)):
            plt.subplot(rows, 10, i+1)
            plt.imshow(setX[i])
            plt.title("pred "+str(predicted[i])+" actual "+str(actual[i]))
        
    confusion = confusion_matrix(actual, predicted)
    tn, fp, fn, tp = confusion.ravel()
    print("True positive:", tp, ", True negative:", tn,
          ", False positive:", fp, ", False negative:", fn)

    print("Total accuracy:", np.sum(predicted==actual) / len(predicted) * 100., "%")
    return (tn, fp, fn, tp)


# In[ ]:


def simple_conv_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.4))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[ ]:


model = simple_conv_model((128, 128, 3))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


model.summary()


# # Training model

# In[ ]:


model.fit_generator(train_image_data.flow(trainX, trainY, batch_size=128),
    steps_per_epoch=128,
    validation_data=validation_image_data.flow(validationX, validationY, batch_size=16),
    validation_steps=100,
    callbacks=[ModelCheckpoint("weights.h5", monitor='val_acc', save_best_only=True, mode='max')],
    epochs=16)


# In[ ]:


check_accuracy(model, validationX/255., validationY)


# In[ ]:


model.save("last-weights.h5")
model.load_weights("weights.h5")


# In[ ]:


check_accuracy(model, trainX/255., trainY, False)


# In[ ]:


check_accuracy(model, validationX/255., validationY)


# The overall generalization of model seems good, overfitting isn't too big. But since this is a medical problem, we have to consider one additional thing.

# # False negative result will kill patient
# False positive result will be an inconvinience.
# 
# We have to punish false negative results while training the model.

# Punishing false negatives may be implemented in several ways.
# * imbalance dataset so there are more positive cases, therefore model will prefer false positives over false negatives
# * make it a multiclass classification and use 'class_weight' parameter of Keras (which is essentially will do the same trick)
# * write custom loss function that is oriented on lowering false negativer rate (or improving _sensitivity_)
# * or write custom metrics, based on which checkpoint will save model
# 
# Let's try the approach with imbalancing training dataset.

# In[ ]:


def imbalance_set(coeff=2):
    imbalanced_trainX = []
    imbalanced_trainY = []
    for i, train_x in enumerate(trainX):
        def add_entry(x, y):
            imbalanced_trainX.append(x)
            imbalanced_trainY.append(y)

        add_entry(train_x, trainY[i])

        if(trainY[i] == 1):
            for j in range(coeff-1):
                add_entry(train_x, trainY[i])
    return (np.array(imbalanced_trainX), np.array(imbalanced_trainY))

imbalanced_trainX, imbalanced_trainY = imbalance_set(2)
print(imbalanced_trainX.shape, imbalanced_trainY.shape)


# In[ ]:


def bigger_conv_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.4))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[ ]:


model = bigger_conv_model((128, 128, 3))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(imbalanced_trainX, imbalanced_trainY, validation_data=(validationX, validationY),
          callbacks=[ModelCheckpoint("weights-fna-model.hdf5", monitor='val_acc', save_best_only=True, mode='max')],
          batch_size=128, epochs=200)
# In[ ]:


model.fit_generator(train_image_data.flow(imbalanced_trainX, imbalanced_trainY, batch_size=128),
    steps_per_epoch=128,
    validation_data=validation_image_data.flow(validationX, validationY, batch_size=16),
    validation_steps=100,
    callbacks=[ModelCheckpoint("bigger_model_checkpoint_weights.h5", monitor='val_acc', save_best_only=True, mode='max')],
    epochs=24)


# In[ ]:


check_accuracy(model, trainX/255., trainY, False)


# In[ ]:


check_accuracy(model, validationX/255., validationY, False)


# In[ ]:


model.save("bigger_model_latest_weights.h5")
model.load_weights("bigger_model_checkpoint_weights.h5")


# In[ ]:


check_accuracy(model, trainX/255., trainY, False)


# In[ ]:


check_accuracy(model, validationX/255., validationY, False)


# # 89% of accuracy on validation set and 0 false negative
# Time to check model on test set

# In[ ]:


check_accuracy(model, testX/255., testY)


# Model showed good results.
# 
# Additional improvements could be made if image augmentation contained alterations of contrast.

# In[ ]:




