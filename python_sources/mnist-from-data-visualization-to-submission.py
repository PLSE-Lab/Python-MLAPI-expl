#!/usr/bin/env python
# coding: utf-8

# # MNIST : from data visualization to submission
# The purpose of this kernel is to take the MNIST dataset, to visualize it to then train a small CNN and output a submission.
# * Data insights
# * Model definition
# * Data augmentation
# * Submission

# In[ ]:


# Libraries import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
tf.set_random_seed(42)


# In[ ]:


# Settings
train_path = os.path.join('..', 'input', 'train.csv')
test_path = os.path.join('..', 'input', 'test.csv')

# CNN model settings
size = 28
lr = 0.001
num_classes = 10

# Training settings
epochs = 30
batch_size = 128


# In the next two cells, I will load the data and process it from an array of 784 pixels values to an 2D 28*28 matrix, to be able to work on images. This will let me use a CNN later.

# In[ ]:


# data loading
raw_train_df = pd.read_csv(train_path)
raw_test_df = pd.read_csv(test_path)


# In[ ]:


# Utils
def parse_train_df(_train_df):
    labels = _train_df.iloc[:,0].values
    imgs = _train_df.iloc[:,1:].values
    imgs_2d = np.array([[[[float(imgs[index][i*28 + j]) / 255] for j in range(28)] for i in range(28)] for index in range(len(imgs))])
    processed_labels = [[0 for _ in range(10)] for i in range(len(labels))]
    for i in range(len(labels)):
        processed_labels[i][labels[i]] = 1
    return np.array(processed_labels), imgs_2d

def parse_test_df(test_df):
    imgs = test_df.iloc[:, 0:].values
    imgs_2d = np.array([[[[float(imgs[index][i * 28 + j]) / 255] for j in range(28)] for i in range(28)] for index in
                        range(len(imgs))])
    return imgs_2d


# In[ ]:


# Data preprocessing
y_train_set, x_train_set = parse_train_df(raw_train_df)
x_test = parse_test_df(raw_test_df)

x_train, x_val, y_train, y_val = train_test_split(x_train_set, y_train_set, test_size=0.20, random_state=42)


# ## Data insights
# 
# Here, we will look a bit into the training dataset. It's important to know with which kind of data we are playing with.
# 
# First, let's check that we have the same number of samples for each one of our classes. If not, it will be harder to train an unbiased classifier.

# In[ ]:


# Training data insights
raw_train_df['label'].value_counts().plot.bar()


# The biggest class is *1* with about 4500 samples in our training data. The smallest one is *5* with about 3800 samples.

# In[ ]:


print("Number of 1: {}".format(len(raw_train_df[raw_train_df['label'] == 1])))
print("Number of 5: {}".format(len(raw_train_df[raw_train_df['label'] == 5])))


# Now, we have the right numbers !! This should be ok to train a classifier with these proportions. I just wanted to check that the dataset doesn't have one big class with ten times more samples than the other ones. (or one with ten times less samples). I will continue this notebook with the raw dataset as it should be fine.
# 
# Let's look at some images, just to see with what we are dealing.

# In[ ]:


# Image visualization
n = 5
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x_train[i]+1)/2
    ax.imshow((-x_train[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
plt.show()


# ## Model definition
# Here, I will define and train a first CNN model. I decided to use 4 Conv2D layers and 2 Denses ones (before the output layer) because of what I have seen on the internet. It works fine, but it is suremy not the best solution. 

# In[ ]:


# CNN model
model = keras.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(size, size, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr),
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_ckpt.{epoch:02d}.hdf5',
                                             save_best_only=True,
                                             save_weights_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3,
                      mode='max', cooldown=3, verbose=1)
callback_list = [checkpoint, lr_reducer]


# In[ ]:


# Training
training_history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=callback_list
)


# Let's look at some graphs to see if the training phase went well.

# In[ ]:


# Training recap
epoch_range = [e for e in range(1, epochs + 1)]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 4))

axs[0].plot(epoch_range,
         training_history.history['loss'],
         training_history.history['val_loss'],
)
axs[0].set_title('Training loss')
axs[1].plot(epoch_range,
         training_history.history['acc'],
         training_history.history['val_acc'],
)
axs[1].set_title('Training accuracy')

plt.show()


# The validation accuracy does not drop or remain still while the accuracy on the training set continue to rise. Moreover, the gap between them is relatively small. We cannot say that the model overfitted.

# ## Data augmentation
# I will now use the ImageDataGenerator to increase the number of samples of the training dataset. This Keras class will take the training images in input and will output new images, produced from the input images by zooming a bit in them, by rotating them in a certain range, by shifting them a bit, ...
# 
# I will only use the zoom, the rotation and the shift transformation as a written digit could be a bit bigger or smaller, a bit rotated or a bit shifted in the image. I want to create image that have a meaning for this dataset. For instance, I won't use the horizontal or the vertical flip transformation as a number upside down isn't a number anymore.
# 
# The training phase is the same than we our previous model.

# In[ ]:


# Creating image generator
image_generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

image_generator.fit(x_train)


# In[ ]:


# Retraining the same model
model_augmented = keras.Sequential()

model_augmented.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(size, size, 1)))
model_augmented.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2)))
model_augmented.add(BatchNormalization())
model_augmented.add(Dropout(0.3))

model_augmented.add(Conv2D(64, (3, 3), activation='relu'))
model_augmented.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model_augmented.add(BatchNormalization())
model_augmented.add(Dropout(0.3))
model_augmented.add(Conv2D(128, (3, 3), activation='relu'))
model_augmented.add(BatchNormalization())

model_augmented.add(Flatten())
model_augmented.add(Dense(256, activation='relu'))
model_augmented.add(Dropout(0.25))
model_augmented.add(Dense(128, activation='relu'))
model_augmented.add(Dense(num_classes, activation='softmax'))

model_augmented.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(0.001),
              metrics=['accuracy'])

aug_result = model_augmented.fit_generator(
    image_generator.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train) // batch_size,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=callback_list
)

model_augmented.save('mnist_model.h5')


# In[ ]:


# Training recap of your augmented dataset
epoch_range = [e for e in range(1, epochs + 1)]
plt.plot(epoch_range,
         aug_result.history['acc'],
         aug_result.history['val_acc'],
)
plt.title('Accuracy')
plt.show()


# ## Submission
# Here, I will create two submission files, one with the raw dataset and one with the augmented one. These files will be created once the notebook forked and commited. To create these submission, I just read the test csv file and use the two models to predict the written digits on it.

# In[ ]:


# Prediction
pred = model.predict(x_test)
pred_aug = model_augmented.predict(x_test)


# In[ ]:


# Submission creation
def convert_prediction_result(model_result):
    result = []
    for i in range(len(model_result)):
        result += [np.argmax(model_result[i])]
    return result


def write_submission(_submission_path, result_arr):
    f_out = open(_submission_path, 'w')
    f_out.write("ImageId,Label\n")
    for i in range(len(result_arr)):
        f_out.write("{},{}\n".format(i+1, result_arr[i]))
    f_out.close()

write_submission('submission_base.csv', convert_prediction_result(pred))
write_submission('submission_aug.csv', convert_prediction_result(pred))


# These small CNN manage to achieve 0.9877 on the MNIST dataset. 
# 
# Feel free to use this notebook as a starting point. And feel free to upvote :)
