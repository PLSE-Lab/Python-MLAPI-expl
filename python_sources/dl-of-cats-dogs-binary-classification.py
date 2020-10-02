#!/usr/bin/env python
# coding: utf-8

# This kernel is to show how to create a basic convolutional neutral network from scratch.And I chose to use Keras model.The proboem is to distinguish if the image is a cat or dog,and the data set is all about the image of them.We are going to set our own neutral network which is just simply containing convolutional layer,maxpooling layer,dropout layer.
# 
# ### Please enjoy your deep learning trip!

# In[ ]:


import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import preprocessing, layers, models, optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Conv2D, Dropout
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# #### Have a general check about the number of input file

# In[ ]:


get_ipython().system('ls ../input/test_set/test_set')
get_ipython().system('ls ../input/test_set/test_set/cats | wc -l')
get_ipython().system('ls ../input/test_set/test_set/dogs | wc -l')


# #### there are about 2000 samples in the test set

# In[ ]:


get_ipython().system('ls ../input/training_set/training_set/')
get_ipython().system('ls ../input/training_set/training_set/dogs | wc -l')
get_ipython().system('ls ../input/training_set/training_set/cats | wc -l')


# #### First we'll load the images into the workplace according to their path, and all the images append to the specific list.

# In[ ]:


path_cats = []
train_path_cats = '../input/training_set/training_set/cats'
for path in os.listdir(train_path_cats):
    if '.jpg' in path:
        path_cats.append(os.path.join(train_path_cats, path))
path_dogs = []
train_path_dogs = '../input/training_set/training_set/dogs'
for path in os.listdir(train_path_dogs):
    if '.jpg' in path:
        path_dogs.append(os.path.join(train_path_dogs, path))
len(path_dogs), len(path_cats)


# #### Then we convert the images into all pixels in a list or dataframe. As for here,we chose to save in some lists. The former 3000 elements are dogs as well as the latter ones are cats.

# In[ ]:


training_set = np.zeros((6000, 150, 150, 3), dtype='float32')
for i in range(6000):
    if i < 3000:
        path = path_dogs[i]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i - 3000]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)
     


# Print out all the elements in this list to have a check.
# 
# And maybe try to print the first image to see if the images are successfully changed into pixels file.

# In[ ]:


y_label = []
for i in range(6000):
    if i < 3000:
        y_label.append(0)
    else:
        y_label.append(1)

print(y_label)


# In[ ]:


print(training_set[0])


# #### For the trying period, we can change the scale into a smaller one to compute faster. After successfully created the whole model and can run it all, then we use the whole dataset.
# 
# #### we'll split the training dataset into training set and validation set,whose rate is basically 9:1. And we are going to use training set to fit a model and then verify if the model fit the validation set well.

# In[ ]:


for i in range(6000):
  training_set[i] = training_set[i].reshape(150,150,3)
  training_set[i] = training_set[i] / 255

# Set the random seed
random_seed = 2

x_train, val, y_train, y_val = train_test_split(training_set, y_label, test_size = 0.1, random_state=random_seed)

x_train = np.array(x_train)
type(x_train)


# In[ ]:


print(x_train.shape)


# #### Here is model-setting time, we don't need to make the model too complicated for the first trying. On the contrary, we can just make it easiest, regardless of the layers, regardless of the parameters and regardless of how well it seems to be or something like that. After the small scale's trying, then ue the biggest dataset to increase the accuracy.

# In[ ]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(5), padding='same',activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv2D(128, kernel_size=(4), padding='same',activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv2D(256, kernel_size=(4), padding='same',activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(layers.Flatten())
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
model.summary()

# def step_decay(epoch):
#     lrate = 0.001
#     if epoch > 2:
#         lrate = 0.0005
#     if epoch > 4:
#         lrate = 0.0001
#     if epoch > 5:
#         lrate = 0.00005
#     return lrate

# lrate = LearningRateScheduler(step_decay)

EPOCHS = 20
#fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
# model.fit(x_train, y_train, batch_size=1, epochs=EPOCHS, callbacks=[lrate])
history = model.fit(x_train, y_train, batch_size=30, validation_data=(val, y_val), epochs=EPOCHS)


# In[ ]:


model.save('cats_and_dogs_small_1.h5')


# After training, we use the matplotlib to visualize the learning history.
# 
# And we find that the model is over-fitting, so we must take some methods to reduce the loss function of validation dataset.

# In[ ]:



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


print(len(y_train))
print(y_train)


# Through some shift and roll, we take the images to a little bit shift or other operation like that. 

# In[ ]:


# Use data augmentation
train_datagen = preprocessing.image.ImageDataGenerator(
   rescale=1./255,
   rotation_range=40,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
)
train_generator = train_datagen.flow(
   x_train,
   y_train,
   batch_size=30)

# do not augment validation data
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow(
   val,
   y_val,
   batch_size=30)

# train
history = model.fit_generator(
   train_generator,
   steps_per_epoch=30,
   epochs= 50,
   validation_data = validation_generator,
   validation_steps=30)


# ### At last, we can apparently see that the loss function of validation dataset is tending to reduce as well as the accuracy is keeping raising. 

# In[ ]:



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


a = []

for i in range (5400):
  if i < 2700:
    if y_train[i] == 1:
      a.append(i)
  else:
    if y_train[i] == 0:
      a.append(i)
    
print(y_train)
print(len(a))


# In[ ]:


# for i in range(len(a)):
#     predict_img = preprocessing.image.array_to_img(x_train[a[i]])
#     i = str(i)
#     img = 'predict_img'+i+'.jpg'
#     predict_img.save(img)

