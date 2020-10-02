#!/usr/bin/env python
# coding: utf-8

# Started on 24 June 2019

# # Introduction

# #### Following from working on [Aerial Cactus Identification][1] problem using CNN, I try to adapt the same approach on this [Histopathologic Cancer Detection][2] problem.
# [1]: https://www.kaggle.com/rhodiumbeng/aerial-cactus-identification-using-cnn
# [2]: https://www.kaggle.com/c/histopathologic-cancer-detection
# #### Unlike the cactus datasets, the size of the data here (essentially the images) are too large to be loaded as a whole into memory for training and prediction. So we have to use the 'flow' functionality from Keras' ImageDataGenerator.
# #### I would like to thank [Marsh][3] for sharing his insightful [kernel][4]. I learned a lot from it.
# [3]: https://www.kaggle.com/vbookshelf
# [4]: https://www.kaggle.com/vbookshelf/cnn-how-to-use-160-000-images-without-crashing
# #### Separately, I also found this wonderful [resource][5] from [Vijayabhaskar J][6] on using "flow_from_dataframe" in ImageDataGenerator. Vijaybhaskar wrote this function that got accepted to the official keras-preprocessing git repo. This allows us to input a Pandas dataframe which contains the filenames column and a column which has the class names and directly read the images from the directory with their respective class names mapped. Wonderful!
# [5]: https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# [6]: https://medium.com/@vijayabhaskar96
# #### I had based this kernel very much from the guidance from the above resources.

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# # Examine the data

# * The images (tif) for the training data and test data are found in the train and test folders. The filenames of the tif image files are used as the unique 'id' in the csv files.
# * 'train_csv' contains the training data ('id' and 'label') and 'sample_submission.csv' contains the test data 'id'.

# In[ ]:


# load data from csv files
train_df = pd.read_csv('../input/train_labels.csv')
test_df = pd.read_csv('../input/sample_submission.csv')
print(train_df.shape, test_df.shape)


# * The 'id' in the csv files are without file extension. So we add '.tif' to make them correspond exactly to the image files.

# In[ ]:


train_df['id'] = train_df['id'].apply(lambda x: x+'.tif')
test_df['id'] = test_df['id'].apply(lambda x: x+'.tif')


# In[ ]:


train_df['label'] = train_df['label'].astype(str)


# In[ ]:


train_df['label'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

train_path = '../input/train/'
test_path = '../input/test/'


# #### Here are some images from the training data that are labelled as positive, i.e. '1':

# In[ ]:


# look at some of the pics from train_df labelled '1'
positive = train_df[train_df['label']=='1']
plt.figure(figsize=(15,7))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(load_img(train_path+positive.iloc[i]['id']))
    plt.title("label=%s" % positive.iloc[i]['label'], y=1)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()


# #### Here are some images from the training data that are labelled negative, i.e. '0':

# In[ ]:


# look at some of the pics from train_df labelled '0'
negative = train_df[train_df['label']=='0']
plt.figure(figsize=(15,7))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(load_img(train_path+negative.iloc[i]['id']))
    plt.title("label=%s" % negative.iloc[i]['label'], y=1)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()


# # Setting up ImageDataGenerator

# In[ ]:


from keras_preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)


# In[ ]:


# set up two data generators; (1) training, (2) validation from train set
n_x = 96
train_generator = datagen.flow_from_dataframe(dataframe=train_df, 
                                              directory=train_path, 
                                              target_size=(n_x,n_x), 
                                              x_col='id', y_col='label', 
                                              subset='training', 
                                              batch_size=128, seed=12, 
                                              class_mode='categorical')


# In[ ]:


valid_generator = datagen.flow_from_dataframe(dataframe=train_df, 
                                              directory=train_path,
                                              target_size=(n_x,n_x), 
                                              x_col='id', y_col='label', 
                                              subset='validation', 
                                              batch_size=128, seed=12, 
                                              class_mode='categorical')


# In[ ]:


# set up data generator for test set
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, 
                                                  directory=test_path, 
                                                  target_size=(n_x,n_x), 
                                                  x_col='id', y_col=None, 
                                                  batch_size=1, seed=12, 
                                                  shuffle=False, 
                                                  class_mode=None)


# In[ ]:


# define step sizes for model training
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
step_size_test = test_generator.n//test_generator.batch_size
print(step_size_train, step_size_valid, step_size_test)


# # Create CNN Model

# In[ ]:


# build the CNN from keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=5, activation='relu', input_shape=(96, 96, 3)))
model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()


# In[ ]:


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])


# #### Run the model on the train and validation data, and capture metrics history to visualise the performance of the model

# In[ ]:


# Train and validate the model
epochs = 20
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train, 
                              validation_data=valid_generator, 
                              validation_steps=step_size_valid,
                              epochs=epochs)


# In[ ]:


# plot and visualise the training and validation losses
loss = history.history['loss']
dev_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

from matplotlib import pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, dev_loss, 'b', label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Predictions

# In[ ]:


# predict on test set
test_generator.reset()
pred = model.predict_generator(test_generator, steps=step_size_test, 
                               verbose=1)


# In[ ]:


# create submission file
sub = pd.read_csv('../input/sample_submission.csv')
sub['label'] = pred[:,0]
sub.head()


# In[ ]:


# generate submission file in csv format
sub.to_csv('submission.csv', index=False)

