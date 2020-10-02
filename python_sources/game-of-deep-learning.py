#!/usr/bin/env python
# coding: utf-8

# # Game of Deep Learning

# ## Problem Statement
# 
# Ship or vessel detection has a wide range of applications, in the areas of maritime safety,  fisheries management, marine pollution, defence and maritime security, protection from piracy, illegal migration, etc.
# 
# Keeping this in mind, a Governmental Maritime and Coastguard Agency is planning to deploy a computer vision based automated system to identify ship type only from the images taken by the survey boats. You have been hired as a consultant to build an efficient model for this project.
# 
# There are 5 classes of ships to be detected which are as follows: 
# - Cargo
# - Military 
# - Carrier
# - Cruise
# - Tankers
# 
# You can download dataset from this [Link](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/)

# ## Dataset Description
# There are 6252 images in train and 2680 images in test data. The categories of ships and their corresponding codes in the dataset are as follows -
# 
# `{'Cargo': 1, 
# 'Military': 2, 
# 'Carrier': 3, 
# 'Cruise': 4, 
# 'Tankers': 5}`
# 
# There are three files provided to you, viz train.zip, test.csv and sample_submission.csv which have the following structure.
# 
# Variable 	|Definition
# ------------- |-----------
# image 	   |  Name of the image in the dataset (ID column)
# category 	|Ship category code

# ## Evaluation Metric
# 
# The Evaluation metric for this competition is weighted F1 Score.

# ## Import Packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.listdir('../input')


# In[ ]:


os.listdir('../input/data/data')


# In[ ]:


train_data=pd.read_csv('../input/data/data/train.csv',dtype=str)
train_data.head()


# In[ ]:


train_data.dtypes


# In[ ]:


train_data.count()


# In[ ]:


test_data=pd.read_csv('../input/data/data/test_ApKoW4T.csv')
test_data.head()


# In[ ]:


test_data.count()


# In[ ]:


sample_sub=pd.read_csv('../input/data/data/sample_submission_ns2btKE.csv')
sample_sub.head()


# In[ ]:


sample_sub.count()


# In[ ]:


sample_sub.tail()


# In[ ]:


train_data.tail()


# ## Total In count for train and test data

# In[ ]:


train_data['category'].value_counts()


# In[ ]:


train_data['category'].value_counts().plot.bar()
plt.show()


# ## See sample image

# In[ ]:


filenames = os.listdir("../input/data/data/images")
sample = random.choice(filenames)
image = load_img("../input/data/data/images/"+sample)
plt.imshow(image)
plt.show()


# In[ ]:





# ## Build Model

# In[ ]:


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB color


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


# In[ ]:





# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:





# In[ ]:


model.summary()


# In[ ]:





# ## Callbacks

# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ### Early Stop
# 
# To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased
# 

# In[ ]:


earlystop = EarlyStopping(patience=10)


# ### Learning Rate Reduction
# 
# We will reduce the learning rate when then accuracy not increase for 2 steps

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


callbacks = [earlystop, learning_rate_reduction]


# ### Prepare Validate and Train Data

# In[ ]:





# In[ ]:


train_df, validate_df = train_test_split(train_data, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)


# In[ ]:


train_df['category'].value_counts()


# In[ ]:


train_df['category'].value_counts().plot.bar()
plt.show()


# In[ ]:


validate_df['category'].value_counts()


# In[ ]:


validate_df['category'].value_counts().plot.bar()
plt.show()


# In[ ]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# ## Traning Generator

# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, 
    directory="../input/data/data/images/", 
    x_col='image',
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# ## Validation Generator

# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/data/data/images/", 
    x_col='image',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[ ]:





# ## See how our generator work

# In[ ]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../input/data/data/images/", 
    x_col='image',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[ ]:





# ## Fit Model

# In[ ]:


epochs=3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[ ]:





# In[ ]:





# ## Save Model

# In[ ]:


model.save_weights("model.h5")


# In[ ]:





# In[ ]:





# ## Virtualize Training

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## Create Testing Generator

# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/data/data/images/", 
    x_col='image',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[ ]:


nb_samples = test_df.shape[0]
nb_samples


# ## Predict

# In[ ]:


test_generator.reset()
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)
predicted_class_indices


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_df.image
results=pd.DataFrame({"image":filenames,
                      "category":predictions})
results.to_csv("results2.csv",index=False)


# In[ ]:





# In[ ]:




