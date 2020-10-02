#!/usr/bin/env python
# coding: utf-8

# **Import all the necessary libraries**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169


# **Create the base directory**

# In[2]:


base_dir = os.path.join('..', 'input')


# **Load the train.csv as Pandas DataFrame**

# In[3]:


train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
train_df.head()


# This is a binary classification problem. Therefore, the last layer of the neural network should use "sigmoid" activation

# In[4]:


#Let's see how many training data:
print("There are {} training images".format(len(train_df)))


# In[5]:


train_images = os.path.join(base_dir, 'train/train')
test_images = os.path.join(base_dir, 'test/test')

print("There are {} images in the train folder.".format(len(os.listdir(train_images))))
print("There are {} images in the test folder.".format(len(os.listdir(test_images))))


# **Process the data**

# In[6]:


#create a column in dataframe to store image paths
#let's test the glob function first:
i = 0
for x in glob(os.path.join(train_images,'*.jpg')):
    print(x)
    i += 1
    
    if i == 5:
        break


# Create a dictionary that matches Image ID and Image Path

# In[7]:


image_id_path = {os.path.basename(x): x for x in glob(os.path.join(train_images, '*.jpg'))}


# Create the column "path":

# In[8]:


train_df['path'] = train_df['id'].map(image_id_path.get)


# In[9]:


train_df.head()


# Let load the image pixels to a column named "images"

# In[10]:


train_df['images'] = train_df['path'].map(lambda x: np.array(Image.open(x).resize((32,32))))


# Normalize the image pixels

# In[11]:


train_df['images'] = train_df['images'] / 255


# In[12]:


#each image's shape
train_df['images'][0].shape


# **EDA**

# Check if there is any null values

# In[13]:


train_df.isnull().sum()


# Check data balance

# In[14]:


train_df['has_cactus'].value_counts().plot(kind='bar')


# There are more than half of the images that contain cactus. The dataset is not balanced. 

# Visualize some sample images

# In[15]:


n_rows = 5
n_cols = 5

fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,16))

i = 0
for each_row in ax:
    
    for each_row_col in each_row:
        each_row_col.imshow(train_df['images'][i])
        
        if train_df['has_cactus'][i] == 1:
            each_row_col.set_title('Cactus')
        else:
            each_row_col.set_title('None')
        
        each_row_col.axis('off')
        i += 1


# Intuitively after looking at the sample images, the ones that contain cactus have columnar shapes or "line" shapes. We'll train a model to recognize such vital features. 

# **Let's have the test data available as well**

# In[16]:


test_df = pd.read_csv(os.path.join(base_dir,'sample_submission.csv'))
test_df.head()


# In[17]:


test_imageid_path = {os.path.basename(x):x for x in glob(os.path.join(test_images,'*.jpg'))}


# In[18]:


test_df['path'] = test_df['id'].map(test_imageid_path.get)


# In[19]:


test_df['images'] = test_df['path'].map(lambda x: np.array(Image.open(x)))


# In[20]:


test_df.head()


# In[21]:


#sample one test image
print("test image's shape: ", test_df['images'][0].shape)


# In[22]:


plt.imshow(test_df['images'][0])


# In[23]:


#check if there is null values in test dataframe:
test_df.isnull().sum()


# In[24]:


#normalize test image pixels
test_df['images'] = test_df['images'] / 255


# In[25]:


test_images = np.asarray(test_df['images'].tolist())


# In[26]:


test_images.shape


# **Create features and labels from training set**

# In[27]:


features = train_df['images']
labels = train_df['has_cactus']


# In[28]:


features = np.asarray(features.tolist())


# In[29]:


labels = np.asarray(labels.tolist())


# **Split train set into train and validation sets**

# In[30]:


x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.1, random_state=1234)


# In[31]:


print("x_train size: ", len(x_train))
print("x_valid size: ", len(x_valid))


# In[32]:


x_train.shape


# Set up the callbacks for training the model later

# In[33]:


callbacks = [ReduceLROnPlateau(monitor= 'val_acc',
                              patience=3,
                              verbose=1,
                              factor=0.5)
#              ,EarlyStopping(monitor='val_acc',
#                            patience=20,
#                           verbose=1,
#                           mode='auto',
#                           restore_best_weights=True)
            ]


# Data Augmentations

# In[34]:


# datagen = ImageDataGenerator(featurewise_center=False,
#                              samplewise_center=False,
#                              featurewise_std_normalization=False,
#                              samplewise_std_normalization=False,
#                              rotation_range=40,
#                              zoom_range=0.3,
#                              width_shift_range=0.3,
#                              height_shift_range=0.3,
#                              vertical_flip=True
#                              #horizontal_flip=True
#                             )


# In[35]:


# datagen.fit(x_train)


# Visualize the training results 

# In[36]:


def visual(model_history):
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    #model accuracy
    acc = model_history.history['acc']
    valid_acc = model_history.history['val_acc']
    
    ax[0].plot(range(1, len(model_history.history['acc'])+1), acc, label='train accuracy')
    ax[0].plot(range(1, len(model_history.history['acc'])+1), valid_acc, label='validation accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()
    
    #model loss
    loss = model_history.history['loss']
    valid_loss = model_history.history['val_loss']
    
    ax[1].plot(range(1, len(model_history.history['acc'])+1), loss, label='train loss')
    ax[1].plot(range(1, len(model_history.history['acc'])+1), valid_loss, label='validation loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()


# **Construct the CNN Model**

# In[37]:


#for cnn model, we are going to use x_train, x_valid, y_train, y_valid
print("x_train's shape: ", x_train.shape)
print("y_train's shape: ", y_train.shape)
print("\nx_valid's shape: ", x_valid.shape)
print("y_valid's shape: ", y_valid.shape)


# In[38]:


input_shape = x_train.shape[1:]

cnn = Sequential([Conv2D(32, kernel_size=(2,2), activation='relu', padding='same', input_shape = input_shape),
                  Conv2D(32, kernel_size=(2,2), activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.2),
                  
                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),
                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.2),
                  
                  Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),
                  Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.5),
                  
#                   Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
#                   Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
#                   MaxPool2D(pool_size=(2,2)),
#                   Dropout(0.5),
                  
                  Flatten(),
                  Dense(64, activation='relu'),
                  Dropout(0.5),
                  Dense(28, activation='relu'),
                  Dropout(0.5),
                  Dense(1, activation='sigmoid')
                 ])


# In[39]:


cnn.summary()


# In[40]:


#compile the cnn model
cnn.compile(optimizer=Adam(lr=0.001), loss = "binary_crossentropy", metrics=['acc'])


# In[41]:


#fit the cnn model
cnn_history = cnn.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_valid,y_valid), callbacks=callbacks)


# In[42]:


# #fit the cnn model with data augmentations
# cnn_history = cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_valid, y_valid),
#                                 steps_per_epoch = x_train.shape[0] // 64,
#                                 callbacks = callbacks)


# In[43]:


visual(cnn_history)


# **Prepare for submission**

# In[44]:


test_pred = cnn.predict_classes(test_images)


# In[45]:


test_df['has_cactus'] = np.squeeze(test_pred)


# In[46]:


test_df.head()


# In[47]:


submission_df = test_df[['id','has_cactus']]


# In[48]:


submission_df.to_csv('cactus_detections.csv', index=False)

