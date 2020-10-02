#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#loading modules
import math, random, os, re, time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
import gc
import cv2
import seaborn as sns
from kaggle_datasets import KaggleDatasets
from tqdm import tqdm


# In[ ]:


#loading data
dirname='../input/siim-isic-melanoma-classification/'
train = pd.read_csv(dirname+'train.csv')
test = pd.read_csv(dirname + 'test.csv')
print(train.head())
print(len(train))
print(len(test))
print(train['target'].value_counts())


# In[ ]:


sns.countplot(train['target'])


# In[ ]:


#sampling dataset: 5000 target 0, 584 target 1
df_0 = train[train['target']==0].sample(5000)
df_1 = train[train['target']==1]
train = pd.concat([df_0, df_1])
train = train.reset_index()
del df_0
del df_1
print(len(train))
sns.countplot(train['target'])


# In[ ]:


#function to resize images
def right_size(arr):
    arr = cv2.resize(arr, (256,256))
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


# In[ ]:


#show one image
arr = cv2.imread(dirname + 'jpeg/' + 'train/' + train['image_name'].iloc[0] + '.jpg')
plt.imshow(right_size(arr))
print(arr.shape)


# In[ ]:


#show 10 images, 1st row is target 0, second row is target 1
fig = plt.figure(figsize=(15,10))
columns = 5
rows = 2
for i in [0,1]:
    df = train[train['target']==i].sample(5)
    df = list(df['image_name'])
    for j in range(5):
        fig.add_subplot(rows, columns, i*columns + j + 1)
        plt.imshow(right_size(cv2.imread(dirname + 'jpeg/train/' + df[j] + '.jpg')))
    del df


# In[ ]:


# prepare paths for training and validation data
data = []
target = []
for i in range(len(train)):
    data.append(dirname + 'jpeg/train/' + train['image_name'].iloc[i] + '.jpg')
    target.append(train['target'].iloc[i])

#prepare dataframe for test data
test_data = []
for i in range(len(test)):
    test_data.append(dirname + 'jpeg/test/' + test['image_name'].iloc[i] + '.jpg')
test_path = pd.DataFrame(test_data)
test_path.columns = ['images']


# In[ ]:


# split train and validation data, turn into dataframes
train_X, val_X, train_Y, val_Y = train_test_split(data, target, test_size = 0.2, random_state = 1)

train = pd.DataFrame(train_X)
train.columns = ['images']
train['target'] = train_Y

val = pd.DataFrame(val_X)
val.columns = ['images']
val['target'] = val_Y


# In[ ]:


#create input pipeline through flow_from_dataframe
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True,vertical_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train, x_col='images', y_col='target', 
                                                   target_size = (256,256), batch_size=10, shuffle=True, class_mode = 'raw')
val_generator = val_datagen.flow_from_dataframe(val, x_col='images', y_col='target', 
                                                   target_size = (256,256), batch_size=10, shuffle=False, class_mode = 'raw')


# In[ ]:


def res_block(X_in, channels):
    X = layers.Conv2D(channels, (3,3), strides=(1,1), padding='same' )(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    
    X = layers.Conv2D(channels, (3,3), strides=(1,1), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([X, X_in])
    X = layers.LeakyReLU()(X)
    
    return X


# In[ ]:


#model
def my_model():
    X_in = layers.Input((256, 256, 3))
    
    X = layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv1')(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)

    X = res_block(X, 64)
    
    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name='max_pool1')(X)

    X = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv2')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    
    X = res_block(X, 128)
    
    X = layers.MaxPool2D(pool_size = (2,2), strides=2, name='max_pool2')(X)

    X = layers.Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv3')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)

    X = res_block(X, 256)
    
    X = layers.MaxPool2D(pool_size = (2,2), strides=2, name='max_pool3')(X)

    X = res_block(X, 256)
    
    X = layers.MaxPool2D(pool_size = (2,2), strides=2, name='max_pool4')(X)
    
    X = res_block(X, 256)
    
    X = layers.MaxPool2D(pool_size = (2,2), strides=2, name='max_pool5')(X)
    
    X = layers.Flatten()(X)
    X = layers.Dense(4096, activation='relu', name='fc1')(X)
    X = layers.Dense(1024, activation='relu', name='fc2')(X)
    X_out = layers.Dense(1, activation='sigmoid', name='answer')(X)

    model = Model(inputs=X_in, outputs=X_out, name='pinnet')
    
    return model


# In[ ]:


model = my_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model = tf.keras.models.load_model('../input/pinnet/pinnet')


# In[ ]:


model.summary()


# In[ ]:


tf.test.is_gpu_available()


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
def callback():
    cb = []
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3, patience=5,
                                   verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=1, min_lr=0.00001)
    cb.append(reduceLROnPlat)
    return cb


# In[ ]:


cb = callback()
#train and validate
epochs = 5
history = model.fit(train_generator, steps_per_epoch = train.shape[0]//10, epochs = epochs,
                    validation_data = val_generator, validation_steps = val.shape[0]//10, callbacks=cb) # callbacks = cb,


# In[ ]:


model.save('pinnet')


# In[ ]:


print(history.history.keys())


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#test data input pipeline
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(test_path, x_col='images', y_col=None, 
                                                   target_size = (256,256), batch_size=10, shuffle=False, class_mode=None)
test_generator.reset()


# In[ ]:


#predict on test data
preds = model.predict(test_generator, steps=test.shape[0]//10+1)
ans = np.array(preds)
print(ans.shape)


# In[ ]:


#prep recorded targets
ans=list(ans)
for i in range(len(ans)):
    ans[i]=ans[i][0]


# In[ ]:


#turn predictions in required format
final = {'image_name':list(test['image_name']), 'target':ans }

sub = pd.DataFrame(final, columns=['image_name', 'target'])
print(sub.head())
print(sub.describe())


# In[ ]:


#save predictions
sub.to_csv('submission.csv', header=True, index=False)


# In[ ]:




