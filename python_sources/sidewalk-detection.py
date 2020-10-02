#!/usr/bin/env python
# coding: utf-8

# # Preparing and import library

# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[ ]:


import tensorflow as tf
import keras as K

import sklearn
from sklearn.metrics import jaccard_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input
from tensorflow.keras.models import Model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import numpy as np
import os
import cv2
import imutils
import random
from PIL import Image
import matplotlib.pyplot as plt


# # Preparing and import data

# In[ ]:


# Setting data path

DATA_PATH = '../input/sidewalk-dataset-straight-sidewalk-only/Dataset_Complete'                            #Root path of data

train_frame_path = DATA_PATH+'/Train_set/Raw_data/'
train_mask_path = DATA_PATH+'/Train_set/'

test_frame_path = DATA_PATH+'/Test_set/Raw_data/'
test_mask_path = DATA_PATH+'/Test_set/'


# Necessary variable

# In[ ]:


seed = 1                                                                                         # Seed we want to use (You can change it if you like)

BATCH_SIZE = 2   # Batch size of generator
IMAGE_SIZE = 256

train_samples = len(os.listdir(train_frame_path))                                                # Amount of train samples
test_samples = len(os.listdir(test_frame_path))                                                    # Amount of validation samples


# In[ ]:


# Check amount of them

print(train_samples,test_samples)


# In[ ]:


def Load_image():
    
    new_path = []
    
    TYPE_OF_INPUT = input("Which sidewalk you want to train? (Straight, Left, Right): ")
    img_path = os.listdir(train_frame_path)                                                                       # List training image name for reading them
    mask_path = os.listdir(train_mask_path +TYPE_OF_INPUT+'_sidewalk/')                                           # List training mask name for reading them
    
    for label in mask_path:
        new_path.append(label.split('_')[1])                                                                      # Select only file number for loading image (Ex. Label_0 -> 0)

    new_path.sort()                                                                                               # Sorting training image's and mask's name

    # Training Set

    train_img = np.zeros((len(new_path), 640, 360, 1)).astype('float')                                            # Create empty array for substitute with train image
    train_mask = np.zeros((len(new_path), 640, 360, 1)).astype('float')                                           # Create empty array for substitute with train mask

    for i in range(0, len(new_path)):                                                                             # Loop for reading image according from amount of them

        train_img1 = cv2.imread(train_frame_path+'Raw_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.                    # Reading Raw_0.jpg ... and So on.

        print(train_frame_path+'Raw_'+new_path[i])                                                                       # Checking image name and path                //using for compare with mask for make sure that it's the correct image and mask

        if train_img1.shape != (1920,1080):                                                                       # If image is landscrape then rotate it to potrait.
            train_img1 = imutils.rotate_bound(train_img1, 90)

        train_img1 =  cv2.resize(train_img1, (360, 640))                                                          # Read an image from folder and resize
        train_img1 = train_img1.reshape(640,360,1)                                                                # Give them the channel of image              //make sure that your image not import with grayscale initialy
        train_img[i] = train_img1                                                                                 # Add to array - img[0], img[1], and so on.

        train_mask1 = cv2.imread(train_mask_path+TYPE_OF_INPUT+'_sidewalk/'+'Label_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.

        print(train_mask_path+'Label_'+new_path[i])

        if train_mask1.shape != (1920,1080): 
            train_mask1 = imutils.rotate_bound(train_mask1, 90)

        train_mask1 = cv2.resize(train_mask1,(360,640))
        train_mask1 = train_mask1.reshape(640, 360, 1)                                                            # Add extra dimension for parity with train_img size [512 * 512 * 3]

        train_mask[i] = train_mask1

        print("Image : {}".format(i))
        
    return train_img, train_mask


# In[ ]:


train_img, train_mask = Load_image()


# In[ ]:


# Checking if the images fit

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(train_img.shape)

print(len(train_img))

plt.imshow(train_img[4].reshape(640,360))
plt.show()

plt.imshow(train_mask[4].reshape(640,360))
plt.show()


# # Creating the Image generator
# 
# If you have a lot of data, when you try to fitting the model, your memory will exceed.
# 
# So, we have to put the data which divide into small amount and take them off from memory which called generator.

# In[ ]:


from keras.preprocessing import image

val_img = train_img[243:]
val_mask = train_mask[243:]

# train_img = train_img.tolist()
# train_mask = train_mask.tolist()

# del train_img[51:68]
# del train_mask[51:68]

# train_img = np.array(train_img)
# train_mask = np.array(train_mask)

print(len(train_img),len(train_mask))

'''
aug = ImageDataGenerator(
        blurring=3, 
		rotation_range=5,
		zoom_range=0.3,
		width_shift_range=0.3,
		height_shift_range=0.3,
		shear_range=0.15,
		fill_mode="nearest")
'''

# Creating the training Image and Mask generator
train_image_datagen = image.ImageDataGenerator(shear_range=0.5, zoom_range=0.3,fill_mode='nearest')
train_mask_datagen = image.ImageDataGenerator(shear_range=0.5,zoom_range=0.3,fill_mode='nearest')
# Keep the same seed for image and mask generators so they fit together

train_image_datagen.fit(train_img,seed=seed,augment=True)
train_mask_datagen.fit(train_mask,seed=seed,augment=True)

train_img=train_image_datagen.flow(train_img,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
train_mask=train_mask_datagen.flow(train_mask,batch_size=BATCH_SIZE,shuffle=True, seed=seed)



# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(val_img, augment=True)
mask_datagen_val.fit(val_mask, augment=True)

val_img=image_datagen_val.flow(val_img,batch_size=BATCH_SIZE,shuffle=False, seed=seed)
val_mask=mask_datagen_val.flow(val_mask,batch_size=BATCH_SIZE,shuffle=False, seed=seed)


# In[ ]:


print(np.squeeze(train_img[5]).shape)


# In[ ]:


# Checking if the images fit

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(len(train_img))

print(len(train_img[0]))

plt.imshow(np.squeeze(train_img[20][1]))
plt.show()
plt.imshow(np.squeeze(train_mask[20][1]))
plt.show()

plt.imshow(np.squeeze(val_img[2][1]))
plt.show()
plt.imshow(np.squeeze(val_mask[2][1]))
plt.show()


# In[ ]:


# Zipping the file to make function see only one variable

train_generator = zip(train_img,train_mask)
val_generator = zip(val_img, val_mask)


# # Training

# In[ ]:


with tf.device('/device:GPU:0'):                                                                                          # Initialize process to GPU
  def dice_coef(y_true, y_pred, smooth=1):                                                                                # Dice coefficient using for validate predict image to truth mask.
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.abs(K.backend.sum(y_true * y_pred))
    union = K.backend.abs(K.backend.sum(y_true_f)) + K.backend.abs(K.backend.sum(y_pred_f))
    dice = K.backend.mean((2. * intersection + smooth)/(union + smooth))                                                  # Dice coefficient equation : Dice = 2*abs(intersection)/abs(union)   //smooth using for make model learning easier
    return dice

  def dice_coef_loss(y_true, y_pred):                                                                                     # Using dice coeffiecient as a loss function                          // Loss is alike to error of the model
      return 1 - dice_coef(y_true, y_pred)


  model = Sequential()                                                                                                    # H

  #1st Layer
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform', input_shape=(640, 360, 1)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=2))

  #1st Layer
  model.add(Conv2D(128, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=2))

  #1st Layer
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=2))


  #Upsampling Part
  model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())

  #Upsampling Part
  model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())

  #Upsampling Part
  model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
  model.add(Conv2D(128, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())

  #Upsampling Part
  model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())

  #Upsampling Part
  model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (17, 6), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (17, 10), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (17, 6), activation='relu', kernel_initializer='he_uniform'))
  model.add(BatchNormalization())

  model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_uniform'))
  

  
#   model = tf.keras.models.load_model('../input/sidewalk-dataset-straight-sidewalk-only/left_sidewalk_model_augmented_v1_early10.h5')

#   model.load_weights('../input/sidewalk-dataset-straight-sidewalk-only/right_sidewalk_model_augmented_v1_early10_k3.h5')
  model.compile(optimizer = 'adam', loss=dice_coef_loss, metrics=[dice_coef])
    
  tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs/")
  earlystopper = EarlyStopping(patience=10, verbose=1)
  csv_logger = CSVLogger('straight-final.csv', append=True, separator=';')
  checkpointer = ModelCheckpoint('./straight_sidewalk_model_augmented_v1_early10.h5', verbose=1, save_best_only=True)

fitting = model.fit(train_generator,
                    epochs=100,
                    steps_per_epoch=len(train_img),
                    validation_data=val_generator,
                    validation_steps=len(val_img),
                    callbacks= [tensorboard_callback, csv_logger, checkpointer])


# In[ ]:


model.save('./straight_sidewalk_model_augmented_v1_early10_final.h5')
model.save_weights('./straight_sidewalk_model_augmented_v1_early10_final.hdf5')


# # Using MobileNetV2 with ImageNet weight.
# 
# MobileNetV2 is only 14 mb size. It is identical model for sidewalk segmentation in my mind.

# In[ ]:


def Load_image():
    
    new_path = []
    
    TYPE_OF_INPUT = input("Which sidewalk you want to train? (Straight, Left, Right): ")
    img_path = os.listdir(train_frame_path)                                                                       # List training image name for reading them
    mask_path = os.listdir(train_mask_path +TYPE_OF_INPUT+'_sidewalk/')                                           # List training mask name for reading them
    
    for label in mask_path:
        new_path.append(label.split('_')[1])                                                                      # Select only file number for loading image (Ex. Label_0 -> 0)

    new_path.sort()                                                                                               # Sorting training image's and mask's name

    print(len(new_path))
    # Training Set

    train_img = np.zeros((len(new_path), 256, 256, 3)).astype('float')                                            # Create empty array for substitute with train image
    train_mask = np.zeros((len(new_path), 256, 256, 1)).astype('float')                                           # Create empty array for substitute with train mask

    for i in range(0, len(new_path)):                                                                             # Loop for reading image according from amount of them

        train_img1 = cv2.imread(train_frame_path+'Raw_'+new_path[i])/255.                    # Reading Raw_0.jpg ... and So on.

        print(train_frame_path+'Raw_'+new_path[i])                                                                       # Checking image name and path                //using for compare with mask for make sure that it's the correct image and mask

        if train_img1.shape != (1920,1080):                                                                       # If image is landscrape then rotate it to potrait.
            train_img1 = imutils.rotate_bound(train_img1, 90)

        train_img1 =  cv2.resize(train_img1, (256, 256))                                                          # Read an image from folder and resize
        train_img1 = train_img1.reshape(256,256,3)                                                                # Give them the channel of image              //make sure that your image not import with grayscale initialy
        train_img[i] = train_img1                                                                                 # Add to array - img[0], img[1], and so on.

        train_mask1 = cv2.imread(train_mask_path+TYPE_OF_INPUT+'_sidewalk/'+'Label_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.

        print(train_mask_path+'Label_'+new_path[i])

        if train_mask1.shape != (1920,1080): 
            train_mask1 = imutils.rotate_bound(train_mask1, 90)

        train_mask1 = cv2.resize(train_mask1,(256, 256))
        train_mask1 = train_mask1.reshape(256, 256, 1)                                                            # Add extra dimension for parity with train_img size [512 * 512 * 3]

        train_mask[i] = train_mask1

        print("Image : {}".format(i))
        
    return train_img, train_mask


# In[ ]:


train_img, train_mask = Load_image()


# In[ ]:


# Checking if the images fit

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(train_img.shape)

print(len(train_img))

plt.imshow(train_img[4].reshape(640,360))
plt.show()

plt.imshow(train_mask[4].reshape(640,360))
plt.show()


# In[ ]:


from keras.preprocessing import image

val_img = train_img[243:]
val_mask = train_mask[243:]

# train_img = train_img.tolist()
# train_mask = train_mask.tolist()

# del train_img[51:68]
# del train_mask[51:68]

# train_img = np.array(train_img)
# train_mask = np.array(train_mask)

print(len(train_img),len(train_mask))

'''
aug = ImageDataGenerator(
        blurring=3, 
		rotation_range=5,
		zoom_range=0.3,
		width_shift_range=0.3,
		height_shift_range=0.3,
		shear_range=0.15,
		fill_mode="nearest")
'''

# Creating the training Image and Mask generator
train_image_datagen = image.ImageDataGenerator(shear_range=0.5, zoom_range=0.3,fill_mode='nearest')
train_mask_datagen = image.ImageDataGenerator(shear_range=0.5,zoom_range=0.3,fill_mode='nearest')
# Keep the same seed for image and mask generators so they fit together

train_image_datagen.fit(train_img,seed=seed,augment=True)
train_mask_datagen.fit(train_mask,seed=seed,augment=True)

train_img=train_image_datagen.flow(train_img,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
train_mask=train_mask_datagen.flow(train_mask,batch_size=BATCH_SIZE,shuffle=True, seed=seed)



# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(val_img, augment=True)
mask_datagen_val.fit(val_mask, augment=True)

val_img=image_datagen_val.flow(val_img,batch_size=BATCH_SIZE,shuffle=False, seed=seed)
val_mask=mask_datagen_val.flow(val_mask,batch_size=BATCH_SIZE,shuffle=False, seed=seed)


# In[ ]:


# Zipping the file to make function see only one variable

train_generator = zip(train_img,train_mask)
val_generator = zip(val_img, val_mask)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


with tf.device('/device:GPU:0'):                # Initialize process to GPU
    def build_model():
        inputs = Input(shape=[IMAGE_SIZE, IMAGE_SIZE,3], name='input_image')

        #Pretrained Encoder
        encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False,alpha=0.35)
        skip_connection_names = ['input_image','block_1_expand_relu','block_3_expand_relu','block_6_expand_relu']
    #     encoder.summary()
        encoder_output = encoder.get_layer('block_13_expand_relu').output

        f = [16,32,48,64]
        x = encoder_output

        for i in range(1,len(f)+1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2,2))(x)
            x = Concatenate()([x,x_skip])

            x = Conv2D(f[-i], (3,3),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(f[-i], (3,3),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = Conv2D(1, (1,1),padding='same')(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs, x)
        return model
    
    def dice_coef(y_true, y_pred, smooth=1):                                                                                # Dice coefficient using for validate predict image to truth mask.
        y_true_f = K.backend.flatten(y_true)
        y_pred_f = K.backend.flatten(y_pred)
        intersection = K.backend.abs(K.backend.sum(y_true * y_pred))
        union = K.backend.abs(K.backend.sum(y_true_f)) + K.backend.abs(K.backend.sum(y_pred_f))
        dice = K.backend.mean((2. * intersection + smooth)/(union + smooth))                                                  # Dice coefficient equation : Dice = 2*abs(intersection)/abs(union)   //smooth using for make model learning easier
        return dice

    def dice_coef_loss(y_true, y_pred):                                                                                     # Using dice coeffiecient as a loss function                          // Loss is alike to error of the model
        return 1 - dice_coef(y_true, y_pred)

    model = build_model()
    model.compile(optimizer = 'adam', loss=dice_coef_loss, metrics=[dice_coef])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs/")
    earlystopper = EarlyStopping(patience=10, verbose=1)
    csv_logger = CSVLogger('straight-final-mobilenetv2.csv', append=True, separator=';')
    checkpointer = ModelCheckpoint('./straight_sidewalk_model_augmented_v1_early10_mobilenetv2.h5', verbose=1, save_best_only=True)

fitting = model.fit(train_generator,
                    epochs=1000,
                    steps_per_epoch=len(train_img),
                    validation_data=val_generator,
                    validation_steps=len(val_img),
                    callbacks= [tensorboard_callback, csv_logger, checkpointer])
    


# In[ ]:


while(True):
    print('Waiting for peat.')


# # Don't use

# In[ ]:





# In[ ]:


model()


# In[ ]:


import random

random.seed( 1 )
print ("first number  - ", random.randint(25,50))  

# random.seed( 1 )
print ("Second number- ", random.randint(25,50))


# In[ ]:


import random

random.seed( 1 )
print ("first number  - ", random.randint(25,50))  

random.seed( 1 )
print ("Second number- ", random.randint(25,50))


# In[ ]:


print(train_img[0].shape)


# In[ ]:


train_pred = model.predict(train_img)
print(train_pred.shape)
fig = plt.figure()
ax1 = fig.add_subplot(231,title='train_img')
ax1.imshow(train_img[0].reshape(224,224,3))

ax2 = fig.add_subplot(232,title='train_pred')
ax2.imshow(train_pred[0].reshape(10,100))

ax3 = fig.add_subplot(233,title='train_mask')
ax3.imshow(train_mask[0].reshape(224,224))

plt.show


# In[ ]:


val_pred = model.predict(val_img[0])

fig = plt.figure()
ax1 = fig.add_subplot(231,title='val_img')
ax1.imshow(val_img[0][0])

ax2 = fig.add_subplot(232,title='val_pred')
ax2.imshow(val_pred[0].reshape(640,360))

ax3 = fig.add_subplot(233,title='val_mask')
ax3.imshow(val_mask[0][0].reshape(640,360))

plt.show


# In[ ]:


get_ipython().system('cat ../input/sidewalk-dataset-straight-sidewalk-only/left_sidewalk_model_augmented_v1_early10.h5')


# In[ ]:


def Load_image():
    
    new_path = []
    
    TYPE_OF_INPUT = input("Which sidewalk you want to train? (Straight, Left, Right): ")
    img_path = os.listdir(test_frame_path)                                                                       # List training image name for reading them
    mask_path = os.listdir(test_mask_path +TYPE_OF_INPUT+'_sidewalk/')                                           # List training mask name for reading them
    
    for label in mask_path:
        new_path.append(label.split('_')[1])                                                                      # Select only file number for loading image (Ex. Label_0 -> 0)

    new_path.sort()                                                                                               # Sorting training image's and mask's name

    # Training Set

    train_img = np.zeros((len(new_path), 640, 360, 1)).astype('float')                                            # Create empty array for substitute with train image
    train_mask = np.zeros((len(new_path), 640, 360, 1)).astype('float')                                           # Create empty array for substitute with train mask

    for i in range(0, len(new_path)):                                                                             # Loop for reading image according from amount of them

        train_img1 = cv2.imread(test_frame_path+'Raw_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.                    # Reading Raw_0.jpg ... and So on.

        print(test_frame_path+'Raw_'+new_path[i])                                                                       # Checking image name and path                //using for compare with mask for make sure that it's the correct image and mask

        if train_img1.shape != (1920,1080):                                                                       # If image is landscrape then rotate it to potrait.
            train_img1 = imutils.rotate_bound(train_img1, 90)

        train_img1 =  cv2.resize(train_img1, (360, 640))                                                          # Read an image from folder and resize
        train_img1 = train_img1.reshape(640,360,1)                                                                # Give them the channel of image              //make sure that your image not import with grayscale initialy
        train_img[i] = train_img1                                                                                 # Add to array - img[0], img[1], and so on.

        train_mask1 = cv2.imread(test_mask_path+TYPE_OF_INPUT+'_sidewalk/'+'Label_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.

        print(train_mask_path+'Label_'+new_path[i])

        if train_mask1.shape != (1920,1080): 
            train_mask1 = imutils.rotate_bound(train_mask1, 90)

        train_mask1 = cv2.resize(train_mask1,(360, 640))
        train_mask1 = train_mask1.reshape(640,360, 1)                                                            # Add extra dimension for parity with train_img size [512 * 512 * 3]

        train_mask[i] = train_mask1

        print("Image : {}".format(i))
        
    return train_img, train_mask


# In[ ]:


def Load_image():
    
    new_path = []
    
    TYPE_OF_INPUT = input("Which sidewalk you want to train? (Straight, Left, Right): ")
    img_path = os.listdir(test_frame_path)                                                                       # List training image name for reading them
    mask_path = os.listdir(test_mask_path +TYPE_OF_INPUT+'_sidewalk/')                                           # List training mask name for reading them
    
    for label in mask_path:
        new_path.append(label.split('_')[1])                                                                      # Select only file number for loading image (Ex. Label_0 -> 0)

    new_path.sort()                                                                                               # Sorting training image's and mask's name

    # Training Set

    train_img = np.zeros((len(new_path), 256, 256, 3)).astype('float')                                            # Create empty array for substitute with train image
    train_mask = np.zeros((len(new_path), 256, 256, 1)).astype('float')                                           # Create empty array for substitute with train mask

    for i in range(0, len(new_path)):                                                                             # Loop for reading image according from amount of them

        train_img1 = cv2.imread(test_frame_path+'Raw_'+new_path[i])/255.                    # Reading Raw_0.jpg ... and So on.

        print(test_frame_path+'Raw_'+new_path[i])                                                                       # Checking image name and path                //using for compare with mask for make sure that it's the correct image and mask

        if train_img1.shape != (1920,1080):                                                                       # If image is landscrape then rotate it to potrait.
            train_img1 = imutils.rotate_bound(train_img1, 90)

        train_img1 =  cv2.resize(train_img1, (256, 256))                                                          # Read an image from folder and resize
        train_img1 = train_img1.reshape(256,256,3)                                                                # Give them the channel of image              //make sure that your image not import with grayscale initialy
        train_img[i] = train_img1                                                                                 # Add to array - img[0], img[1], and so on.

        train_mask1 = cv2.imread(test_mask_path+TYPE_OF_INPUT+'_sidewalk/'+'Label_'+new_path[i],cv2.IMREAD_GRAYSCALE)/255.

        print(train_mask_path+'Label_'+new_path[i])

        if train_mask1.shape != (1920,1080): 
            train_mask1 = imutils.rotate_bound(train_mask1, 90)

        train_mask1 = cv2.resize(train_mask1,(256, 256))
        train_mask1 = train_mask1.reshape(256, 256, 1)                                                            # Add extra dimension for parity with train_img size [512 * 512 * 3]

        train_mask[i] = train_mask1

        print("Image : {}".format(i))
        
    return train_img, train_mask


# In[ ]:


test_img, test_mask = Load_image()


# In[ ]:


model.evaluate(test_img, test_mask)


# In[ ]:


train_pred = model.predict(test_img)

fig = plt.figure()
ax1 = fig.add_subplot(231,title='test_img')
ax1.imshow(test_img[1].reshape(256,256,3))

ax2 = fig.add_subplot(232,title='test_pred')
ax2.imshow(train_pred[1].reshape(256,256))

ax3 = fig.add_subplot(233,title='test_mask')
ax3.imshow(test_mask[1].reshape(256,256))

plt.show


# In[ ]:


train_pred = model.predict(test_img)

fig = plt.figure()
ax1 = fig.add_subplot(231,title='test_img')
ax1.imshow(test_img[1].reshape(640,360))

ax2 = fig.add_subplot(232,title='test_pred')
ax2.imshow(train_pred[1].reshape(640,360))

ax3 = fig.add_subplot(233,title='test_mask')
ax3.imshow(test_mask[1].reshape(640,360))

plt.show


# In[ ]:


# Validation set

img_path = os.listdir(val_frame_path)
mask_path = os.listdir(val_mask_path)

img_path.sort()
mask_path.sort()

val_img = np.zeros((val_samples, 640, 360, 1)).astype('float')
val_mask = np.zeros((val_samples, 640, 360, 1)).astype('float')

for i in range(0, val_samples): #initially from 0 to 16, c = 0. 

    val_img1 = cv2.imread(val_frame_path+img_path[i],cv2.IMREAD_GRAYSCALE)/255.

    print(val_frame_path+img_path[i])

    if val_img1.shape != (1920,1080): 
        val_img1 = imutils.rotate_bound(val_img1, 90)

    val_img1 =  cv2.resize(val_img1, (360, 640))# Read an image from folder and resize
    val_img1 = val_img1.reshape(640,360,1)
    val_img[i] = val_img1 #add to array - img[0], img[1], and so on.


    val_mask1 = cv2.imread(val_mask_path+mask_path[i],cv2.IMREAD_GRAYSCALE)

    print(val_mask_path+mask_path[i])

    if val_mask1.shape != (1920,1080,1): 
        val_mask1 = imutils.rotate_bound(val_mask1, 90)

    val_mask1 =  cv2.resize(val_mask1, (360, 640))# Read an image from folder and resize
    val_mask1 = val_mask1.reshape(640, 360, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

    val_mask[i] = val_mask1

    print("Image : {}".format(i))


# # Tensorboard showing 
# (You need to open the link below)

# In[ ]:


# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip
LOG_DIR = './logs/' # Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


from IPython.display import FileLink
FileLink('./right_sidewalk_model_augmented_v1_early10_final.h5')

