#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle


# In[ ]:


#For Keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[ ]:


image_paths = []
imseg_paths = []
for x in ['dataA', 'dataB', 'dataC', 'dataD', 'dataE']:
    image_path_dir = '../input/lyft-udacity-challenge/' + x + '/' + x + '/' + 'CameraRGB'
    imseg_path_dir = '../input/lyft-udacity-challenge/' + x + '/' + x + '/' + 'CameraSeg'

    for dirname, _, filenames in os.walk(image_path_dir):        
        for filename in filenames:
            image_path = image_path_dir + '/' + filename
            image_paths.append(image_path)
            imseg_path = imseg_path_dir + '/' + filename
            imseg_paths.append(imseg_path) 
            
# Number of images
num_images = len(image_paths)
print("Total number of images = ", num_images)


# In[ ]:


class Args:
    L2_REG = 1e-5
    STDEV = 1e-2
    KEEP_PROB = 0.5
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    BATCH_SIZE = 8
    IMAGE_SHAPE = (512, 768)
    NUM_CLASSES = 1
    SEGMENT = 7 # Segment lable for road

args = Args()


# In[ ]:


def read_image(path):
    image = cv2.imread(path)
    image = image_resize(image)
    return np.array(image)

def read_imseg(path):
    imseg = np.array(cv2.imread(path))
    imseg = image_resize(imseg)
    imseg = np.array([max(imseg[i, j]) for i in range(imseg.shape[0]) for j in range(imseg.shape[1])]).reshape(imseg.shape[0], imseg.shape[1])   
    return imseg

def image_resize(image):
    height, width = args.IMAGE_SHAPE
    return np.array(cv2.resize(image, (width, height), cv2.INTER_AREA))

def imseg2roadseg(imseg):
    height, width = imseg.shape
    imseg_road = np.zeros((height, width, 1), dtype=np.int8)
    imseg_road[np.where(imseg==args.SEGMENT)[0], np.where(imseg==args.SEGMENT)[1]] = 1
    return np.array(imseg_road)

def pipeline(X_path, y_path):
    image_BGR = read_image(X_path)
    imseg = read_imseg(y_path)
    imseg_road = imseg2roadseg(imseg)
    return image_BGR, imseg_road


# In[ ]:


# read all the images
def read_data(image_paths,imseg_paths):
    height, width = args.IMAGE_SHAPE
    images = np.zeros((len(image_paths), height, width, 3), dtype=np.int16)
    imsegs_road = np.zeros((len(image_paths), height, width, 1), dtype=np.int8)
    for index in tqdm(range(len(image_paths))):
        X_path, y_path = image_paths[index], imseg_paths[index]
        images[index], imsegs_road[index] = pipeline(X_path, y_path)
    return images, imsegs_road

X, y = read_data(image_paths,imseg_paths)


# In[ ]:


from random import randint
index = randint(0,len(image_paths))
height, width = args.IMAGE_SHAPE
segment = 7

image = read_image(image_paths[index])
imseg = read_imseg(imseg_paths[index])
imseg_road = imseg2roadseg(imseg)

print(image.shape)
print(imseg.shape)
print(imseg_road.shape)

fig, axes = plt.subplots(1, 3, figsize=(30,20))
axes[0].imshow(image)
axes[0].set_title('RGB Image')
axes[1].imshow(imseg)
axes[1].set_title('Segmented image')
axes[2].imshow(imseg_road.reshape(height,width))
axes[2].set_title('Segmented road image')


# In[ ]:


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle =True,  random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle =True,  random_state=42)

print("Training images = ", len(X_train), "Validation images= ", len(X_val), "Test images = ", len(X_test))


# In[ ]:


def seg_model(image_shape, num_classes):

    vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape, classes=1000)
    #vgg_model.summary()
    
    img_input = vgg_model.layers[0].output

    block3_output = vgg_model.layers[10].output
    block4_output = vgg_model.layers[14].output
    block5_output = vgg_model.layers[18].output
    
    # Freeze the layers except the last 4 layers
    for layer in vgg_model.layers:
        layer.trainable = False
    
    
    # Define the regularizer for the kernel
    kernel_regularizer = tf.keras.regularizers.l2(args.L2_REG)
    
    # Convolutional 1x1 to maintain spacial information of block5_output layer
    block5_conv_1x1 = tf.keras.layers.Conv2D(filters=num_classes, 
                                          kernel_size=1, 
                                          strides=(1, 1), 
                                          padding='same',
                                          kernel_regularizer = kernel_regularizer)(vgg_model.output)

    # Deconv layer
    deconv_block5 = tf.keras.layers.Conv2DTranspose(filters = num_classes,
                                                 kernel_size = 4, 
                                                 strides=(2, 2), 
                                                 padding='same', 
                                                 kernel_regularizer=kernel_regularizer)(block5_conv_1x1)

    
    # Convolutional 1x1 to maintain spacial information of block4_output layer
    block4_conv_1x1 = tf.keras.layers.Conv2D(filters=num_classes, 
                                          kernel_size=1, 
                                          strides=(1, 1), 
                                          padding='same',
                                          kernel_regularizer = kernel_regularizer)(block4_output)

    skip_connection_1 = tf.keras.layers.Add()([deconv_block5, block4_conv_1x1])
    
    # Deconv layer
    deconv_layer_4_5 = tf.keras.layers.Conv2DTranspose(filters = num_classes,
                                                 kernel_size = 4, 
                                                 strides=(2, 2), 
                                                 padding='same', 
                                                 kernel_regularizer=kernel_regularizer)(skip_connection_1)
    
    
    # Convolutional 1x1 to maintain spacial information of vgg_layer 3
    block3_conv_1x1 = tf.keras.layers.Conv2D(filters=num_classes, 
                                          kernel_size=1, 
                                          strides=(1, 1), 
                                          padding='same',
                                          kernel_regularizer = kernel_regularizer)(block3_output)
    
    skip_connection_2 = tf.keras.layers.Add()([deconv_layer_4_5, block3_conv_1x1])
    
    output = tf.keras.layers.Conv2DTranspose(filters = num_classes,
                                                 kernel_size = 16, 
                                                 strides=(8, 8), 
                                                 padding='same', 
                                                 kernel_regularizer=kernel_regularizer)(skip_connection_2)
    
    predictions = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(output)
    model = tf.keras.Model(inputs = vgg_model.input, outputs = predictions, name='seg_model')
    return model


# In[ ]:


def mirror(x):
    return x[:,::-1,:]

def hsv_augment(x):
    x_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    x_hsv[:, :, 0] = x_hsv[:, :, 0] * random.uniform(0.9, 1.1)   # change hue
    x_hsv[:, :, 1] = x_hsv[:, :, 1] * random.uniform(0.5, 2.0)   # change saturation
    x_hsv[:, :, 2] = x_hsv[:, :, 2] * random.uniform(0.5, 2.0)   # change brightness
    x_hsv = np.clip(x_hsv, 0, 255)  
    return cv2.cvtColor(x_hsv, cv2.COLOR_HSV2BGR)

def augment(image, imseg_road):
    if np.random.rand() < 0.6:
        image = mirror(image)
        imseg_road = mirror(imseg_road)
    if np.random.rand() < 0.6:
        image = hsv_augment(image)
    return image, imseg_road


# In[ ]:


# Define batch generator
def batch_generator(X, y, args, is_training):
    """
    Generate training images 
    X: image paths
    y: segmentation paths
    is_training: True for training, False for inference
    """

    
    if is_training:
        print("Model is training")
    while True:
        for index in np.random.permutation(len(X)):
            height, width = args.IMAGE_SHAPE
            images = np.empty([args.BATCH_SIZE, height, width, 3])
            segments = np.empty([args.BATCH_SIZE, height, width, 1])
            for i in range(args.BATCH_SIZE):           
                image, imseg_road = X[index], y[index]
                # augmentation
                if is_training and np.random.rand() < 0.6:
                    image, imseg_road = augment(image, imseg_road)

                images[i] = image
                segments[i] = imseg_road
            yield images, segments


# In[ ]:


from tensorflow.keras.models import Model
image_shape = (args.IMAGE_SHAPE[0], args.IMAGE_SHAPE[1], 3)
num_classes = args.NUM_CLASSES
my_model = seg_model(image_shape, num_classes)

#compile the model
my_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

my_model.summary()


# In[ ]:


#train your model on data
checkpoints = tf.compat.v1.keras.callbacks.ModelCheckpoint("./models/model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



my_model.fit_generator(batch_generator(X_train, y_train, args, True),
                    steps_per_epoch=args.BATCH_SIZE*300,
                    epochs=5,
                    validation_data=batch_generator(X_val, y_val, args, False),
                    validation_steps=len(X_val),
                    callbacks = [checkpoints],
                    verbose=1, shuffle=True)

