#!/usr/bin/env python
# coding: utf-8

# # Using darkfield microscope images, highlight regions with bacteria
# 
# This kernel uses a lot of learnings from Long Nguyen's [kernel](https://www.kaggle.com/longnguyen2306/unet-plus-plus-for-image-segmentation) (class imbalance, model architecture) to facilitate my learning with these differences.

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Constants

# In[ ]:


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 16
NUM_CLASSES = 3
IMG_PATH = './images/'
MASK_PATH = './masks/'
LABEL_PATH = '../input/bacteria-detection-with-darkfield-microscopy/masks/'
IMG_SUB_PATH = './images/images/'
MASK_SUB_PATH = './masks/masks/'


# ## Copy data over to output file system
# Keras data generator expects the images to be in a subfolder

# In[ ]:


os.system('mkdir ./images/')
os.system('cp -r ../input/bacteria-detection-with-darkfield-microscopy/images/ ./images/')
os.system('mkdir ./masks/')
os.system('cp -r ../input/bacteria-detection-with-darkfield-microscopy/masks/ ./masks/')


# Because this is a multi class segmentation task, each mask needs to be one hot encoded

# In[ ]:


mask_files = os.listdir(MASK_SUB_PATH)
for mf in tqdm (mask_files):
    mask_img = cv2.imread(os.path.join(MASK_SUB_PATH, mf), cv2.IMREAD_GRAYSCALE)
    mask_img = np.around(tf.keras.utils.to_categorical(mask_img, NUM_CLASSES))
    cv2.imwrite(os.path.join(MASK_SUB_PATH, mf), mask_img)


# ## Helper functions
# Useful functions for any project

# In[ ]:


def show_img(img, title=''):
    # given a numpy array, plot it
    vis = plt.imshow(img)
    plt.title(title)
    plt.show()
    
def show_imgs(imgs, titles=None):
    # show two images side by side, useful for segmentation projects
    fig = plt.figure(figsize=(15, 15))

    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        if titles:
            plt.title(titles[i])
        plt.imshow(imgs[i])
    
    plt.show()

def label_to_image(m):
    # given a mask, turn it into an image, use for binary segmentations
    return tf.keras.preprocessing.image.array_to_img(m.reshape((m.shape[0], m.shape[1], 1)))

def output_to_image(o):
    # given model output o that is one hot encoded, turn it into an image, use for multi class segmentations
    mask_im = tf.argmax(o, axis=-1)
    mask_im = tf.keras.preprocessing.image.array_to_img(mask_im[..., tf.newaxis])
    
    return mask_im


# In[ ]:


def visualise_source(title): 
    # given an image title, visualise the source data
    test_img = cv2.imread(os.path.join(IMG_SUB_PATH, title + '.png'))
    mask_img = cv2.imread(os.path.join(LABEL_PATH, title + '.png'), cv2.IMREAD_GRAYSCALE)
    show_imgs([test_img, mask_img])


# ## Have a look at an image

# In[ ]:


visualise_source('003')


# ## Use TPU/GPU

# In[ ]:


# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ## Load data
# Load data into Keras generators and also apply some augmentation to it. The image and masks are loaded separately with the same seed and some augmentations are applied

# In[ ]:


# https://github.com/keras-team/keras/issues/3059#issuecomment-364787723
training_generation_args = dict(
#     width_shift_range=0.3,
#     height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    validation_split=0.1
)
train_image_datagen = ImageDataGenerator(**training_generation_args)
train_label_datagen = ImageDataGenerator(**training_generation_args)

# data load
training_image_generator = train_image_datagen.flow_from_directory(
    IMG_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    seed=1
)
training_label_generator = train_label_datagen.flow_from_directory(
    MASK_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    # color_mode='grayscale',
    seed=1
)


# validation data load
validation_image_generator = train_image_datagen.flow_from_directory(
    IMG_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    seed=1
)
validation_label_generator = train_label_datagen.flow_from_directory(
    MASK_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    # color_mode='grayscale',
    seed=1
)

train_generator = zip(training_image_generator, training_label_generator)
validation_generator = zip(validation_image_generator, validation_label_generator)


# Have a look at the dataset and make sure there's no issues

# In[ ]:


test_imgs, labels = train_generator.__next__()
show_imgs([test_imgs[0] / 255., labels[0]])


# ## Class imblanace
# From experimentation, and visual inspection classes in each image are highly imbalance (eg: too much background). This resulted in models outputing "blank" all background images and consistently yield high accuracy and low loss. Need to calculate `loss_weights` and pass it to the model when compiling to compensate for this
# 
# 
# Network predict blank results: [https://stackoverflow.com/questions/52123670](https://stackoverflow.com/questions/52123670/neural-network-converges-too-fast-and-predicts-blank-results) 
# Keras model loss_weight: [Keras - compile method](https://keras.io/api/models/model_training_apis/#compile-method)
# 

# In[ ]:


loss_weights = {
    0: 0,
    1: 0,
    2:0
}
mask_files = os.listdir(MASK_SUB_PATH)
for mf in tqdm(mask_files):
    mask_img = cv2.imread(os.path.join(MASK_SUB_PATH, mf))
    classes = tf.argmax(mask_img, axis=-1).numpy()
    class_counts = np.unique(classes, return_counts=True)
    
    for c in range(len(class_counts[0])):
        loss_weights[class_counts[0][c]] += class_counts[1][c]

print(loss_weights)


# Calculate the weights based on counts

# In[ ]:


total = sum(loss_weights.values())
for cl, v in loss_weights.items():
    # do inverse
    loss_weights[cl] = total / v
    
loss_weights


# Create a modifier that is the same shape as output

# In[ ]:



w = [[loss_weights[0], loss_weights[1], loss_weights[2]]] * IMAGE_WIDTH
h = [w] * IMAGE_HEIGHT
loss_mod = np.array(h)


# ## Model
# Define the model. Similar architecture to [Unet](https://arxiv.org/abs/1505.04597) though smaller and more straight forward with no dropout and other regularisation

# In[ ]:


inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
x = inputs
# downstack
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)

# upstack
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv4, x])
x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv3, x])
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv2, x])
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv1, x])
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(3, (1, 1), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs=x)


# In[ ]:


model.summary()


# For the loss function, use `categorical crossentropy` since the label is one hot encoded. Also pass the loss weights computed before into the model

# In[ ]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=loss_mod)


# In[ ]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


model_history = model.fit(train_generator,
                          epochs=20,
                          steps_per_epoch=100,
                          validation_data=validation_generator,
                          validation_steps=9)


# # Have a look at some of the outputs

# In[ ]:


test_imgs, labels = validation_generator.__next__()
predictions = model.predict(test_imgs, use_multiprocessing=False)

for i in range(min(len(predictions), 5)):
    show_imgs(
        [test_imgs[i] / 255., labels[i], predictions[i]],
        ['Source Image', 'True Mask', 'Prediction']
    )


# ## Plot loss and accuracy over time

# In[ ]:


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()

# Loss 

plt.plot(model_history.history['val_loss'])
plt.plot(model_history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# ## Remove the training data from output

# In[ ]:


os.system('rm -r ' + IMG_PATH)
os.system('rm -r ' + MASK_PATH)


# In[ ]:




