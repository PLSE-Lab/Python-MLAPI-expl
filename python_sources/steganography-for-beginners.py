#!/usr/bin/env python
# coding: utf-8

# # References
# * JMiPOD: https://hal-utt.archives-ouvertes.fr/hal-02542075/file/J_MiPOD_vPub.pdf
# * J-UNIWARD: https://www.researchgate.net/publication/259639875_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain
# * UERD: https://ieeexplore.ieee.org/abstract/document/7225122/
# 
# Also very useful, as it contains a nice introduction to the topic: http://dde.binghamton.edu/vholub/pdf/Holub_PhD_Dissertation_2014.pdf

# # Introduction
# 
# Steganography is a scientific field I did not came into contact yet, everything I am writing here is the result of me trying to understand various papers about the topic. This notebook is part of this process ;)
# A consequence of this could be that most stuff I am writing here is not of interest for people which already know this field
# 
# What I understood so far:
# * A certain payload is "embedded" into a cover image, in this case a JPEG image
# * The most naive way to do this embedding is for raw images using the LSB of the pixel values
# * For JPEG compressed images, not the pixel values themselves, but the DCT (discrete cosine transform) values are used for embedding
# 
# The embedding is not simply done using random pixels. Every pixel is assigned an "embedding probability", this probability states how likely it is that we will use this pixel for embedding. A cost function is defined and, given a certain payload, minimized with the embedding probabilities of the pixels as parameters. What is interesting is that most algorithms define their cost function in the **spatial domain**. So even when using the DCT coefficients for embedding, the cost function is not necessarily defined in this domain!
# 
# Most algorithms define their cost function in such way that they are "adaptive". This means that it should be more likely to use a pixel for the embedding, which neighborhood shows a lot of structure. In this way it gets harder to spot the embedding.

# # Adaptive Embedding Probability
# In the following I am trying to validate my assumption that the embedding probability is adaptive. I am using the spatial domain here, although the DCT domain is used for embedding. In http://dde.binghamton.edu/vholub/pdf/Holub_PhD_Dissertation_2014.pdf it is mentioned, that many sophisticated steganography algorithms are not necessarily detected the easiest in their embedding domain (DCT), but in the spatial domain, since they are using a cost function in the spatial domain.

# In[ ]:


import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


# In[ ]:


# Load one example images, one cover and one example of every algorithm
image_cover = color.rgb2ycbcr(io.imread('/kaggle/input/alaska2-image-steganalysis/Cover/00001.jpg'))
image_jmipod = color.rgb2ycbcr(io.imread('/kaggle/input/alaska2-image-steganalysis/JMiPOD/00001.jpg'))
image_juniward = color.rgb2ycbcr(io.imread('/kaggle/input/alaska2-image-steganalysis/JUNIWARD/00001.jpg'))
image_uerd = color.rgb2ycbcr(io.imread('/kaggle/input/alaska2-image-steganalysis/UERD/00001.jpg'))

fig, ax = plt.subplots(1,2)
ax[0].imshow(io.imread('/kaggle/input/alaska2-image-steganalysis/Cover/00001.jpg'))
ax[1].imshow(image_cover[:,:,0])
plt.show()


# I am using only the Y channel for this experiment, the other channels should (hopefully) behave the same. 
# I define structure as the intensity standard deviation within one JPEG block of 8x8 pixels.

# In[ ]:


# Calculate the standard deviation within the JPEG blocks (8x8)
def calc_block_std(image_channel):
    
    std_image = np.zeros((64, 64))
    for i in range(63):
        for j in range(63):
            
            std_image[i,j] = np.std(image_channel[i*8:(i+1)*8, j*8:(j+1)*8])
                                    
    return std_image

# Calculate standard deviation for the Y channel of the cover image
cover_y_std = calc_block_std(image_cover[:,:,0])
plt.imshow(cover_y_std)


# Now let's calculate which blocks have at least one changed pixel value with respect to the cover image

# In[ ]:


# Indicate which blocks have at least one changed pixel value
def block_changed(image_cover, image_hidden):
    
    changed_image = np.ones((64, 64))
    for i in range(63):
        for j in range(63):
            
            changed_image[i,j] = not np.allclose(image_cover[i*8:(i+1)*8, j*8:(j+1)*8],
                                            image_hidden[i*8:(i+1)*8, j*8:(j+1)*8])
                                    
    return changed_image.astype(bool)    


# In[ ]:


jmipod_diff = block_changed(image_cover[:,:,0], image_jmipod[:,:,0])
juniward_diff = block_changed(image_cover[:,:,0], image_juniward[:,:,0])
uerd_diff = block_changed(image_cover[:,:,0], image_uerd[:,:,0])
fig, ax = plt.subplots(1,4)
fig.set_figheight(15)
fig.set_figwidth(15)
ax[0].imshow(cover_y_std)
ax[1].imshow(jmipod_diff)
ax[2].imshow(juniward_diff)
ax[3].imshow(uerd_diff)
plt.show()


# It is clearly visible that areas which show a higher standard deviation have a higher embedding probability.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.set_figheight(5)
fig.set_figwidth(15)

sns.kdeplot(cover_y_std[jmipod_diff].reshape(-1), label="JMiPod Changed", ax=ax[0])
sns.kdeplot(cover_y_std[jmipod_diff==False].reshape(-1), label="JMiPod Not Changed", ax=ax[0])
sns.kdeplot(cover_y_std[juniward_diff].reshape(-1), label="Juniward Changed", ax=ax[1])
sns.kdeplot(cover_y_std[juniward_diff==False].reshape(-1), label="Juniward Not Changed", ax=ax[1])
sns.kdeplot(cover_y_std[uerd_diff].reshape(-1), label="Uerd Changed", ax=ax[2])
sns.kdeplot(cover_y_std[uerd_diff==False].reshape(-1), label="Uerd Not Changed", ax=ax[2])
ax[0].set_xlabel("Intensity std. within one block")
ax[1].set_xlabel("Intensity std. within one block")
ax[2].set_xlabel("Intensity std. within one block")
plt.show()


# # Modelling
# 
# I don't have too much experience with model building, especially not for steganalysis.The core ideas of the model (extracted from multiple papers) are:
# * Make it easier for the model to converge by preprocessing the image using several high-pass filters. This is rather straight forward, the manipulation of the DCT coefficients will lead to a high-frequency "noise" component. Additionally we found out above that areas with a high intensity standard deviation have a high embedding probability.
# * Instead of DCT or YCbCr I am using RGB grayscale
# * We have the cover image and three different embedding algorithms --> four classes (I am neglecting here the different compression levels)
# * Instead of a dense head using Global Average Pooling, the concept of this was rather new to me

# In[ ]:


import tensorflow as tf
import os
from math import ceil
from skimage.io import imread
from skimage import color, transform
import numpy as np
from scipy import ndimage, misc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal


# In[ ]:


cover_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/Cover/')
cover_paths_labels = [(os.path.join(cover_path, f), 0) for f in os.listdir(cover_path)]

jmipod_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/JMiPOD/')
jmipod_paths_labels = [(os.path.join(jmipod_path, f), 1) for f in os.listdir(jmipod_path)]

juniward_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/JUNIWARD/')
juniward_paths_labels = [(os.path.join(juniward_path, f), 2) for f in os.listdir(juniward_path)]

uerd_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/UERD/')
uerd_paths_labels = [(os.path.join(uerd_path, f), 3) for f in os.listdir(uerd_path)]


# In[ ]:


BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
NUM_CLASSES = 4

def _parse_function(image_path, label):

    bits = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(bits, channels=1)
    image= tf.cast(image , tf.float32) / 255.  
    image = tf.reshape(image, [512, 512, 1])
    
    label = tf.one_hot(label, NUM_CLASSES)
    label = tf.reshape(label, (4,))

    return image, label

def _augment(image, label):
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Rotate the image
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    
    return image, label
    

def create_dataset(image_paths_labels, augment=True, shuffle=True):

    # This works with arrays as well
    dataset = tf.data.Dataset.from_generator(lambda: image_paths_labels, (tf.string, tf.int32))

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Set batch size
    dataset = dataset.batch(BATCH_SIZE)
    
    # Augmentation
    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set the number of datapoints you want to load and shuffle 
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
    
    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# In[ ]:


# Plot some images and compare embedding algorithms
data_cover = create_dataset(cover_paths_labels, augment=False, shuffle=False).take(1)
data_jmipod = create_dataset(jmipod_paths_labels, augment=False, shuffle=False).take(1)
data_juniward = create_dataset(juniward_paths_labels, augment=False, shuffle=False).take(1)
data_uerd = create_dataset(uerd_paths_labels, augment=False, shuffle=False).take(1)

for cov, jmi, jun, uerd in zip(data_cover, data_jmipod, data_juniward, data_uerd):
    fig, ax = plt.subplots(2,4, figsize=(15,5))
    ax[0, 0].imshow(np.reshape(cov[0][1], (512,512)))
    ax[0, 1].imshow(np.reshape(jmi[0][1], (512,512)))
    ax[0, 2].imshow(np.reshape(jun[0][1], (512,512)))
    ax[0, 3].imshow(np.reshape(uerd[0][1], (512,512))) 
    ax[1, 0].imshow(np.abs(np.reshape(jmi[0][1] - cov[0][1], (512,512))))
    ax[1, 1].imshow(np.abs(np.reshape(jun[0][1] - cov[0][1], (512,512))))
    ax[1, 2].imshow(np.abs(np.reshape(uerd[0][1] - cov[0][1], (512,512))))
    ax[1, 3].imshow(np.abs(np.reshape(uerd[0][1] - jmi[0][1], (512,512))))


# In the following I am initializing the kernels for the first convolutional layer as high-pass filters:
# * 4 kernels of order 1. In theory there are 8 of them, but I did not include mirrored ones
# * All 4 kernels of order 2
# * 4 kernels of order 3. Same argumentation as for the kernels of order 1, potentially we would have 8. Including this order gave me a huge boost in accuracy
# * All(?) 4 3x3 edge kernels
# * 3x3 and 5x5 square kernel

# In[ ]:


def weight_init(shape, dtype=None):
    
    kernel_first_order_1 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 1, -1, 0 ,0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_first_order_1 = tf.reshape(kernel_first_order_1, [5, 5, 1, 1])
    
    kernel_first_order_2 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, -1, 0 ,0],
                           [0, 0, 1, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_first_order_2 = tf.reshape(kernel_first_order_2, [5, 5, 1, 1])    
    
    kernel_first_order_3 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, -1, 0 ,0],
                           [0, 0, 0, 1 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_first_order_3 = tf.reshape(kernel_first_order_3, [5, 5, 1, 1])  
    
    kernel_first_order_4 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, -1, 0 ,0],
                           [0, 1, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_first_order_4 = tf.reshape(kernel_first_order_4, [5, 5, 1, 1]) 
    
    kernel_second_order_1 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 1, -2, 1 ,0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_second_order_1 = tf.reshape(kernel_second_order_1, [5, 5, 1, 1]) 
    
    kernel_second_order_2 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 1, 0, 0 ,0],
                           [0, 0, -2, 0 ,0],
                           [0, 0, 0, 1 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_second_order_2 = tf.reshape(kernel_second_order_2, [5, 5, 1, 1]) 
    
    kernel_second_order_3 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 1, 0 ,0],
                           [0, 0, -2, 0 ,0],
                           [0, 0, 1, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_second_order_3 = tf.reshape(kernel_second_order_3, [5, 5, 1, 1]) 
    
    kernel_second_order_4 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 1 ,0],
                           [0, 0, -2, 0 ,0],
                           [0, 1, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_second_order_4 = tf.reshape(kernel_second_order_4, [5, 5, 1, 1]) 
    
    kernel_third_order_1 = tf.constant([[0, 0, 0, 0, -1],
                           [0, 0, 0, 3 ,0],
                           [0, 0, -3, 0 ,0],
                           [0, 1, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_third_order_1 = tf.reshape(kernel_third_order_1, [5, 5, 1, 1]) 
    
    kernel_third_order_2 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 1, 0, 0 ,0],
                           [0, 0, -3, 0 ,0],
                           [0, 0, 0, 3 ,0],
                           [0, 0, 0, 0 ,-1]], dtype=tf.float32)
    kernel_third_order_2 = tf.reshape(kernel_third_order_2, [5, 5, 1, 1]) 
    
    kernel_third_order_3 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [-1, 3, -3, 1 ,0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_third_order_3 = tf.reshape(kernel_third_order_3, [5, 5, 1, 1]) 
    
    kernel_third_order_4 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 1, 0 ,0],
                           [0, 0, -3, 0 ,0],
                           [0, 0, 3, 0 ,0],
                           [0, 0, -1, 0 ,0]], dtype=tf.float32)
    kernel_third_order_4 = tf.reshape(kernel_third_order_4, [5, 5, 1, 1]) 
    
    kernel_edge_three_1 = tf.constant([[0, 0, 0, 0, 0],
                           [0, -1, 2, -1 ,0],
                           [0, 2, -4, 2 ,0],
                           [0, 0, 0, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_edge_three_1 = tf.reshape(kernel_edge_three_1, [5, 5, 1, 1])
    
    kernel_edge_three_2 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0 ,0],
                           [0, 2, -4, 2 ,0],
                           [0, -1, 2, -1 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_edge_three_2 = tf.reshape(kernel_edge_three_2, [5, 5, 1, 1]) 
    
    kernel_edge_three_3 = tf.constant([[0, 0, 0, 0, 0],
                           [0, -1, 2, 0 ,0],
                           [0, 2, -4, 0 ,0],
                           [0, -1, 2, 0 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_edge_three_3 = tf.reshape(kernel_edge_three_3, [5, 5, 1, 1]) 
    
    kernel_edge_three_4 = tf.constant([[0, 0, 0, 0, 0],
                           [0, 0, 2, -1 ,0],
                           [0, 0, -4, 2 ,0],
                           [0, 0, 2, -1 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    kernel_edge_three_4 = tf.reshape(kernel_edge_three_4, [5, 5, 1, 1]) 
    
    square_kernel_1 = tf.constant([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6 ,2],
                           [-2, 8, -12, 8 ,-2],
                           [2, -6, 8, -6 ,2],
                           [-1, 2, -2, 2 ,-1]], dtype=tf.float32)
    square_kernel_1 = tf.reshape(square_kernel_1, [5, 5, 1, 1]) 
    
    
    square_kernel_2 = tf.constant([[0, 0, 0, 0, 0],
                           [0, -1, 2, -1 ,0],
                           [0, 2, -4, 2 ,0],
                           [0, -1, 2, -1 ,0],
                           [0, 0, 0, 0 ,0]], dtype=tf.float32)
    square_kernel_2 = tf.reshape(square_kernel_2, [5, 5, 1, 1]) 
    
    kernel_sum = tf.stack([kernel_first_order_1, 
                           kernel_first_order_2,
                           kernel_first_order_3,
                           kernel_first_order_4,
                           kernel_second_order_1,
                           kernel_second_order_2,
                           kernel_second_order_3,
                           kernel_second_order_4,
                           kernel_third_order_1,
                           kernel_third_order_2,
                           kernel_third_order_3,
                           kernel_third_order_4,
                           kernel_edge_three_1,
                           kernel_edge_three_2,
                           kernel_edge_three_3,
                           kernel_edge_three_4,
                           square_kernel_1,
                           square_kernel_2], axis=3)
    kernel_collection = tf.reshape(kernel_sum, [5, 5, 1, 18])
    
    return kernel_collection


# In[ ]:


# Create training/validation dataset
paths_labels = cover_paths_labels + jmipod_paths_labels + juniward_paths_labels + uerd_paths_labels
paths_labels_train, paths_labels_valid = train_test_split(paths_labels, shuffle=True, test_size=0.20)

data_train = create_dataset(paths_labels_train)
data_valid = create_dataset(paths_labels_valid)


# In[ ]:


model_dummy = Sequential()
model_dummy.add(Conv2D(18, kernel_initializer=weight_init, kernel_size=5, padding="valid", input_shape=(512,512, 1)))

# Some testing
data_dummy = create_dataset(paths_labels_train).take(1)
dummy_images = model_dummy.predict(data_dummy)

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].hist(np.reshape(dummy_images[0, :,:, 13]*255, (508*508,)))
ax[1].imshow(dummy_images[0, :,:, 13])


# Okay we don't see too much in these images, let's try to take the absolute value. This method has been mentioned in a few papers, others don't use that, but rather use a parametrized ReLU version to handle negative values.

# In[ ]:


from tensorflow.keras import layers

class AbsLayer(layers.Layer):
    
    def get_config(self):
        base_config = super(AbsLayer, self).get_config()
        return dict(list(base_config.items()))
    
    def __init__(self, **kwargs):
        super(AbsLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(AbsLayer, self).build(input_shape)
    
    def call(self, inputs):
        return tf.abs(inputs)


# In[ ]:


model_dummy_abs = Sequential()
model_dummy_abs.add(Conv2D(18, kernel_initializer=weight_init, kernel_size=5, padding="valid", input_shape=(512,512, 1)))
model_dummy_abs.add(AbsLayer())

dummy_images_abs = model_dummy_abs.predict(data_dummy)

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].hist(np.reshape(dummy_images_abs[0, :,:, 1]*255, (508*508,)))
ax[1].imshow(dummy_images_abs[0, :,:, 1])


# This looks much better, but we still have a problem due to the large dynamic range of the pixel values. Let's truncate the pixel values.
# When using truncation we should not use max. pooling! I learned that the hard way (kind of makes sense).

# In[ ]:


# Define a custom activation function which truncates the pixel values
def act_trunc(inputs):
    TRUNCATION_VAL = tf.constant(10./255., dtype=tf.float32)
    return tf.clip_by_value(inputs, -TRUNCATION_VAL, TRUNCATION_VAL)


# In[ ]:


model_dummy_trunc = Sequential()
model_dummy_trunc.add(Conv2D(18, activation=act_trunc, kernel_initializer=weight_init, kernel_size=5, padding="valid", input_shape=(512,512, 1)))
model_dummy_trunc.add(AbsLayer())

dummy_images_trunc = model_dummy_trunc.predict(data_dummy)

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].hist(np.reshape(dummy_images_trunc[0, :,:, 1]*255, (508*508,)))
ax[1].imshow(dummy_images_trunc[0, :,:, 1])


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=1e-7, verbose=1)

# Build the model. I am not using truncation here, I could not get it to work reliably.
model = Sequential()
model.add(Conv2D(18, strides=(2,2), kernel_initializer=weight_init, kernel_size=5, padding="valid", input_shape=(512,512, 1)))
model.add(AbsLayer())
model.add(Conv2D(32, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=2, padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(Conv2D(1024, kernel_size=2, strides=(2,2), padding="same", kernel_initializer=he_normal()))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(4,  activation='softmax'))
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Just train for a few steps, offline I achieved roughly ~86% with a similar model

# In[ ]:


model.fit(x=data_train, validation_data=data_valid,
          steps_per_epoch=1000, validation_steps=10, callbacks=[reduce_lr])


# # Inference

# In[ ]:


test_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/Test/')
test_labels = [(os.path.join(test_path, f), -1) for f in os.listdir(test_path)]

data_test = create_dataset(test_labels, shuffle=False, augment=False)


# In[ ]:


from tensorflow.keras.models import load_model

custom_obj = {}
custom_obj['weight_init'] = weight_init
custom_obj['AbsLayer'] = AbsLayer

predictions = model.predict(data_test, verbose=1)


# In[ ]:


# We are not interested in the specific embedding class but rather whether it is a cover or stego image
prediction_stego = [np.sum(pred[1:4]) for pred in predictions]

import pandas as pd
test_path = os.path.join(os.getcwd(), '/kaggle/input/alaska2-image-steganalysis/Test/')
sub = pd.DataFrame({'Id': os.listdir(test_path),
                    'Label': prediction_stego})

sub.to_csv("submission.csv", index=False)

