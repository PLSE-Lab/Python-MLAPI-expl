#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals

get_ipython().system('pip install tensorflow==2.0.0-beta1')
import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:


tf.executing_eagerly()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image


# In[ ]:


data_dir = '../input/master/P1_Facial_Keypoints-master/data'
training_dir = '../input/master/P1_Facial_Keypoints-master/data/training'
test_dir = '../input/master/P1_Facial_Keypoints-master/data/test'


# In[ ]:


label_frame = pd.read_csv(os.path.join(data_dir,'training_frames_keypoints.csv'), index_col='Unnamed: 0')
label_frame.head()


# In[ ]:


test_label_frame = pd.read_csv(os.path.join(data_dir,'test_frames_keypoints.csv'), index_col='Unnamed: 0')
test_label_frame.head()


# In[ ]:


label_frame.shape


# In[ ]:


test_label_frame.shape


# In[ ]:


def get_img_paths(path):
    file_path = os.listdir(path)
    file_path_full = [os.path.join(training_dir, x) for x in file_path]
    return file_path_full


# In[ ]:


training_img_paths = get_img_paths(training_dir)
len(training_img_paths)


# In[ ]:


test_img_paths = get_img_paths(test_dir)
len(test_img_paths)


# In[ ]:


idx = list(test_label_frame.index)
new_paths = []
for x in test_img_paths:
    y = os.path.basename(x)
    if y in idx:
        new_paths.append(x)
test_img_paths = np.array(new_paths)
test_img_paths.shape


# In[ ]:


def get_labels(data_frame, img_paths=training_img_paths):
    return np.array([data_frame.loc[os.path.basename(x)].values for x in img_paths])


# In[ ]:


training_labels = get_labels(label_frame, training_img_paths)
test_labels = get_labels(test_label_frame, test_img_paths)
print(training_labels.shape, test_labels.shape)
print(training_labels[0].shape)


# In[ ]:


input_size = 128


# In[ ]:


def _rotate_data(image, label):
    angle = tf.random.uniform([1,1], minval=-45, maxval=45)
    
    image = Image.fromarray(np.array(tf.squeeze(image)))
    image = Image.Image.rotate(image, angle)
   
    image = tf.convert_to_tensor(np.array(image))
    image = tf.expand_dims(image, -1)
    
    rad = (22*angle)/(7*180)
    cos, sin = tf.math.cos(rad), tf.math.sin(rad)
    rot_mat = np.array([[cos, -sin], [sin, cos]], dtype='float64')
    rot_mat = tf.squeeze(rot_mat)
    
    label = tf.reshape(label, [-1,2])
    
    x0 = input_size/2
    label -= (x0, x0)
    label = tf.matmul(label, rot_mat)
    label += (x0, x0)
    
    label = tf.reshape(label, [-1])
    
    return image, label


# In[ ]:


def _preprocess_data(image, label):
    
    mfx = input_size / image.shape[0]
    mfy = input_size / image.shape[1]

    new_label = np.zeros(label.shape)
    new_label[::2] = label[::2] * mfy
    new_label[1::2] = label[1::2] * mfx
    
    image = tf.image.resize(image, [input_size, input_size])
    image = tf.image.rgb_to_grayscale(image)
#     return image, new_label
    return _rotate_data(image, new_label)


# In[ ]:


def _load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


# In[ ]:


def check_dataset(t=0, img_paths = training_img_paths,labels = training_labels):
    img = _load_image(img_paths[t])
    label = labels[t].copy()

    img, label = _preprocess_data(img, label)
    
    img = tf.squeeze(img)
    label = tf.reshape(label, [-1,2])
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(label[:, 0], label[:, 1], s=20, marker='.', c='m')
    plt.show()
    
check_dataset(100)


# In[ ]:


check_dataset(0, test_img_paths, test_labels)


# dataset variables
# * training_labels
# * training_img_paths

# # model

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


get_ipython().system("wget 'https://www.kaggleusercontent.com/kf/18233839/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..BfTDA0Blro9eIy1dl4UVpg.Mf_2a9Ij_7fgsZmhivYj44MDIngT7YaW9RLQT8HiFxnvJhX-YjJhhIWqH58godjcHtykW0G-lY81KLwHtVrDiAmqXjEdsnkmemwDPQz-ShMxCiDWR6mXBHOUGM-c1eFrmAQ1VYq0WHTWUNxGeeg0mEIuKCARsNRIQ6pcKxdLXgk.oSf8WyatV_djYgkKuSsyjw/model8.h5'")


# In[ ]:


model1 = keras.models.load_model('model8.h5')


# In[ ]:


model1.summary()


# In[ ]:


keras.utils.plot_model(model1, 'model1.png')


# In[ ]:


model1.compile(optimizer=tf.keras.optimizers.Adam() ,
              loss='mae')


# In[ ]:


len(model1.trainable_variables)


# # training
# dataset variables
# * training_labels
# * training_img_paths

# In[ ]:


BATCH_SIZE = 1024
steps_per_epoch=tf.math.ceil(BATCH_SIZE/32).numpy()
steps_per_epoch


# In[ ]:


def train_step(images, labels):
#     print('train step')
    model1.fit(images, labels, epochs=5, verbose=1, batch_size=32)#, steps_per_epoch=steps_per_epoch)


# In[ ]:


def train_model(epochs, image_paths = training_img_paths, train_labels = training_labels):
    images = np.zeros((BATCH_SIZE,input_size, input_size,1))
    labels = np.zeros((BATCH_SIZE,train_labels.shape[1]))
    
    for epoch in range(epochs):
        print(epoch+1, end=' ')
        x = tf.random.uniform([BATCH_SIZE], minval=0, maxval=len(image_paths), dtype=tf.dtypes.int32)
        for i in range(BATCH_SIZE):
            image = _load_image(image_paths[x[i]])
            label = train_labels[x[i]].copy()
            images[i], labels[i] = _preprocess_data(image, label)
            
#         print('here')
#         print(images.shape)
        train_step(images, labels)
        


# In[ ]:


train_model(5)


# In[ ]:


def check_model(t=0):

    img = _load_image(training_img_paths[t])
    label = training_labels[t].copy()

    img, true_label = _preprocess_data(img, label)
    img = tf.reshape(img, [1,input_size, input_size,1])
    label = model1.predict(img).reshape((-1,2))
    
    img = tf.squeeze(img)
    label = tf.reshape(label, [-1,2])
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(label[:, 0], label[:, 1], s=20, marker='.', c='m')
    plt.show()

#     true_label = tf.reshape(true_label, [-1,2])
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.scatter(true_label[:, 0], true_label[:, 1], s=20, marker='.', c='m')
#     plt.show()


for i in [0,10,110,1110,1,11,111,1111]:
    check_model(i)


# In[ ]:


model1.save('model9.h5')

# Recreate the exact same model purely from the file:
# model = keras.models.load_model('path_to_my_model.h5')


# In[ ]:


train_model(5)


# In[ ]:


def check_model(t=0):

    img = _load_image(training_img_paths[t])
    label = training_labels[t].copy()

    img, true_label = _preprocess_data(img, label)
    img = tf.reshape(img, [1,input_size, input_size,1])
    label = model1.predict(img).reshape((-1,2))
    
    img = tf.squeeze(img)
    label = tf.reshape(label, [-1,2])
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(label[:, 0], label[:, 1], s=20, marker='.', c='m')
    plt.show()

#     true_label = tf.reshape(true_label, [-1,2])
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.scatter(true_label[:, 0], true_label[:, 1], s=20, marker='.', c='m')
#     plt.show()


for i in [0,10,110,1110,1,11,111,1111]:
    check_model(i)


# In[ ]:


model1.save('model10.h5')

# Recreate the exact same model purely from the file:
# model = keras.models.load_model('path_to_my_model.h5')


# In[ ]:


train_model(5)


# In[ ]:


def check_model(t=0):

    img = _load_image(training_img_paths[t])
    label = training_labels[t].copy()

    img, true_label = _preprocess_data(img, label)
    img = tf.reshape(img, [1,input_size, input_size,1])
    label = model1.predict(img).reshape((-1,2))
    
    img = tf.squeeze(img)
    label = tf.reshape(label, [-1,2])
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(label[:, 0], label[:, 1], s=20, marker='.', c='m')
    plt.show()

#     true_label = tf.reshape(true_label, [-1,2])
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.scatter(true_label[:, 0], true_label[:, 1], s=20, marker='.', c='m')
#     plt.show()


for i in [0,10,110,1110,1,11,111,1111]:
    check_model(i)


# In[ ]:


model1.save('model11.h5')

# Recreate the exact same model purely from the file:
# model = keras.models.load_model('path_to_my_model.h5')


# In[ ]:




