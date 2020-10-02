#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import glob
import itertools
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def read_voc_images(root="../input/pascal-voc-2012/VOC2012/", is_train=True):
    txt_name = "{}/ImageSets/Segmentation/{}".format(root, "train.txt" if is_train else "val.txt")
    with open(txt_name, "r") as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = '{}JPEGImages/{}.jpg'.format(root, fname)
        labels[i] = '{}SegmentationClass/{}.png'.format(root, fname)
    return features, labels


# In[13]:


features_train, labels_train = read_voc_images()
features_test, labels_test = read_voc_images(is_train=False)


# In[34]:



import numpy as np
import cv2
import glob
import itertools


def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):

    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img/255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


def getSegmentationArr(path, nClasses,  width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (width, height))

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
()
    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width*height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images, segmentations, batch_size,  n_classes, input_height, input_width, output_height, output_width):

    
    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert(im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)


# In[35]:


from keras.layers.core import Layer, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import  UpSampling2D, ZeroPadding2D
from keras import models
from keras.layers import Conv2D, InputLayer, MaxPool2D
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


df = "channels_first"

def segnet(nClasses, input_height=360, input_width=480):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    model = models.Sequential()
    model.add(InputLayer((3, input_height, input_width)))

    # encoder
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(128, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(256, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size), strides=2, data_format=df))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(512, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    # decoder
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(512, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(256, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(128, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=df))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=df))
    model.add(Conv2D(filter_size, kernel, padding='valid', data_format=df))
    model.add(BatchNormalization())

    model.add(Conv2D(nClasses, 1, padding='valid', data_format=df))

    # model.summary()
    # --------------------------------------------
    outputHeight = model.output_shape[2]
    outputWidth = model.output_shape[3]
    print("output hieight = ", outputHeight)
    print("output  Width = ", outputWidth)

    model.add(Reshape((nClasses,  outputHeight*outputWidth)))

    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight


    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'] )
    model.summary()
    return model


# In[36]:


input_height = 320
input_width = 640
epochs = 1
n_classes = 21
train_batch_size = 8
val_batch_size = 8


# In[37]:


m = segnet(n_classes,  input_height=input_height, input_width=input_width)


# In[38]:


output_height = m.outputHeight
output_width = m.outputWidth

G = imageSegmentationGenerator(features_train,
                               labels_train,
                               train_batch_size,
                               n_classes,
                               input_height,
                               input_width,
                               output_height,
                               output_width)


# In[39]:


G2 = imageSegmentationGenerator(features_test,
                                labels_test,
                                val_batch_size,
                                n_classes,
                                input_height,
                                input_width,
                                output_height,
                                output_width)


# In[40]:


print(len(features_train), len(features_test))


# In[ ]:


for ep in range(epochs):
    m.fit_generator(G, len(features_train) // 8, validation_data=G2, validation_steps=len(features_test) // 8,  epochs=10)
    m.save_weights("wight" + "." + str(ep))
    m.save("mo" + ".model." + str(ep))


# In[ ]:





# In[ ]:


import random


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:




# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

# for imgName, seg in zip(images, segs):
#     X = getImageArr(imgName, input_width, input_height)
#     immm = cv2.imread(imgName)
#     immm = cv2.resize(immm, (input_width, input_height))
#     pr = m.predict(np.array([X]))[0]
#     pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
#     seg_img = np.zeros((output_height, output_width, 3))
#     segg = cv2.imread(seg)
#     seg_img1 = np.zeros_like(segg)
#     for c in range(n_classes):
#         seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
#         seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
#         seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        
#         seg_img1[:, :, 0] += ((segg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
#         seg_img1[:, :, 1] += ((segg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
#         seg_img1[:, :, 2] += ((segg[:, :, 0] == c) * (colors[c][2])).astype('uint8')
#     seg_img = cv2.resize(seg_img, (input_width, input_height))
#     seg_img1 = cv2.resize(seg_img1, (input_width, input_height))
#     immm = cv2.resize(immm, (input_width, input_height))
    
#     plt.imshow(immm)
#     plt.show()
#     plt.imshow((seg_img * 255).astype(np.uint8))
#     plt.show()
#     plt.imshow((seg_img1 * 255).astype(np.uint8))
#     plt.show()
# #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
# #     ax1.set_title("orignal")
# #     ax1.imshow(immm)
# #     ax2.set_title("GT")
# #     ax2.imshow((seg_img * 255).astype(np.uint8))
# #     ax3.set_title("Predicted")
# #     ax3.imshow((seg_img1 * 255).astype(np.uint8))
# #     plt.show()
    


# In[ ]:





# In[ ]:


m.save("model.h5")


# In[ ]:




