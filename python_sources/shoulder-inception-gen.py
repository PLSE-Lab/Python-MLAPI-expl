#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import numpy as np
import cv2
from random import shuffle
import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.applications import InceptionV3
from PIL import Image
import os


# In[2]:


shoulder_path_train = "../input/xr_shoulder_train/XR_SHOULDER_TRAIN"
shoulder_path_test = "../input/xr_shoulder_valid/XR_SHOULDER_VALID"
print(os.listdir("../input"))


# In[3]:


import os 
train_names = []
f = open("../input/train_image_paths.csv")
for row in f:
    values = row.strip()
    train_names.append(values)


# In[ ]:





# In[4]:


valid_names = []
f = open("../input/valid_image_paths.csv")
for row in f:
    values = row.strip()
    valid_names.append(values)


# In[5]:


valid_names


# In[6]:


train_names[4].split("/")[4].split("_")[1]


# In[7]:


SHOULDER_TRAIN_PATHS = []
for name in train_names:
    label = None
    arr = name.split("/")
    if arr[2] == "XR_SHOULDER":
        if arr[4].split("_")[1] == "positive":
            label = 1
        else:
            label = 0
        root = shoulder_path_train
        root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]
        SHOULDER_TRAIN_PATHS.append([root, label])


# In[8]:


SHOULDER_TRAIN_PATHS[0]


# In[9]:


SHOULDER_VALID_PATHS = []
for name in valid_names:
    label = None
    arr = name.split("/")
    if arr[2] == "XR_SHOULDER":
        if arr[4].split("_")[1] == "positive":
            label = 1
        else:
            label = 0
        root = shoulder_path_test
        root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]
        SHOULDER_VALID_PATHS.append([root, label])


# In[10]:


SHOULDER_VALID_PATHS


# In[11]:


len(SHOULDER_TRAIN_PATHS)


# In[12]:


shuffle(SHOULDER_TRAIN_PATHS)
shuffle(SHOULDER_VALID_PATHS)


# In[13]:


def getImageArr(path, size):
    try:
        bgr = cv2.imread(path)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.resize(bgr, (size, size))
        img = np.divide(img, 255)
        return img
    except Exception as e:
        img = np.zeros((size, size, 3))
        return img


# In[14]:


import itertools
def imageSegmentationGenerator(pathes_labels, batch_size, output_size):
    x1 = np.array([i[0] for i in pathes_labels])
    y1 = np.array([i[1] for i in pathes_labels])
    zipped = itertools.cycle(zip(x1, y1))

    while True:
        X = []
        Y = []
        for m in range(batch_size):
            pa, la = next(zipped)
            img = getImageArr(pa, output_size)
            X.append(img)
            Y.append(to_categorical(la, 2))
            flip_img = np.fliplr(img)
            X.append(flip_img)
            Y.append(to_categorical(la, 2))
            for i in range(output_size, 1, -1):
                for j in range(output_size):
                    if (i < output_size-20):
                        img[j][i] = img[j][i-20]
                    elif (i < output_size-1):
                        img[j][i] = 0
            X.append(img)
            Y.append(to_categorical(la, 2))
            m += 3

        yield np.array(X), np.array(Y)


# In[ ]:


Incp_con = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


model = models.Sequential()
model.add(Incp_con)


# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.summary()


# In[ ]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile( loss = "categorical_crossentropy", 
               optimizer = sgd, 
               metrics=['accuracy']
             )


# In[ ]:


G = imageSegmentationGenerator(SHOULDER_TRAIN_PATHS, 32, 224)


# In[ ]:


G2 = imageSegmentationGenerator(SHOULDER_VALID_PATHS, 32, 224)


# In[ ]:


epochs = 7


# In[ ]:


for ep in range(epochs):
    model.fit_generator(G, 785, validation_data=G2, validation_steps=52,  epochs=2 ,use_multiprocessing=True)
    model.save_weights("wight" + "." + str(ep))
    model.save("mo" + ".model." + str(ep))


# In[ ]:


model.save("model.h5")


# In[ ]:





# In[ ]:




