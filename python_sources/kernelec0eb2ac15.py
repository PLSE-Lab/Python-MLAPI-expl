#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import numpy as np
import cv2
from random import shuffle
from random import shuffle
import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.applications import InceptionV3
from PIL import Image

shoulder = 0
forearm = 1
hand = 2
finger = 3
humerus = 4
elbow = 5
wrist = 6

#SHOULDER ----> 0
shoulder_path_train = "../input/xr_shoulder_train/XR_SHOULDER_TRAIN"
shoulder_path_test = "../input/xr_shoulder_valid/XR_SHOULDER_VALID"

#FOREARM ----> 1
forearm_path_train = "../input/xr_forearm_train/XR_FOREARM_TRAIN"
forearm_path_test = "../input/xr_forearm_valid/XR_FOREARM_VALID"

#HAND  ----> 2
hand_path_train = "../input/xr_hand_train/XR_HAND_TRAIN"
hand_path_test = "../input/xr_hand_valid/XR_HAND_VALID"

#FINGER  ----> 3
finger_path_train = "../input/xr_finger_train/XR_FINGER_TRAIN"
finger_path_test = "../input/xr_finger_valid/XR_FINGER_VALID"

#HUMERUS ----> 4
humerus_path_train = "../input/xr_humerus_train/XR_HUMERUS_TRAIN"
humerus_path_test = "../input/xr_humerus_valid/XR_HUMERUS_VALID"

#ELBOW -----> 5
elbow_path_train = "../input/xr_elbow_train/XR_ELBOW_TRAIN"
elbow_path_test = "../input/xr_elbow_valid/XR_ELBOW_VALID"

#WRIST  -----> 6
wrist_path_train = "../input/xr_wrist_train/XR_WRIST_TRAIN"
wrist_path_test = "../input/xr_wrist_valid/XR_WRIST_VALID"


# In[2]:


import os
print(os.listdir("../input"))


# In[3]:


import os 
names = []
f = open("../input/train_image_paths.csv")
for row in f:
    names.append(row.strip())


# In[4]:


names


# In[5]:


train_paths = []
for name in names:
    label = None
    arr = name.split("/")
    if arr[2] == "XR_SHOULDER":
        root = shoulder_path_train
        label = shoulder
        
    elif arr[2] == "XR_FINGER":
        root = finger_path_train
        label = finger
        
    elif arr[2] == "XR_ELBOW":
        root = elbow_path_train
        label = elbow
        
    elif arr[2] == "XR_WRIST":
        root = wrist_path_train
        label = wrist
        
    elif arr[2] == "XR_HUMERUS":
        root = humerus_path_train
        label = humerus
        
    elif arr[2] == "XR_FOREARM":
        root = forearm_path_train
        label = forearm
        
    elif arr[2] == "XR_HAND":
        root = hand_path_train
        label = hand
        
    root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]
    train_paths.append([root, label])


# In[6]:


train_paths


# In[7]:


import os 
valid_names = []
f = open("../input/valid_image_paths.csv")
for row in f:
    valid_names.append(row.strip())


# In[8]:


valid_paths = []
for name in valid_names:
    label = None
    arr = name.split("/")
    if arr[2] == "XR_SHOULDER":
        root = shoulder_path_test
        label = shoulder
        
    elif arr[2] == "XR_FINGER":
        root = finger_path_test
        label = finger
        
    elif arr[2] == "XR_ELBOW":
        root = elbow_path_test
        label = elbow
        
    elif arr[2] == "XR_WRIST":
        root = wrist_path_test
        label = wrist
        
    elif arr[2] == "XR_HUMERUS":
        root = humerus_path_test
        label = humerus
        
    elif arr[2] == "XR_FOREARM":
        root = forearm_path_test
        label = forearm
        
    elif arr[2] == "XR_HAND":
        root = hand_path_test
        label = hand
        
    root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]
    valid_paths.append([root, label])


# In[9]:


train_paths


# In[10]:


shuffle(valid_paths)
shuffle(train_paths)


# In[11]:



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


# In[12]:


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
            Y.append(to_categorical(la, 7))
            flip_img = np.fliplr(img)
            X.append(flip_img)
            Y.append(to_categorical(la, 7))
            for i in range(output_size, 1, -1):
                for j in range(output_size):
                    if (i < output_size-20):
                        img[j][i] = img[j][i-20]
                    elif (i < output_size-1):
                        img[j][i] = 0
            X.append(img)
            Y.append(to_categorical(la, 7))
            m += 3

        yield np.array(X), np.array(Y)


# In[13]:


Incp_con = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
Incp_con.summary()


# In[14]:


model = models.Sequential()
model.add(Incp_con)


# In[15]:


model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

model.summary()


# In[16]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile( loss = "categorical_crossentropy", 
               optimizer = sgd, 
               metrics=['accuracy']
             )


# In[17]:


G = imageSegmentationGenerator(train_paths, 32, 224)


# In[18]:


G


# In[19]:


G2 = imageSegmentationGenerator(valid_paths, 32, 224)


# In[20]:


epochs = 2


# In[21]:


epochs


# In[ ]:


for ep in range(epochs):
    model.fit_generator(G, 512, validation_data=G2, validation_steps=200,  epochs=9 ,use_multiprocessing=True)
    model.save_weights("wight" + "." + str(ep))
    model.save("mo" + ".model." + str(ep))


# In[ ]:


model.save("model.h5")


# In[ ]:




