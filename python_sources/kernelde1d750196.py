#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import threading
import urllib
import cv2
import time

import keras
from keras.applications import ResNet50
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D,     Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

from collections import Counter

import os


# In[ ]:


def check_size(url):
    r = requests.get(url, stream=True)
    return int(r.headers['Content-Length'])

def download_file(url, filename, bar=True):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    try:
        chunkSize = 1024
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            if bar:
                pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize): 
                if chunk: # filter out keep-alive new chunks
                    if bar: 
                        pbar.update (len(chunk))
                    f.write(chunk)
        return filename
    except Exception as e:
        print(e)
        return
    
def download_image_cv2_urllib(url):
    """
    Modifying the url to download the 360p or 720p version actually slows it down. 
    """
    try:
        resp = urllib.request.urlopen(url)
        foo = np.asarray(bytearray(resp.read()), dtype="uint8")
        foo = cv2.imdecode(foo, cv2.IMREAD_COLOR)
        foo = cv2.resize(foo,(128, 128), interpolation=cv2.INTER_AREA)
        return foo
    except:
        return np.array([])


# In[ ]:


download_file("https://s3.amazonaws.com/google-landmark/metadata/train.csv", "train.csv")


# In[ ]:


train = pd.read_csv("train.csv")
print(train.head())
print(train.shape)
print("Number of classes {}".format(len(train.landmark_id.unique())))


# In[ ]:


NUM_THRESHOLD = 250

counts = dict(Counter(train['landmark_id']))
landmarks_dict = {x:[] for x in train.landmark_id.unique() if counts[x] >= NUM_THRESHOLD}
NUM_CLASSES = len(landmarks_dict)
print("Total number of valid classes: {}".format(NUM_CLASSES))

i = 0
landmark_to_idx = {}
idx_to_landmark = []
for k in landmarks_dict:
    landmark_to_idx[k] = i
    idx_to_landmark.append(k)
    i += 1

all_urls = train['url'].tolist()
all_landmarks = train['landmark_id'].tolist()
valid_urls_dict = {x[0].split("/")[-1]:landmark_to_idx[x[1]] for x in zip(all_urls, all_landmarks) if x[1] in landmarks_dict}
valid_urls_list = [x[0] for x in zip(all_urls, all_landmarks) if x[1] in landmarks_dict]

NUM_EXAMPLES = len(valid_urls_list)
print("Total number of valid examples: {}".format(NUM_EXAMPLES))


# In[ ]:


w=20
h=20
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 4
i = 1
for url in valid_urls_list[:16]:
    im = download_image_cv2_urllib(url)
    if im.size != 0:
        fig.add_subplot(rows, columns, i)
        plt.title("Landmark: "+str(idx_to_landmark[valid_urls_dict[url.split("/")[-1]]]))
        plt.imshow(im)
        i += 1


# In[ ]:


w=20
h=20
fig=plt.figure(figsize=(16, 16))
columns = 5
rows = 4
landmarks = [idx_to_landmark[valid_urls_dict[x]] for x in random.sample(valid_urls_dict.keys(), rows)]
for i in range(rows):
    landmark = landmarks[i]
    urls = [x[0] for x in zip(all_urls, all_landmarks) if x[1]==landmark]
    for j in range(columns):
        im = download_image_cv2_urllib(urls[j])
        if im.size != 0:
            fig.add_subplot(rows, columns, i*columns+j+1)
            plt.title("Landmark: "+str(landmark))
            plt.imshow(im)


# In[ ]:


train_urls, validation_urls = train_test_split(valid_urls_list, test_size=1.5*NUM_CLASSES/NUM_EXAMPLES)


# In[ ]:


validation_images = []
validation_y = []
for url in validation_urls:
    im = download_image_cv2_urllib(url)
    if im.size != 0:
        validation_images.append(im)
        validation_y.append(valid_urls_dict[url.split("/")[-1]])

valid_x = np.array(validation_images)
valid_y = np.zeros((len(validation_images), NUM_CLASSES))
        
for i in range(len(validation_y)):
    valid_y[i,validation_y[i]] = 1.


# In[ ]:


class DataGen(Sequence):
    def __init__(self, data, batch_size=24, verbose=1):
        self.batch_size=batch_size
        self.data_urls = data

    def normalize(self,data):
        return data
    
    def __getitem__(self, index):
        batch_urls = random.sample(self.data_urls, self.batch_size)
        
        output = []
        y_classes = []
        for url in batch_urls:
            im = download_image_cv2_urllib(url)
            if im.size != 0:
                output.append(im)
                y_classes.append(valid_urls_dict[url.split("/")[-1]])
        
        x = np.array(output)
        y = np.zeros((len(output), NUM_CLASSES))
        
        for i in range(len(y_classes)):
            y[i,y_classes[i]] = 1.
        
        return x,y
            
    def on_epoch_end(self):
        return

    def __len__(self):
        #return len(valid_urls_list) // self.batch_size
        return 10
def accuracy_class(y_true, y_pred):
    true = K.argmax(y_true, axis=1)
    pred = K.argmax(y_pred, axis=1)
    matches = K.equal(true, pred)
    return K.mean(matches)
res = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))


# In[ ]:


for layer in res.layers[:120]:
    layer.trainable = False
out = Flatten()(res.output)
out = Dense(NUM_CLASSES, activation='softmax')(out)
model = Model(res.input, out)
model.summary()


# In[ ]:


opt = Adam(0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy_class])
model.fit_generator(generator=DataGen(train_urls, batch_size=128),
                    validation_data=[valid_x, valid_y],
                    epochs=80,
                    use_multiprocessing=True,
                    workers=8,
                    verbose=1)

