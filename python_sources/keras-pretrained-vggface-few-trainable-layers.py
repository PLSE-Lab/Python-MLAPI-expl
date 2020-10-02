#!/usr/bin/env python
# coding: utf-8

# [Used as a refererence](https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197/)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
import keras
from random import choice, sample
from keras.preprocessing import image
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pip install git+https://github.com/rcmalli/keras-vggface.git


# In[ ]:


from keras_vggface.vggface import VGGFace
from keras_vggface import utils


# In[ ]:


relationships = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")
test_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")


# In[ ]:


def preprocess(filepath):
    img = cv2.imread(filepath)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    x = x.reshape((224,224,3))
    return x


# In[ ]:


#visualizing a test set image
import cv2
ex = cv2.imread("../input/recognizing-faces-in-the-wild/test/face06124.jpg")
ex = cv2.cvtColor(ex, cv2.COLOR_BGR2RGB)
imshow(ex)
print(ex.shape)
ex_pre = preprocess("../input/recognizing-faces-in-the-wild/test/face06124.jpg")
print(ex_pre.shape)


# In[ ]:


#convolutional features
vggfeatures = VGGFace(include_top = False, input_shape = (224,224,3),pooling = 'avg')
for x in vggfeatures.layers[:]:
    x.trainable = False
base_model = vggfeatures


# In[ ]:


base_model.summary()


# In[ ]:


def baseline_model():
    input1 = Input(shape=(224,224,3))
    input2 = Input(shape=(224,224,3))
    
    x1 = base_model(input1)
    x2 = base_model(input2)
    
    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])
    x = Dense(512, activation = 'relu', name = 'fc_final0')(x)
    x = Dense(64, activation = 'relu', name = 'fc_final1')(x)
    x = Dense(1, activation = 'sigmoid', name = 'preds')(x)
    
    model = Model(inputs = [input1,input2], outputs = x, name = 'Face_Sim')
    
    return model 


# In[ ]:


my_model = baseline_model()
my_model.summary()


# In[ ]:


train_base_path = '../input/recognizing-faces-in-the-wild/train/'
families = sorted(os.listdir(train_base_path))
print('We have {} families in the dataset'.format(len(families)))
print(families[:5])


# In[ ]:


members = {i:sorted(os.listdir(train_base_path+i)) for i in families}


# In[ ]:


test_path = '../input/recognizing-faces-in-the-wild/test/'
test_imgs_names = os.listdir(test_path)
print(test_imgs_names[:5])


# In[ ]:


from collections import defaultdict
from glob import glob
train_folders_path = "../input/recognizing-faces-in-the-wild/train/"
val_famillies = "F09"

all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)


# In[ ]:


relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


# In[ ]:


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([preprocess(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([preprocess(x) for x in X2])

        yield [X1, X2], labels


# In[ ]:


file_path = "weights.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

my_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

my_model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True, validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=2, workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)


# In[ ]:


test_folder = "../input/test/"


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

predictions = []

for batch in tqdm(chunker(test_df.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([preprocess(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([preprocess(test_path + x) for x in X2])

    pred = my_model.predict([X1, X2]).ravel().tolist()
    predictions += pred
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0
test_df['is_related'] = predictions

test_df.to_csv("result.csv", index=False)

