#!/usr/bin/env python
# coding: utf-8

# ## Model combining Facenet and VGG in Keras
# 
# This is my single best scoring model (0.893 public leaderboard, 0.900 private leaderboard)
# I used this model in combination with other 36 models to get my final submission.
# From the 37 models that I used, 7 are public kernels (reference below)
# 
# https://www.kaggle.com/shivamsarawagi/wildimagedetection-0-875
# 
# https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197
# 
# https://www.kaggle.com/arjunrao2000/kinship-detection-with-vgg16
# 
# https://www.kaggle.com/leonbora/kinship-recognition-transfer-learning-vggface
# 
# https://www.kaggle.com/janpreets/just-another-feature-extractor-0-824-lb
# 
# https://www.kaggle.com/tenffe/vggface-cv-focal-loss
# 
# https://www.kaggle.com/vaishvik25/smile 
# 
# 
# I used pretrained Facenet model from this repo https://github.com/nyoki-mtl/keras-facenet). 
# 

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
import pandas as pd
from tqdm import tqdm


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


import h5py
from collections import defaultdict
from glob import glob
from random import choice, sample

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


# In[ ]:


train_file_path = "../input/recognizing-faces-in-the-wild/train_relationships.csv"
train_folders_path = "../input/recognizing-faces-in-the-wild/train/"
val_families = "F09"


# In[ ]:


all_images = glob(train_folders_path + "*/*/*.jpg")
print(all_images[0])


# In[ ]:


train_images = [x for x in all_images if val_families not in x]
val_images = [x for x in all_images if val_families in x]


# In[ ]:


ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]
print(ppl[0])


# In[ ]:


train_person_to_images_map = defaultdict(list)

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_families not in x[0]]
val = [x for x in relationships if val_families in x[0]]


# In[ ]:


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


# In[ ]:


model_path = '../input/facenet-keras/facenet_keras.h5'
model_fn = load_model(model_path)


# In[ ]:


for layer in model_fn.layers[:-3]:
    layer.trainable = True


# In[ ]:


model_vgg = VGGFace(model='resnet50', include_top=False)
for layer in model_vgg.layers[:-3]:
    layer.trainable = True


# Define image size for facenet and vgg

# In[ ]:


IMG_SIZE_FN = 160
IMG_SIZE_VGG = 224


# In[ ]:


def read_img_fn(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_FN,IMG_SIZE_FN))
    img = np.array(img).astype(np.float)
    return prewhiten(img)

def read_img_vgg(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_VGG,IMG_SIZE_VGG))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

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
        X1_FN = np.array([read_img_fn(x) for x in X1])
        X1_VGG = np.array([read_img_vgg(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2_FN = np.array([read_img_fn(x) for x in X2])
        X2_VGG = np.array([read_img_vgg(x) for x in X2])

        yield [X1_FN, X2_FN, X1_VGG, X2_VGG], labels


# In[ ]:


def signed_sqrt(x):
    return K.sign(x)*K.sqrt(K.abs(x)+1e-9)


# In[ ]:


def baseline_model():
    input_1 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_2 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_3 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))
    input_4 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))

    x1 = model_fn(input_1)
    x2 = model_fn(input_2)
    x3 = model_vgg(input_3)
    x4 = model_vgg(input_4)
    
    x1 = Reshape((1, 1 ,128))(x1)
    x2 = Reshape((1, 1 ,128))(x2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x1t = Lambda(lambda tensor  : K.square(tensor))(x1)
    x2t = Lambda(lambda tensor  : K.square(tensor))(x2)
    x3t = Lambda(lambda tensor  : K.square(tensor))(x3)
    x4t = Lambda(lambda tensor  : K.square(tensor))(x4)
    
    merged_add_fn = Add()([x1, x2])
    merged_add_vgg = Add()([x3, x4])
    merged_sub1_fn = Subtract()([x1,x2])
    merged_sub1_vgg = Subtract()([x3,x4])
    merged_sub2_fn = Subtract()([x2,x1])
    merged_sub2_vgg = Subtract()([x4,x3])
    merged_mul1_fn = Multiply()([x1,x2])
    merged_mul1_vgg = Multiply()([x3,x4])
    merged_sq1_fn = Add()([x1t,x2t])
    merged_sq1_vgg = Add()([x3t,x4t])
    merged_sqrt_fn = Lambda(lambda tensor  : signed_sqrt(tensor))(merged_mul1_fn)
    merged_sqrt_vgg = Lambda(lambda tensor  : signed_sqrt(tensor))(merged_mul1_vgg)

    
    merged_add_vgg = Conv2D(128 , [1,1] )(merged_add_vgg)
    merged_sub1_vgg = Conv2D(128 , [1,1] )(merged_sub1_vgg)
    merged_sub2_vgg = Conv2D(128 , [1,1] )(merged_sub2_vgg)
    merged_mul1_vgg = Conv2D(128 , [1,1] )(merged_mul1_vgg)
    merged_sq1_vgg = Conv2D(128 , [1,1] )(merged_sq1_vgg)
    merged_sqrt_vgg = Conv2D(128 , [1,1] )(merged_sqrt_vgg)
    
    merged = Concatenate(axis=-1)([Flatten()(merged_add_vgg), (merged_add_fn), Flatten()(merged_sub1_vgg), (merged_sub1_fn),
                                   Flatten()(merged_sub2_vgg), (merged_sub2_fn), Flatten()(merged_mul1_vgg), (merged_mul1_fn), 
                                   Flatten()(merged_sq1_vgg), (merged_sq1_fn), Flatten()(merged_sqrt_vgg), (merged_sqrt_fn)])
    
    merged = Dense(100, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    merged = Dense(25, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model([input_1, input_2, input_3, input_4], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


file_path = "facenet_vgg.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

model = baseline_model() 


# In[ ]:


model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=150, verbose=1,
                    workers = 4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes = True)


# In[ ]:


test_path = "../input/recognizing-faces-in-the-wild/test/"


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')


# In[ ]:


predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1_FN = np.array([read_img_fn(test_path + x) for x in X1])
    X1_VGG = np.array([read_img_vgg(test_path + x) for x in X1])
 
    X2 = [x.split("-")[1] for x in batch]
    X2_FN = np.array([read_img_fn(test_path + x) for x in X2])
    X2_VGG = np.array([read_img_vgg(test_path + x) for x in X2])
    
    pred = model.predict([X1_FN, X2_FN, X1_VGG, X2_VGG]).ravel().tolist()
    
    predictions += pred

submission['is_related'] = predictions

submission.to_csv("facenetvgg.csv", index=False)

