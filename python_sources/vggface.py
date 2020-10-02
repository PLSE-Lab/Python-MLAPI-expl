#!/usr/bin/env python
# coding: utf-8

# This kernel is a fork of https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197 by Beans.
# 
# Changes with respect to the forked kernel:
# *  I used the original images size 224x224
# *  I freezed all the layers of VGGFace model
# *  I concatenated the 2048d features vectors outputted from VGGFace Siamese Networks
# *  I trained a model with 3 dense layers with L2 regularization
#  
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from collections import defaultdict
from glob import glob
from random import choice, sample
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam, Adagrad, RMSprop
from keras import regularizers


# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


# In[ ]:


train_file_path = "../input/train_relationships.csv"
train_folders_path = "../input/train/"
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


relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


# In[ ]:


base_model = VGGFace(model='resnet50', include_top=False)


# In[ ]:


def read_img(path):
    img = image.load_img(path, target_size=(224, 224))
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
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers:
        x.trainable = False

    x1 = base_model(input_1)
    x2 = base_model(input_2)
    
    x = Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(512, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    #x = Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.01))(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=RMSprop(lr=1e-4))

    model.summary()

    return model


# In[ ]:


file_path = "vgg_face.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

es = EarlyStopping(monitor='val_acc',patience=20)

callbacks_list = [checkpoint, reduce_on_plateau]

model = baseline_model()
#model.load_weights(file_path)
model.fit_generator(gen(train, train_person_to_images_map, batch_size=32), use_multiprocessing=True,
                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=2,
                    workers=4, callbacks=callbacks_list, steps_per_epoch=400, validation_steps=100)


# In[ ]:


test_path = "../input/test/"
model.load_weights('vgg_face.h5')  

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.read_csv('../input/sample_submission.csv')

predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)


# In[ ]:


from IPython.display import HTML
import base64
# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


create_download_link(submission)

