#!/usr/bin/env python
# coding: utf-8

# Because we know that each plate contains 277 distinct classes, we can boost our predictions by solving the Assignment Problem. This approach can also be combined with predicting which group a plate belongs to (see https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak), to narrow the 1108 classes to exactly 277. However, this is left out for simplicity.
# 
# Uses the Hungarian algorithm to boost results from https://www.kaggle.com/chandyalex/recursion-cellular-keras-densenet (which scores 0.113), and acheives a + 0.016 boost over the baseline.

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


import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.applications.densenet import preprocess_input
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import scipy
import scipy.special
import scipy.optimize


WORKERS = 2
CHANNEL = 3

import warnings
warnings.filterwarnings("ignore")
SIZE = 224
NUM_CLASSES = 1108


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras.optimizers import Nadam
from keras import backend as K
import keras
from keras.models import Model


# In[ ]:


def create_model(input_shape,n_out):
    input_tensor = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False,
                   weights=None,
                   input_tensor=input_tensor)
    base_model.load_weights('../input/densenet-keras/DenseNet-BC-121-32-no-top.h5')
    x = GlobalAveragePooling2D()(base_model.output)
#     x = Dropout(0.1)(x)
    x = Dense(1024, activation='relu')(x)
 
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# In[ ]:


submit = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv')
model = create_model(input_shape=(SIZE,SIZE,3),n_out=NUM_CLASSES)
# Load model from "Recursion Cellular Keras Densenet" that achieves 0.113 LB score
model.load_weights('../input/recursion-cellular-keras-densenet/Densenet121.h5')


# In[ ]:


def assign_plate(plate):
    probabilities = np.array(plate)
    cost = probabilities * -1
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    chosen_elements = set(zip(rows.tolist(), cols.tolist()))

    for sample in range(cost.shape[0]):
        for sirna in range(cost.shape[1]):
            if (sample, sirna) not in chosen_elements:
                probabilities[sample, sirna] = 0

    return probabilities.argmax(axis=1).tolist()


# In[ ]:


current_plate = None
plate_probabilities = []
predicted = []
for i, name in tqdm(enumerate(submit['id_code'])):
    path = os.path.join('../input/recursion-cellular-image-classification-224-jpg/test/test/', name+'_s1.jpeg')
    experiment, plate, _ = name.split('_')
    if plate != current_plate:
        if current_plate is not None:
            predicted.extend([str(x) for x in assign_plate(plate_probabilities)])
        plate_probabilities = []
        current_plate = plate

    image = cv2.imread(path)
#     image = cv2.resize(image, (SIZE, SIZE))
    score_predict = model.predict((image[np.newaxis])/255)
    plate_probabilities.append(scipy.special.softmax(score_predict.squeeze()))
predicted.extend([str(x) for x in assign_plate(plate_probabilities)])


# In[ ]:


submit['sirna'] = predicted
submit.to_csv('submission.csv', index=False)
submit.head()
# submission['sirna'] = preds.astype(int)
# submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])

