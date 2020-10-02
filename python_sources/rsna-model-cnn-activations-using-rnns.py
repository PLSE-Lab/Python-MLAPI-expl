#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import (
    Input, Dense, GRU, GlobalAveragePooling1D, 
    GlobalAveragePooling2D, TimeDistributed, 
    concatenate, Masking, Bidirectional
)
import tensorflow.keras.backend as K
import tensorflow as tf

import os


# In[ ]:


BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = 'stage_1_train_images/'
TEST_DIR = 'stage_1_test_images/'


# # Loading Files

# In[ ]:


train_df = pd.read_csv(BASE_PATH + 'stage_1_train.csv')
sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')

train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])
sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])

print(train_df.shape)
train_df.head()


# In[ ]:


test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
print(test_df.shape)
test_df.head()


# In[ ]:


train_meta_df = pd.read_csv('/kaggle/input/rsna-generate-metadata-csvs/train_metadata.csv')
test_meta_df = pd.read_csv('/kaggle/input/rsna-generate-metadata-csvs/test_metadata.csv')
print(train_meta_df.shape)
train_meta_df.head()


# In[ ]:


np.random.seed(1749)
sample_files = np.random.choice(os.listdir(BASE_PATH + TRAIN_DIR), 4000)
sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]


# # Helper functions

# Source: https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing

# In[ ]:


def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


# # Preprocessing

# In[ ]:


N_SAMPLE = 40
series = train_meta_df['StudyInstanceUID'].unique()[:N_SAMPLE]
series


# In[ ]:


def generate_single_instance(filenames, max_padding_len=32):
    padding_img = np.zeros((256, 256))
    padding_label = np.ones(6) * -1

    label_df = train_df[train_df['filename'].isin(filenames)]

    images = []
    labels = []

    for filename in filenames[:max_padding_len]:
        dcm = pydicom.dcmread(BASE_PATH + TRAIN_DIR + filename)
        window_center , window_width, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array
        img = window_image(img, window_center, window_width, intercept, slope)
        resized = cv2.resize(img, (256, 256))

        label = label_df[label_df['filename'] == filename].sort_values('type')['Label'].values

        images.append(resized)
        labels.append(label)

    for _ in range(max_padding_len - len(filenames)):
        images.append(padding_img)
        labels.append(padding_label)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    
    return images, labels


# In[ ]:


images = []
labels = []

for instance in tqdm(series):
    instance_df = train_meta_df[train_meta_df['StudyInstanceUID'] == instance].copy()
    instance_df['ImageHeight'] = instance_df['ImagePositionPatient'].apply(lambda ls: eval(ls)[2])
    instance_df.sort_values('ImageHeight', inplace=True)
    filenames = instance_df['SOPInstanceUID'].apply(lambda x: x + '.dcm').values
    
    instance_images, instance_labels = generate_single_instance(filenames)
    images.append(instance_images)
    labels.append(instance_labels)

images = np.stack(images)
images = np.expand_dims(images, axis=-1)
labels = np.stack(labels)

print(images.shape)
print(labels.shape)


# # Modelling

# In[ ]:


def build_cnn_rnn_model(max_len=32, input_shape=(256, 256, 1)):
    densenet = DenseNet121(include_top=False, weights=None, input_shape=input_shape)
    
    inputs = Input(shape=(max_len, *input_shape))

    convolved = TimeDistributed(densenet)(inputs)
    pooled = TimeDistributed(GlobalAveragePooling2D())(convolved)
    reduced = TimeDistributed(Dense(256, activation='relu'))(pooled)

    masked = Masking(0.0)(reduced)
    out = Bidirectional(GRU(256, return_sequences=True))(masked)
    out = TimeDistributed(Dense(6, activation='sigmoid'))(out)

    model = Model(inputs=inputs, outputs=out)
    return model


# In[ ]:


model = build_cnn_rnn_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# # Training

# In[ ]:


model.fit(images, labels, batch_size=1, epochs=10)

