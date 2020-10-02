#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# In this kernel, I go through each steps of the process for building a multi-label classifier in the RSNA Intracranial Hemorrhage competition. I will be only using a subset of the total dataset.
# 
# ## Notes
# 
# * V7: I noticed that it takes roughly 1000 seconds to run 2000 iterations. Therefore, I set each batch to be 2000 iterations, so that we can estimate the total running time to be a multiple of 1000 seconds (and at every slice of 2000 iterations, we can get the validation results).

# In[ ]:


import os
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.ops import array_ops
from tqdm import tqdm

  
from keras import backend as K
import tensorflow as tf


# In[ ]:


BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = 'stage_1_train_images/'
TEST_DIR = 'stage_1_test_images/'


# # Load CSVs

# In[ ]:


train_df = pd.read_csv(BASE_PATH + 'stage_1_train.csv')
sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')

train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])
sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])

print(train_df.shape)
train_df.head()


# In[ ]:


test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
print(test_df.shape)
test_df.head()


# We only select a sample of the total training dataset (we randomly choose 400k files), and call it `sample_df`.

# In[ ]:


np.random.seed(1749)
sample_files = np.random.choice(os.listdir(BASE_PATH + TRAIN_DIR), 4000)
sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]


# `pivot_df` is simply `sample_df` reformatted so that each column is a label (this way, we can use multi-label in our data generator later).

# In[ ]:


pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
print(pivot_df.shape)
pivot_df.head()


# # Rescale, Resize and Convert to PNG

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


# In[ ]:


def save_and_resize(filenames, load_dir):    
    save_dir = '/kaggle/tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')
        
        dcm = pydicom.dcmread(path)
        window_center , window_width, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array
        img = window_image(img, window_center, window_width, intercept, slope)
        
        resized = cv2.resize(img, (224, 224))
        res = cv2.imwrite(new_path, resized)


# In[ ]:


save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)
save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)


# # Data Generator

# In[ ]:


BATCH_SIZE = 64

def create_datagen():
    return ImageDataGenerator(validation_split=0.15)

def create_test_gen():
    return ImageDataGenerator().flow_from_dataframe(
        test_df,
        directory='/kaggle/tmp/',
        x_col='filename',
        class_mode=None,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        pivot_df, 
        directory='/kaggle/tmp/',
        x_col='filename', 
        y_col=['any', 'epidural', 'intraparenchymal', 
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='multi_output',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        subset=subset
    )

# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')
val_gen = create_flow(data_generator, 'validation')
test_gen = create_test_gen()


# # Train Model

# ### Loss function
# 
# Focal loss is good for unbalanced datasets, like this one.

# In[ ]:


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(6, activation='sigmoid', 
#                            bias_initializer=Constant(value=-5.5)))
    model.add(layers.Dense(6, activation='sigmoid'))
    
    model.compile(
#         loss=focal_loss,
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_loss', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

total_steps = sample_files.shape[0] / BATCH_SIZE

history = model.fit_generator(
    train_gen,
    steps_per_epoch=2000,
    validation_data=val_gen,
    validation_steps=total_steps * 0.15,
    callbacks=[checkpoint],
    epochs=5
)


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# # Submission

# In[ ]:


model.load_weights('model.h5')
y_test = model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)


# In[ ]:


# Append the output predicts in the wide format to the y_test
test_df = test_df.join(pd.DataFrame(y_test, columns=[
    'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'
]))

# Unpivot table, i.e. wide (N x 6) to long format (6N x 1)
test_df = test_df.melt(id_vars=['filename'])

# Combine the filename column with the variable column
test_df['ID'] = test_df.filename.apply(lambda x: x.replace('.png', '')) + '_' + test_df.variable
test_df['Label'] = test_df['value']

test_df[['ID', 'Label']].to_csv('submission.csv', index=False)

