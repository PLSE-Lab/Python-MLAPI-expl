#!/usr/bin/env python
# coding: utf-8

# This note book is 1st draft. This note has two big problems (I wrote the last section). If you know solutions, please teach me!!!

# In[ ]:


import math
import os
import gc
import glob
import numpy as np
np.random.seed(0)


import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split

from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read Image

# In[ ]:


train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")
train.head()


# In[ ]:


train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")
train.head()


# In[ ]:


def get_image_path(image_id):
    root_path = "../input/landmark-retrieval-2020/train/"
    extension = ".jpg"
    image_paht = root_path + image_id[0] + "/" + image_id[1] + "/" + image_id[2] + "/"                  + image_id + extension
    return image_paht


# In[ ]:


train["path"] = train["id"].map(get_image_path)
train["path"][0]


# In[ ]:


fig = plt.figure(figsize=(20, 12))
im = Image.open(train["path"][999])
plt.imshow(im)


# # Training Setting

# In[ ]:


num_classes = len(set(train["landmark_id"]))
print("There are ",num_classes, "classes in training data.")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labels = np.array(train["landmark_id"])
lab=LabelEncoder()


# There are so many trainig data, I choose appropriate size data randomlly. 
# 
# To save GPU time, I just use ~ 1% of training data. 

# In[ ]:


# You should change data size here!
train_len = len(train) // 100
train = train.sample(n=train_len)


# In[ ]:


labels=lab.fit_transform(train["landmark_id"])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(train["path"]), labels, test_size=0.33, random_state=42)
del labels
gc.collect()


# In[ ]:


# tf.dataset setting
AUTO = tf.data.experimental.AUTOTUNE

# training configuration
EPOCHS = 3 #10
BATCH_SIZE = 8

# for model
IMAGE_SIZE = 64


# # Dataset

# In[ ]:


def decode_image(filename, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image
    
def to_onehot(label):
    label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
    label = tf.cast(label, tf.int32)
    return label

#def data_augment(image):
#    image = tf.image.random_flip_left_right(image)
#    image = tf.image.random_flip_up_down(image)
#    
#    return image


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds_train = tf.data.Dataset.from_tensor_slices(X_train).map(decode_image)
label_ds_train = tf.data.Dataset.from_tensor_slices(y_train).map(to_onehot)
image_ds_test = tf.data.Dataset.from_tensor_slices(X_test).map(decode_image)
label_ds_test = tf.data.Dataset.from_tensor_slices(y_test).map(to_onehot)

train_dataset = tf.data.Dataset.zip((image_ds_train, label_ds_train)).shuffle(1024).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
valid_dataset = tf.data.Dataset.zip((image_ds_test, label_ds_test)).batch(BATCH_SIZE)


# # Model & Training
# 
# Now, I use my private pretrained ResNet50 Model.

# In[ ]:


model = tf.keras.Sequential([
        ResNet50(
            include_top=False, weights=None,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(num_classes, activation='sigmoid')
    ])

model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()


# In[ ]:


STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE

history = model.fit(
    train_dataset, 
    batch_size=BATCH_SIZE,
    epochs=EPOCHS, 
    validation_data=valid_dataset,
    steps_per_epoch=STEPS_PER_EPOCH
#     callbacks=[],
)


# In[ ]:


def display_training_curves(training, validation, title, subplot):
    """
    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
    """
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# In[ ]:


display_training_curves(
    history.history['loss'], 
    history.history['val_loss'], 
    'loss', 211)
display_training_curves(
    history.history['accuracy'], 
    history.history['val_accuracy'], 
    'accuracy', 212)


# # Export Model

# I also struggled submit. 
# 
# Requirement:
#   - The SavedModel should take a [H,W,3] uint8 tensor as input.
#   - The output should be a dict containing key 'global_descriptor' mapped to a [D] float tensor.
#   
# Refferd kaggle contents:
# 
#   https://www.kaggle.com/mayukh18/creating-submission-from-your-own-model  
#   https://www.kaggle.com/c/landmark-retrieval-2020/discussion/163350
# 
# You should also reffer following document.  
# https://www.tensorflow.org/api_docs/python/tf/saved_model/save

# In[ ]:


class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
  ])
    def my_serve(self, input_image):
        input_image = tf.cast(input_image, tf.float32) / 255        # pre-processing
        input_image = tf.image.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE))
        input_image = tf.expand_dims(input_image, 0)
        probabilities = self.model(input_image)[0]                # prediction from model
        named_output_tensors = {}
        named_output_tensors['global_descriptor'] = tf.identity(probabilities,name='global_descriptor')
        return named_output_tensors


# In[ ]:


tf.keras.backend.set_learning_phase(0) # Make sure no weight update happens
serving_model = ExportModel(model)


# In[ ]:


tf.saved_model.save(serving_model, "./submission",
                    signatures={'serving_default': serving_model.my_serve})


# In[ ]:


from zipfile import ZipFile

variables_datas = glob.glob("./submission/variables/*")

with ZipFile('submission.zip','w') as zip:  
    zip.write('./submission/saved_model.pb', arcname='saved_model.pb') 

    for data in variables_datas:
        path = "variables/" + data.split("/")[-1]
        zip.write(data, arcname=path)


# # Problems
# 
# - I still don't success to sumbit.
#   ~~I think, I may success to save my model as saved_model but why??? 
#   Notebook Exceeded Allowed Compute occur...;(
# - Memory leak occur while trainig.
#   Even I use tf.data.Dataset for trainig, CPU RAM continuauslly increase and finally memory leak occur.
