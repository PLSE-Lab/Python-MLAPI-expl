#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, InceptionV3, Xception, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('pip install efficientnet')
import efficientnet.tfkeras as efn


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()


# In[ ]:


img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
print(img.shape)
plt.imshow(img)


# In[ ]:


path='../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv(path + 'sample_submission.csv')

train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values

train_labels = train.loc[:, 'healthy':].values


# In[ ]:


nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 768
EPOCHS = 40
SEED = 12345
AUG_BATCH = BATCH_SIZE
IMAGE_SIZE = [img_size, img_size]


# In[ ]:


def mixup(image, label, PROBABILITY = 1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = IMAGE_SIZE[0]
    CLASSES = nb_classes
    
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*float(lab1) + a*float(lab2))
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image2,label2


# In[ ]:


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
    
def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_brightness(image, max_delta=0.05, seed=seed)
#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
#     image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .map(mixup)
    .prefetch(AUTO)
    )


# In[ ]:


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# In[ ]:


LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
T_max = EPOCHS
LR_RAMPUP_EPOCHS = 25 #15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[ ]:


def get_model1():
    base_model =  efn.EfficientNetB7(input_shape=(img_size, img_size, 3), weights='noisy-student', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


def get_model2():
    base_model =  efn.EfficientNetB6(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


def get_model3():
    model = tf.keras.Sequential([
        ResNet152V2(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
        L.GlobalAveragePooling2D(),
        L.Dense(train_labels.shape[1], activation='softmax')
    ])
    return model


# In[ ]:


def get_model4():
    model = tf.keras.Sequential([
        InceptionResNetV2(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False),
        L.GlobalAveragePooling2D(),
        L.Dense(train_labels.shape[1], activation='softmax')
    ])
    return model


# In[ ]:


with strategy.scope():
    model1 = get_model1()
    
model1.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy() ,metrics=['categorical_accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model1.fit(\n    train_dataset, \n    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,\n    callbacks=[lr_callback],\n    epochs=EPOCHS\n)')


# In[ ]:


# with strategy.scope():
#     model2 = get_model2()
    
# model2.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy() ,metrics=['categorical_accuracy'])


# In[ ]:


# %%time
# model2.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[ ]:


# with strategy.scope():
#     model3 = get_model3()
    
# model3.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['categorical_accuracy'])


# In[ ]:


# %%time
# model3.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[ ]:


# with strategy.scope():
#     model4 = get_model4()
    
# model4.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),metrics=['categorical_accuracy'])


# In[ ]:


# %%time
# model4.fit(
#     train_dataset, 
#     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
#     callbacks=[lr_callback],
#     epochs=EPOCHS
# )


# In[ ]:


get_ipython().run_cell_magic('time', '', "probs1 = model1.predict(test_dataset, verbose=1)\n# probs2 = model2.predict(test_dataset, verbose=1)\n# probs3 = model3.predict(test_dataset, verbose=1)\n# probs4 = model4.predict(test_dataset, verbose=1)\n# probs_avg = (probs1 + probs2 + probs3 + probs4) / 4\nprobs_avg = probs1\nsub.loc[:, 'healthy':] = probs_avg\nsub.to_csv('submission.csv', index=False)\nsub.head()")

