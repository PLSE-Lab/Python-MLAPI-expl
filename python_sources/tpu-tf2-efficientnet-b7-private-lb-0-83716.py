#!/usr/bin/env python
# coding: utf-8

# ## 1.Introduction
# ---------------------
# Fine grain classification of fashion image
# 
# 
# this is pretty much a fork of the this starter notebook and i followed his code structure https://www.kaggle.com/xhlulu/flowers-tpu-concise-efficientnet-b7
# 
# ## 2.Motivation
# ----------------------
# Recently i came across tensorflow2's API and TPU and i have seen some impressive speed ups (x100) when moving from GPU to TPU. 
# 
# My motivation for the project is really to get a better feel of TF2 and the general workflow of TPU projects.
# 
# Before trying out the TPU, I was running a GCP EC2 instance with P4 GPU, and each EPOCH took 2hrs and now the fastest EPOCH was only 80 Seconds. 
# A whopping 60X increase in speed
# 
# 
# In order to reach ~75% Accuracy took a grand total of 70 Compute hours @ 40cents per hour, around 28 dollars worth of GCP credits for just training 
# ![rsz_hjtraining.png](attachment:rsz_hjtraining.png)
# 
# 
# ## 3.Model
# ----------------------
# Tried state of the art Effcientnet models. Theses models are trained on imagenet and noisy-student - both variations of the weights are available in the Keras version. I prefer noisy-student pretrained.
# 
# 
# To balance out the bias in the dataset, i choose not to oversample/undersample, and choose class weights instead, which underweight the influence of overrepresented classes and overweight for underrepresented. Im not exactly, sure how does the class weightage works but i believe that it affects the optimiser.
# 
# 
# My inital model used EfficientB7+mish+ranger but implementing the exact same model in TF2 was unrealistic in 1 day.
# 
# The learning rate schedular use was generic from the user [afshiin] (https://www.kaggle.com/afshiin/flower-classification-focal-loss-0-98)
# 
# 
# ## 4.TPU Tips
# -----------------------
# When using TPU for model training, the bottle neck in most cases is fetching images from memory to the TPU. In order to combat the bottleneck there are a couple of tricks you can use:
# 1. HUGE batch size (128bs)
# 2. HIGH resolution (512,512) <- **Couldn't get this to work, if someone knows why I would love to know**
# 3. Enable out of order processing, so process whichever image arrives at the TPU first.
# 4. CACHE !!!
# (did not try)
# 5. Mixed Precision
# 6. XLA Accelerate
# 
# 
# ## 5.Potential Areas for improvement
# -----------------------
# 1. I spent a 2 days trying to get an ensemble to work..  but it kept shutting down, later realised that TPU access on google colab is limited to 3 hrs per session. 
#    My intended submission was to submit an ensemble of effnet B7,B6,B5,B4. This can still be ideally achieved by saving down each model ensemble at a later time.
#    
# 2. I have a hunch that because i did not use TFREC format when loading the images into the TPU, which resulted in an overall decrease in MXU usage and an increase in TPU idle time
# 
# 3. I did not do much of data augmentation, the only tranforms i tried was flip LR
# 
# 
# ## 6.Closing Comments
# -----------------------
# The Notebook/model takes about 16 bucks to run (2hrs) on the TPU, Compared to 28 bucks (70hrs) on a GPU.     
# **Thank you Google for being generous  :)**
# 
# EDA was done on a seperate notebook, if anyone is interested i can share them with you 
# 
# 
# I would also love to find out how did the other better performing teams achieve their score! 
# 
# In addition intend on doing a detailed write up and deep dive on TPU as well as new ways of doing data augmentation, 
# if you are interested to discuss or talk about it hit me up on my **social media** [Hong Jun](https://www.linkedin.com/in/chewhongjun/) 
# and check out my write up [*Here*](https://www.linkedin.com/posts/chewhongjun_data-datascience-ai-activity-6685401910876471296-q2WD)
# 
# 

# ## Dependencies
# ----------------------------

# In[ ]:


get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)

ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [400, 400]
EPOCHS = 15


# In[ ]:


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(30,30))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.axis("off")


# ## A Look into the images
# -------------------------------
# 

# In[ ]:


show_batch(image_batch.numpy(), label_batch.numpy())


# ## Data Processing
# ----------------------------
# - file path from GCP
# - train-val-split 80/20

# In[ ]:


filenames = tf.io.gfile.glob(str(GCS_PATH + '/train/train/train/*/*'))

train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)


# In[ ]:


train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)


TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))


# In[ ]:


# PREDICT CLASSNAMES
CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str(GCS_PATH + "/train/train/train/*"))])
CLASS_NAMES


# ## Helper functions
# ----------------------------
# - Fetching and processing of training data
# - get class label
# - simple data augmentation flip LR

# In[ ]:


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return int(parts[-2])


# In[ ]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3,try_recover_truncated=True,acceptable_fraction=0.5)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)


# In[ ]:


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.image.random_flip_left_right(img)
    return img, label


# In[ ]:


def process_path_test(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


# In[ ]:


train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# ## Sanity Check
# ------------------------------
# - Image Shape, Channel(RGB)
# - corr label

# In[ ]:


for image, label in train_ds.take(5):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# In[ ]:


files=pd.read_csv("../input/shopee-product-detection-student/test.csv")["filename"]
flist=[]
for i in range(12186):
    flist.append(str(GCS_PATH + '/test/test/test/'+ files[i]))


# # Caching
# ---------------------
# - I'm not exactly sure how it works, but you can see the speed up after caching of data in the training phase

# In[ ]:


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(filename)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# In[ ]:


train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))


# ## Dirty way of getting test
# - Count of 12186 is correct
# - The first time i ran my test, i shuffled my test bad idea!

# In[ ]:


print("Start")
test_list_ds = tf.data.Dataset.from_tensor_slices(flist)
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

TEST_IMAGE_COUNT


# ### Start model training
# - freeze bottom layer
# - start with a small lr to warm the model

# In[ ]:


with strategy.scope():
    enb7 = efn.EfficientNetB7(input_shape=[*IMAGE_SIZE, 3], weights='noisy-student', include_top=False)
    enb7.trainable = False 
    
    model1 = tf.keras.Sequential([
        enb7,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.14),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
        
model1.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4 * strategy.num_replicas_in_sync, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
model1.summary()


# ## Balancing biased Dataset
# - There an imbalance in the data classes, to address that i choose to given different weightage, instead of something like SMOTE
# - Weights = (1 / CLASS_COUNT)*(TRAIN_IMG_COUNT)/CLASS_LEN

# In[ ]:


COUNT_00 = len([filename for filename in train_filenames if "/00/" in filename])
print("category 00 images count in training set: " + str(COUNT_00))
COUNT_01 = len([filename for filename in train_filenames if "/01/" in filename])
print("category 01 images count in training set: " + str(COUNT_01))
COUNT_02 = len([filename for filename in train_filenames if "/02/" in filename])
print("category 02 images count in training set: " + str(COUNT_02))
COUNT_03 = len([filename for filename in train_filenames if "/03/" in filename])
print("category 03 images count in training set: " + str(COUNT_03))
COUNT_04 = len([filename for filename in train_filenames if "/04/" in filename])
print("category 04 images count in training set: " + str(COUNT_04))
COUNT_05 = len([filename for filename in train_filenames if "/05/" in filename])
print("category 05 images count in training set: " + str(COUNT_05))
COUNT_06 = len([filename for filename in train_filenames if "/06/" in filename])
print("category 06 images count in training set: " + str(COUNT_06))
COUNT_07 = len([filename for filename in train_filenames if "/07/" in filename])
print("category 07 images count in training set: " + str(COUNT_07))
COUNT_08 = len([filename for filename in train_filenames if "/08/" in filename])
print("category 08 images count in training set: " + str(COUNT_08))
COUNT_09 = len([filename for filename in train_filenames if "/09/" in filename])
print("category 09 images count in training set: " + str(COUNT_09))
COUNT_10 = len([filename for filename in train_filenames if "/10/" in filename])
print("category 10 images count in training set: " + str(COUNT_10))
COUNT_11 = len([filename for filename in train_filenames if "/11/" in filename])
print("category 11 images count in training set: " + str(COUNT_11))
COUNT_12 = len([filename for filename in train_filenames if "/12/" in filename])
print("category 12 images count in training set: " + str(COUNT_12))
COUNT_13 = len([filename for filename in train_filenames if "/13/" in filename])
print("category 13 images count in training set: " + str(COUNT_13))
COUNT_14 = len([filename for filename in train_filenames if "/14/" in filename])
print("category 14 images count in training set: " + str(COUNT_14))
COUNT_15 = len([filename for filename in train_filenames if "/15/" in filename])
print("category 15 images count in training set: " + str(COUNT_15))
COUNT_16 = len([filename for filename in train_filenames if "/16/" in filename])
print("category 16 images count in training set: " + str(COUNT_16))
COUNT_17 = len([filename for filename in train_filenames if "/17/" in filename])
print("category 17 images count in training set: " + str(COUNT_17))
COUNT_18 = len([filename for filename in train_filenames if "/18/" in filename])
print("category 18 images count in training set: " + str(COUNT_18))
COUNT_19 = len([filename for filename in train_filenames if "/19/" in filename])
print("category 19 images count in training set: " + str(COUNT_19))
COUNT_20 = len([filename for filename in train_filenames if "/20/" in filename])
print("category 20 images count in training set: " + str(COUNT_20))
COUNT_21 = len([filename for filename in train_filenames if "/21/" in filename])
print("category 21 images count in training set: " + str(COUNT_21))
COUNT_22 = len([filename for filename in train_filenames if "/22/" in filename])
print("category 22 images count in training set: " + str(COUNT_22))
COUNT_23 = len([filename for filename in train_filenames if "/23/" in filename])
print("category 23 images count in training set: " + str(COUNT_23))
COUNT_24 = len([filename for filename in train_filenames if "/24/" in filename])
print("category 24 images count in training set: " + str(COUNT_24))
COUNT_25 = len([filename for filename in train_filenames if "/25/" in filename])
print("category 25 images count in training set: " + str(COUNT_25))
COUNT_26 = len([filename for filename in train_filenames if "/26/" in filename])
print("category 26 images count in training set: " + str(COUNT_26))
COUNT_27 = len([filename for filename in train_filenames if "/27/" in filename])
print("category 27 images count in training set: " + str(COUNT_27))
COUNT_28 = len([filename for filename in train_filenames if "/28/" in filename])
print("category 28 images count in training set: " + str(COUNT_28))
COUNT_29 = len([filename for filename in train_filenames if "/29/" in filename])
print("category 29 images count in training set: " + str(COUNT_29))
COUNT_30 = len([filename for filename in train_filenames if "/30/" in filename])
print("category 30 images count in training set: " + str(COUNT_30))
COUNT_31 = len([filename for filename in train_filenames if "/31/" in filename])
print("category 31 images count in training set: " + str(COUNT_31))
COUNT_32 = len([filename for filename in train_filenames if "/32/" in filename])
print("category 32 images count in training set: " + str(COUNT_32))
COUNT_33 = len([filename for filename in train_filenames if "/33/" in filename])
print("category 33 images count in training set: " + str(COUNT_33))
COUNT_34 = len([filename for filename in train_filenames if "/34/" in filename])
print("category 34 images count in training set: " + str(COUNT_34))
COUNT_35 = len([filename for filename in train_filenames if "/35/" in filename])
print("category 35 images count in training set: " + str(COUNT_35))
COUNT_36 = len([filename for filename in train_filenames if "/36/" in filename])
print("category 36 images count in training set: " + str(COUNT_36))
COUNT_37 = len([filename for filename in train_filenames if "/37/" in filename])
print("category 27 images count in training set: " + str(COUNT_37))
COUNT_38 = len([filename for filename in train_filenames if "/38/" in filename])
print("category 28 images count in training set: " + str(COUNT_38))
COUNT_39 = len([filename for filename in train_filenames if "/39/" in filename])
print("category 29 images count in training set: " + str(COUNT_39))
COUNT_40 = len([filename for filename in train_filenames if "/40/" in filename])
print("category 30 images count in training set: " + str(COUNT_40))
COUNT_41 = len([filename for filename in train_filenames if "/41/" in filename])
print("category 31 images count in training set: " + str(COUNT_41))

weight_for_0 = (1 / COUNT_00)*(TRAIN_IMG_COUNT)/42.0 
weight_for_1 = (1 / COUNT_01)*(TRAIN_IMG_COUNT)/42.0
weight_for_2 = (1 / COUNT_02)*(TRAIN_IMG_COUNT)/42.0 
weight_for_3 = (1 / COUNT_03)*(TRAIN_IMG_COUNT)/42.0
weight_for_4 = (1 / COUNT_04)*(TRAIN_IMG_COUNT)/42.0 
weight_for_5 = (1 / COUNT_05)*(TRAIN_IMG_COUNT)/42.0
weight_for_6 = (1 / COUNT_06)*(TRAIN_IMG_COUNT)/42.0 
weight_for_7 = (1 / COUNT_07)*(TRAIN_IMG_COUNT)/42.0
weight_for_8 = (1 / COUNT_08)*(TRAIN_IMG_COUNT)/42.0 
weight_for_9 = (1 / COUNT_09)*(TRAIN_IMG_COUNT)/42.0
weight_for_10 = (1 / COUNT_10)*(TRAIN_IMG_COUNT)/42.0 
weight_for_11 = (1 / COUNT_11)*(TRAIN_IMG_COUNT)/42.0
weight_for_12 = (1 / COUNT_12)*(TRAIN_IMG_COUNT)/42.0 
weight_for_13 = (1 / COUNT_13)*(TRAIN_IMG_COUNT)/42.0
weight_for_14 = (1 / COUNT_14)*(TRAIN_IMG_COUNT)/42.0 
weight_for_15 = (1 / COUNT_15)*(TRAIN_IMG_COUNT)/42.0
weight_for_16 = (1 / COUNT_16)*(TRAIN_IMG_COUNT)/42.0 
weight_for_17 = (1 / COUNT_17)*(TRAIN_IMG_COUNT)/42.0
weight_for_18 = (1 / COUNT_18)*(TRAIN_IMG_COUNT)/42.0 
weight_for_19 = (1 / COUNT_19)*(TRAIN_IMG_COUNT)/42.0
weight_for_20 = (1 / COUNT_20)*(TRAIN_IMG_COUNT)/42.0 
weight_for_21 = (1 / COUNT_21)*(TRAIN_IMG_COUNT)/42.0
weight_for_22 = (1 / COUNT_22)*(TRAIN_IMG_COUNT)/42.0 
weight_for_23 = (1 / COUNT_23)*(TRAIN_IMG_COUNT)/42.0
weight_for_24 = (1 / COUNT_24)*(TRAIN_IMG_COUNT)/42.0 
weight_for_25 = (1 / COUNT_25)*(TRAIN_IMG_COUNT)/42.0
weight_for_26 = (1 / COUNT_26)*(TRAIN_IMG_COUNT)/42.0 
weight_for_27 = (1 / COUNT_27)*(TRAIN_IMG_COUNT)/42.0
weight_for_28 = (1 / COUNT_28)*(TRAIN_IMG_COUNT)/42.0 
weight_for_29 = (1 / COUNT_29)*(TRAIN_IMG_COUNT)/42.0
weight_for_30 = (1 / COUNT_30)*(TRAIN_IMG_COUNT)/42.0 
weight_for_31 = (1 / COUNT_31)*(TRAIN_IMG_COUNT)/42.0
weight_for_32 = (1 / COUNT_32)*(TRAIN_IMG_COUNT)/42.0 
weight_for_33 = (1 / COUNT_33)*(TRAIN_IMG_COUNT)/42.0
weight_for_34 = (1 / COUNT_34)*(TRAIN_IMG_COUNT)/42.0 
weight_for_35 = (1 / COUNT_35)*(TRAIN_IMG_COUNT)/42.0
weight_for_36 = (1 / COUNT_36)*(TRAIN_IMG_COUNT)/42.0 
weight_for_37 = (1 / COUNT_37)*(TRAIN_IMG_COUNT)/42.0
weight_for_38 = (1 / COUNT_38)*(TRAIN_IMG_COUNT)/42.0 
weight_for_39 = (1 / COUNT_39)*(TRAIN_IMG_COUNT)/42.0
weight_for_40 = (1 / COUNT_40)*(TRAIN_IMG_COUNT)/42.0 
weight_for_41 = (1 / COUNT_41)*(TRAIN_IMG_COUNT)/42.0


class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4, 5: weight_for_5,
                6: weight_for_6, 7: weight_for_7, 8: weight_for_8, 9: weight_for_9, 10: weight_for_10, 11: weight_for_11,
                12: weight_for_12, 13: weight_for_13, 14: weight_for_14, 15: weight_for_15, 16: weight_for_16, 
                17: weight_for_17, 18: weight_for_18, 19: weight_for_19, 20: weight_for_20, 21: weight_for_21,
                22: weight_for_22, 23: weight_for_23, 24: weight_for_24, 25: weight_for_25, 26: weight_for_26, 
                27: weight_for_27, 28: weight_for_28, 29: weight_for_29, 30: weight_for_30, 31: weight_for_31,
                32: weight_for_32, 33: weight_for_33, 34: weight_for_34, 35: weight_for_35, 36: weight_for_36, 
                37: weight_for_37, 38: weight_for_38, 39: weight_for_39, 40: weight_for_40, 41: weight_for_41,
               }


# In[ ]:


class_weight


# ## Warmup Top Layers

# In[ ]:


history = model1.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
    epochs=5,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
    class_weight=class_weight,
)


# ## Learning rate finder

# In[ ]:


import seaborn as sns

LR_START = 0.00000001
LR_MIN = 0.000001
LEARNING_RATE = 3e-5 * strategy.num_replicas_in_sync
LR_MAX = LEARNING_RATE
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 6))
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers

ES_PATIENCE = 2

for layer in model1.layers:
    layer.trainable = True # Unfreeze layers


es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, 
                   restore_best_weights=True, verbose=1)
lr_callback = LearningRateScheduler(lrfn, verbose=1)

callback_list = [es, lr_callback]

optimizer = optimizers.Adam(lr=3e-5 * strategy.num_replicas_in_sync)
model1.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')
model1.summary()


# In[ ]:


history = model1.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
    epochs=14,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
    callbacks = callback_list,
    class_weight=class_weight,
)


# ## Visualizing model performance
# - Ploting the model accuray and loss, 

# In[ ]:


def plot_metrics(history, metric_list):
    fig, axes = plt.subplots(1, 2, sharex='col', figsize=(24, 12))
    axes = axes.flatten()
    
    for index, metric in enumerate(metric_list):
        axes[index].plot(history.history[metric], label='Train %s' % metric)
        axes[index].plot(history.history['val_%s' % metric], label='Validation %s' % metric)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric)

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()

plot_metrics(history, metric_list=['sparse_categorical_accuracy','loss'])


# In[ ]:


# i wanted to get retain the model with train+val before making a predicition, but a TPU session can only be open for 3 hours :(

# all_list_ds = tf.data.Dataset.from_tensor_slices(filenames)
# all_ds = all_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# all_ds = all_ds.batch(BATCH_SIZE)
# optimizer = optimizers.Adam(lr=4e-5)
# model1.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')
# model1.summary()

# history = model1.fit(
#     all_ds,
#     steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
#     epochs=3,
#     validation_data=val_ds,
#     validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
#     callbacks = callback_list,
#     class_weight=class_weight,
# )


# In[ ]:


ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False


# In[ ]:


files=pd.read_csv("../input/shopee-product-detection-student/test.csv")["filename"]
files


# In[ ]:


probabilities = model1.predict(test_ds)


# In[ ]:


predictions = np.argmax(probabilities, axis=-1)


# In[ ]:


pred = pd.Series(predictions)
pred.to_csv('predEffnet.csv')


# In[ ]:


fnames = pd.Series(files)
df2=pd.concat([fnames, pred], axis=1)


# In[ ]:


df2


# In[ ]:


df2.columns = ['filename', 'category']
df2


# In[ ]:


# Thanks Tong Hui Kang for the function
df2["category"] = df2["category"].apply(lambda x: "{:02}".format(x))  # pad zeroes
df2.to_csv("submission.csv", index=False)


# I overslept admission time if not this suibmission would have gotten  12 on the leaderboard
# 
# ![Screenshot%20from%202020-07-05%2010-15-38.png](attachment:Screenshot%20from%202020-07-05%2010-15-38.png)

# In[ ]:


# if you have reached the end of the notebook,i really appreciate that you took the time out to
# read my solution to this image classification challenge, i am currently a 3rd year cs major in ntu
# for those who were not able to solve this challenge, i urge you to keep trying! For shopee challenge last year, 
# my rank was 274/360, and i have progressed quite abit in only a year to do well in ai matters it is important 
# to keep up to date with the latest models and technology, the reason why i was able to do well is in part due to TPU usage
# next year a new sota model and hardward will be realised if you are familiar enough to utilise it i am certain you'll do well too

# cleaning this notebook up was truly a pain i would appreciate it if you can like the notebook
# make it feel like someone actually read this,

# Cheers and stay safe,
# Hong Jun

