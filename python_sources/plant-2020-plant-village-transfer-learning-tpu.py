#!/usr/bin/env python
# coding: utf-8

# # Plant Pathology 2020
# 
# This kernel attempts to improve upon the current [highest scoring single model public kernel](https://www.kaggle.com/ateplyuk/fork-of-plant-2020-tpu-915e9c) by pretraining on the Plant Village Apple dataset.

# In[ ]:


get_ipython().system('pip install efficientnet > /dev/null')


# In[ ]:


from pathlib import Path
from functools import partial

import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
from PIL import Image
from keras.backend.tensorflow_backend import clear_session

SEED = 420

def decode_image(filename, label=None, image_size=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    if image_size:
        image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label

    
def show_batch(ds, labels):
    row = 6; col = 4;
    row = min(row, BATCH_SIZE//col)

    for (img, label) in ds:
        plt.figure(figsize=(15,int(15*row/col)))
        for j in range(row*col):
            plt.subplot(row,col,j+1)
            plt.axis('off')
            plt.imshow(img[j,])
            plt.title(labels[label[j,].numpy().argmax()])
        plt.show()
        break


# In[ ]:





# ## TPU

# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

def get_strategy():
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
    return strategy

strategy = get_strategy()


# ## Plant Village

# ### EDA

# In[ ]:


LABELS = [
    'healthy', 'Cedar_apple_rust', 'Black_rot', 'Apple_scab', 
]

IMG_PATH = Path('../input/plantvillageapplecolor')


# 

# In[ ]:


all_img_ids = []
all_img_labels = []
for l in LABELS:
    img_ids = [
        '/'.join(l.parts[-2:])
        for l in (IMG_PATH / ('Apple___' + l)).iterdir()
    ]
    all_img_ids.extend(img_ids)
    all_img_labels.extend([l] * len(img_ids))

train_df = pd.DataFrame({'img_id': all_img_ids, 'img_label': all_img_labels})
train_df.img_label = pd.Categorical(train_df.img_label, categories=LABELS, ordered=False)
train_df = pd.concat([train_df, pd.get_dummies(train_df.img_label)], axis=1)
train_df = train_df.sample(frac=1., random_state=SEED)


# In[ ]:


train_df.img_label.value_counts().plot.bar(title="Plant village label distribution")


# In[ ]:


hs, ws = [], []
for path in tqdm(all_img_ids, total=len(all_img_ids)):
    img = Image.open(Path('../input/plantvillageapplecolor')/path)
    h, w = img.size
    hs.append(h)
    ws.append(w)


# In[ ]:


_, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column, vals in zip(axes, ['heights', 'widths'], [hs, ws]):
    ax.hist(vals, bins=100)
    ax.set_title(f'{column} distribution')

plt.show()


# ### Pretrain

# In[ ]:


NB_CLASSES = 4
IMG_SIZE = 256
EPOCHS = 40
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
N_FOLDS = 5


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path('plantvillageapplecolor')


# In[ ]:


train_labels = train_df[LABELS].values.astype(np.int64)
train_paths = train_df.img_id.apply(lambda x: GCS_DS_PATH + '/' + x).values

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(partial(decode_image, image_size=(IMG_SIZE, IMG_SIZE)), num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# In[ ]:


show_batch(train_dataset, LABELS)


# In[ ]:


LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
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

def get_model():
    # Note the input shape needs to be a variable dimension: (None, None, 3).
    base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(None, None, 3))
    x = base_model.output
    predictions = Dense(NB_CLASSES, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


strategy = get_strategy()

with strategy.scope():
    model = get_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC()])

model.fit(
    train_dataset,
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)],
    epochs=EPOCHS
)


# ### Save model

# In[ ]:


model.save("plant_village_pretrain_b7.h5")


# In[ ]:


clear_session()

del model
del train_dataset
del train_labels
del strategy


# In[ ]:


gc.collect()


# ## Transfer learning

# In[ ]:


strategy = get_strategy()


# In[ ]:


NB_CLASSES = 4
IMG_SIZE = 768
EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path('plant-pathology-2020-fgvc7')


# In[ ]:


path = '../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv(path + 'sample_submission.csv')

train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values

train_labels = train.loc[:, 'healthy':].values


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(partial(decode_image, image_size=(IMG_SIZE, IMG_SIZE)), num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# In[ ]:


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(partial(decode_image, image_size=(IMG_SIZE, IMG_SIZE)), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# In[ ]:


show_batch(train_dataset, LABEL_COLS)


# In[ ]:


from tensorflow.keras.models import load_model

def get_model():
    base_model = load_model('./plant_village_pretrain_b7.h5', compile=False)
    x = base_model.layers[-2].output
    predictions = Dense(NB_CLASSES, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


with strategy.scope():
    model = get_model()
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(\n    train_dataset, \n    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,\n    callbacks=[lr_callback],\n    epochs=EPOCHS\n)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'probs = model.predict(test_dataset)')


# In[ ]:


sub.loc[:, 'healthy':] = probs
sub.to_csv('submission.csv', index=False)
sub.head()

