#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from matplotlib import pyplot as plt

print(tf.__version__)
print(tf.keras.__version__)


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
GCS_DS_PATH = KaggleDatasets().get_gcs_path('panda-resized-train-data-512x512')


# In[ ]:


train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
print(train_df.shape)
train_df.head()


# In[ ]:


plt.imshow(plt.imread('/kaggle/input/panda-resized-train-data-512x512/train_images/train_images/'
                      + train_df.iloc[0]['image_id'] + '.png'))


# In[ ]:


msk = np.random.rand(len(train_df)) < 0.85
train = train_df[msk]
valid = train_df[~msk]


# In[ ]:


print(train.shape) 
print(valid.shape)
train.head()


# In[ ]:


train_paths = train["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '.png').values
valid_paths = valid["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '.png').values


# In[ ]:


train_labels = pd.get_dummies(train['isup_grade']).astype('int32').values
valid_labels = pd.get_dummies(valid['isup_grade']).astype('int32').values

print(train_labels.shape) 
print(valid_labels.shape)


# In[ ]:


BATCH_SIZE= 8 * strategy.num_replicas_in_sync
img_size = 512
EPOCHS = 11
nb_classes = 6


# In[ ]:


LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 1
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


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .cache()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )


# In[ ]:


valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


# In[ ]:


def get_model():
    base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


with strategy.scope():
    model = get_model()    
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(\n            train_dataset, \n            validation_data = valid_dataset, \n            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            \n            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            \n            callbacks=[lr_callback],\n            epochs=EPOCHS\n)')


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')

plt.show()


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy over epochs')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')

plt.show()


# In[ ]:


model.save('model.h5')

