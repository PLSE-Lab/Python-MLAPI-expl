#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import clear_output


# In[ ]:


get_ipython().system('pip install -U efficientnet')
clear_output()


# In[ ]:


import math, re, os, gc
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


import efficientnet.tfkeras as efn
from matplotlib import pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2


# In[ ]:


#TPU or GPU detection
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


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
get_ipython().system('gsutil ls $GCS_DS_PATH/images/')
clear_output()


# In[ ]:


img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_10.jpg')
print(img.shape)
plt.imshow(img)


# In[ ]:


train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
sub_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


img_size = 896 #(Trying out 512+256+128)
#img_size = 1000
EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
n_classes = 4


# In[ ]:


#inspired from https://www.kaggle.com/ateplyuk/fork-of-plant-2020-tpu-915e9c

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
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
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[ ]:


train_df.iloc[:, 1:].values;


# In[ ]:


train_tpu_paths = train_df.image_id.apply(lambda x: GCS_DS_PATH+"/images/"+x+".jpg").values
train_labels = train_df.iloc[:, 1:].values
test_tpu_paths = test_df.image_id.apply(lambda x: GCS_DS_PATH+"/images/"+x+".jpg").values


# In[ ]:





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
    
    if tf.random.uniform([1])>0.4:
        image = tf.image.adjust_brightness(image, 0.2)
        
    
    if label is None:
        return image
    else:
        return image, label


# In[ ]:


train_set = (tf.data.Dataset
            .from_tensor_slices((train_tpu_paths, train_labels))
            .map(decode_image, num_parallel_calls=AUTO)
            .map(data_augment, num_parallel_calls=AUTO)
            .repeat()
            .shuffle(512)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_tpu_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# In[ ]:


def get_model_eff():
    base_model =  efn.EfficientNetB7(weights='noisy-student', include_top=False, pooling='avg', input_shape=(None, None, 3))
    x = base_model.output
    predictions = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


with strategy.scope():
    model1 = get_model_eff()
    model1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


train_labels.shape[0]


# In[ ]:


hist1 = model1.fit(train_set,
         steps_per_epoch=train_labels.shape[0]//BATCH_SIZE,
         epochs = EPOCHS,
         callbacks=[lr_callback],
         )


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preds1 = model1.predict(test_dataset)')


# In[ ]:


preds1


# In[ ]:


with strategy.scope():
    gc.collect()


# In[ ]:


def get_model_inc():
    base_model =  InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

with strategy.scope():
    model2 = get_model_inc()
    model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
 
    model2.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )
    


# In[ ]:


hist2 = model2.fit(train_set,
         steps_per_epoch=train_labels.shape[0]//BATCH_SIZE,
         epochs = EPOCHS,
         callbacks=[lr_callback],
         )


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preds2 = model2.predict(test_dataset)')


# In[ ]:


preds = (preds1+preds2)/2


# In[ ]:


preds.shape


# In[ ]:


sub_df.shape


# In[ ]:


sub_df.iloc[:, 1:]=preds


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:


su=pd.read_csv("submission.csv")


# In[ ]:


su


# In[ ]:





# In[ ]:




