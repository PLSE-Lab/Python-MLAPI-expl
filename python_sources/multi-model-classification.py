#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[ ]:


import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
print(tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
GCS_PATH=KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')


# **Hardware Detection**

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set.
    # On Kaggle this is always the case.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# **Hyper parameters**

# In[ ]:


SEED = 42

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [512,512]

LR = 0.00004

EPOCHS = 25

WARMUP = 5

WEIGHT_DECAY = 0

LABEL_SMOOTHING = 0.05

TTA = 4


# **Data Splitting**

# In[ ]:


def seed_everything(SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

seed_everything(SEED)

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH+'/train*')

TRAINING_FILENAMES,VALIDATION_FILENAMES=train_test_split(TRAINING_FILENAMES,test_size=0.2,random_state=SEED)

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH+'/test*')


# In[ ]:


print(len(TRAINING_FILENAMES),len(VALIDATION_FILENAMES))


# # Preparing the dataset

# In[ ]:


def decode_augument_image(image_data,seed=SEED):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.bfloat16) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    image = tf.image.rot90(image,k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image,seed=seed)
    image = tf.image.random_flip_up_down(image,seed=seed)
    return image

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.bfloat16) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "age_approx": tf.io.FixedLenFeature([], tf.int64),  
        "sex": tf.io.FixedLenFeature([], tf.int64), 
        "anatom_site_general_challenge" : tf.io.FixedLenFeature([] , tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64),  
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_augument_image(example['image'])
    age = tf.cast(example['age_approx'], tf.bfloat16)
    sex = tf.cast(example['sex'], tf.bfloat16)
    asg = tf.cast(example['anatom_site_general_challenge'] , tf.bfloat16)
    target = tf.cast(example['target'], tf.int32)
    return (image, tf.stack([age,sex,asg])),target

def read_valid_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "age_approx": tf.io.FixedLenFeature([], tf.int64),  
        "sex": tf.io.FixedLenFeature([], tf.int64), 
        "anatom_site_general_challenge" : tf.io.FixedLenFeature([] , tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64),  
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    age = tf.cast(example['age_approx'], tf.bfloat16)
    sex = tf.cast(example['sex'], tf.bfloat16)
    asg = tf.cast(example['anatom_site_general_challenge'] , tf.bfloat16)
    target = tf.cast(example['target'], tf.int32)
    return (image, tf.stack([age,sex,asg])),target

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "age_approx": tf.io.FixedLenFeature([], tf.int64),  
        "sex": tf.io.FixedLenFeature([], tf.int64), 
        "anatom_site_general_challenge" : tf.io.FixedLenFeature([] , tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    age = tf.cast(example['age_approx'], tf.bfloat16)
    sex = tf.cast(example['sex'], tf.bfloat16)
    asg = tf.cast(example['anatom_site_general_challenge'] , tf.bfloat16)
    idnum = example['image_name']
    return (image,tf.stack([age,sex,asg])),idnum# returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False, valid=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    if labeled:
      dataset = dataset.map(read_labeled_tfrecord,num_parallel_calls=AUTO) 
    elif labeled and valid:
      dataset = dataset.map(read_valid_labeled_tfrecord,num_parallel_calls=AUTO)
    else:
      dataset = dataset.map(read_unlabeled_tfrecord,num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES ,labeled=True,valid=False)
    dataset = dataset.shuffle(SEED)
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True ,valid=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=True):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, valid=False, ordered=ordered)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALID_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = NUM_VALID_IMAGES // BATCH_SIZE
print('Dataset: {} training images ,{} validation images,{} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_VALID_IMAGES,NUM_TEST_IMAGES))
print("STEPS_PER_EPOCH are {}".format(STEPS_PER_EPOCH))
print("validation Steps are {}".format(VALIDATION_STEPS))


# # Training 

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras import *
from efficientnet.tfkeras import *


# In[ ]:


def create_model():
    base_model=EfficientNetB7(include_top=False,weights='imagenet',pooling='avg',input_shape=(*IMAGE_SIZE,3))
    base_model.trainable=False
    inp1=Input(shape=(*IMAGE_SIZE,3))
    inp2=Input(shape=(3,))
    X=base_model(inp1,training=False)
    #X=GlobalAveragePooling2D()(X)
    Z=Dense(512,activation='relu')(inp2)
    Z=BatchNormalization()(Z)
    Z=Dropout(0.4)(Z)
    Z=Dense(1024,activation='relu')(Z)
    Z=BatchNormalization()(Z)
    Z=Dropout(0.4)(Z)
    X=Concatenate()([X,Z])
    X=Dense(1024,activation='relu')(X)
    X=BatchNormalization()(X)
    X=Dropout(0.4)(X)
    Y=Dense(1,activation='sigmoid')(X)
    return Model(inputs=[inp1,inp2],outputs=Y)


# In[ ]:


with strategy.scope():
    model = create_model()
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),
                  metrics=[tf.keras.metrics.AUC(),'accuracy'])
    
    model.summary()


# In[ ]:


tf.keras.utils.plot_model(model,show_layer_names=True,show_shapes=True)


# In[ ]:


def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Modified version of the get_cosine_schedule_with_warmup from huggingface.
    (https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup)

    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule = get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)

es=tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=3,verbose=1)


# In[ ]:


model.fit(get_training_dataset(),
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=get_validation_dataset(),
          validation_steps=VALIDATION_STEPS,
          callbacks=[es,lr_schedule],
          verbose=2
         )


# In[ ]:


import h5py

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("mo.h5")
print("Saved model to disk")


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
       model.history.history['loss'], 
       model.history.history['val_loss'], 
       'loss', 211)
display_training_curves(
       model.history.history['accuracy'], 
       model.history.history['val_accuracy'], 
       'accuracy',212)
display_training_curves(
    model.history.history['auc'],
    model.history.history['val_auc'],
    'auc score ', 311)


# In[ ]:


y_pred = model.predict(get_validation_dataset())


y_true = np.array([
    target.numpy() for _, target in iter(get_validation_dataset().unbatch())])

print(y_true.shape)
print(y_pred.shape)


# In[ ]:


from sklearn.metrics import roc_auc_score
metric_score=roc_auc_score(y_true,y_pred)
print(metric_score)


# #  Generate Predictions

# In[ ]:


def predictions():
    test_ds=get_test_dataset(ordered=True)

    test_ds_features=test_ds.map(lambda feat,imname:(feat,0)).batch(BATCH_SIZE) #Getting the features of the Test_ds 

    preds=model.predict(test_ds_features) #predicting with the model

    test_ids_ds = test_ds.map(lambda img, imname: imname)

    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

    prediction_df=pd.DataFrame({'image_name':test_ids ,'target':np.concatenate(preds)}) #writing to Dataframs

    prediction_df.to_csv("submission.csv",index=False) #Generating CSV file
    
predictions()

