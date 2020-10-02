#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from google.cloud import storage
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import subprocess
import sys
import tensorflow as tf
import time
from tqdm.notebook import tqdm

from tensorflow.keras.backend import dot


# In[107]:


def porc(c):
    print_output(run_command(c))

def print_output(output):
    """Prints output from string."""
    for l in output.split('\n'):
        print(l)

def print_pred_metrics(label_actual, label_pred):
    """Prints prediction evaluation metrics and report."""
    print(classification_report(label_actual, label_pred))
#     print(pd.crosstab(label_actual, label_pred, margins=True))
        
def run_command(command):
    """Runs command line command as a subprocess returning output as string."""
    STDOUT = subprocess.PIPE
    process = subprocess.run(command, shell=True, check=False,
                             stdout=STDOUT, stderr=STDOUT, universal_newlines=True)
    return process.stdout

def show_images(imgs, titles=None, hw=(3,3), rc=(4,4)):
    """Show list of images with optional list of titles."""
    h, w = hw
    r, c = rc
    fig=plt.figure(figsize=(w*c, h*r))
    gs1 = gridspec.GridSpec(r, c, fig, hspace=0.2, wspace=0.05)
    for i in range(r*c):
        img = imgs[i].squeeze()
        ax = fig.add_subplot(gs1[i])
        if titles != None:
            ax.set_title(titles[i], {'fontsize': 10})
        plt.imshow(img)
        plt.axis('off')
    plt.show()


# In[4]:


output = run_command('pip freeze | grep efficientnet')
if output == '':
    print_output(run_command('pip install efficientnet'))
else:
    print_output(output)
from efficientnet import tfkeras as efn


# In[5]:


KAGGLE = os.getenv('KAGGLE_KERNEL_RUN_TYPE') != None

BUCKET = 'flowers-caleb'
client = storage.Client(project='fastai-caleb')
bucket = client.get_bucket(BUCKET)

if KAGGLE:
    from kaggle_datasets import KaggleDatasets
    DATASET_DIR = Path('/kaggle/input/flowers-caleb')
    GCS_DATASET_DIR = KaggleDatasets().get_gcs_path(DATASET_DIR.parts[-1])
    MODEL_BUCKET = GCS_DATASET_DIR.split('/')[-1]
    PATH = Path('/kaggle/input/flower-classification-with-tpus')
    TFRECORD_DIR = KaggleDatasets().get_gcs_path(PATH.parts[-1])
    TPU_NAME = None
else:
    DATASET_DIR = Path('./flowers-caleb')
    MODEL_BUCKET = BUCKET
    PATH = Path('/home/jupyter/.fastai/data/flowers')
    TFRECORD_DIR = f'gs://{BUCKET}'
    TPU_NAME = 'dfdc-1'
    
SIZES = {s: f'{s}x{s}' for s in [192, 224, 331, 512]}

AUTO = tf.data.experimental.AUTOTUNE


# In[6]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
    print('Running on TPU ', tpu.master())
except:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# # Datasets

# In[7]:


# https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
    
    # CONVERT DEGREES TO RADIANS
    pi = tf.constant(3.14159265359, tf.float32)
    rotation = pi * rotation / 180.
    shear = pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return dot(dot(rotation_matrix, shear_matrix), dot(zoom_matrix, shift_matrix))


# In[8]:


# https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

def transform(image):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = tf.shape(image)[0]
    XDIM = DIM%2 #fix for size 331
    
    rot = 15. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32') 
    w_shift = 16. * tf.random.normal([1],dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = dot(m,tf.cast(idx,dtype='float32'))
    idx2 = tf.cast(idx2,dtype='int32')
    idx2 = tf.clip_by_value(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])


# In[127]:


def augment(example):
    new_example = example.copy()
    image = new_example['image']
#     image = transform(image)
    shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
    max_resize = tf.cast(tf.cast(shape[-2], tf.float32) * .2 , tf.int32)
    image = tf.image.resize(image, tf.shape(image)[1:3] + tf.random.uniform((), 0, max_resize, tf.int32))
    image = tf.image.random_crop(image, shape)
#     image = tf.image.random_brightness(image, 0.3)
#     image = tf.image.random_contrast(image, 0.9, 1.1)
#     image = tf.image.random_hue(image, 0.1)
#     image = tf.image.random_jpeg_quality(image, 70, 100)
#     image = tf.image.random_saturation(image, 0.95, 1.05)
    new_example['image'] = tf.cast(image, tf.uint8)
    
    return new_example

def get_preprocess_fn(input_size=(224, 224), batch_size=128, norm=None, test=False):
    
    def imagenet_norm(image):
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        
        return (image / tf.constant(255, tf.float32) - mean) / std
    
    norm_fn = {'per_image': tf.image.per_image_standardization,
               'imagenet': imagenet_norm,
               None: tf.image.per_image_standardization
              }

    def preprocess(batch):
        image = tf.image.resize(batch['image'], input_size)
        image = norm_fn[norm](image)

        if test:
            return image
        
        else:
            image = tf.reshape(image, (batch_size, *input_size, 3))
            label = tf.cast(batch['label'], tf.float32)
            label = tf.reshape(label, (batch_size,))
                
            return image, label
        
    return preprocess
    
CLASSES = tf.constant(pd.read_csv(DATASET_DIR/'classes.csv').values.squeeze(), tf.string)

def get_parse_fn(split):
    def parse_fn(example):
        features = {"image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
                    "id": tf.io.FixedLenFeature([], tf.string),
                    "class": tf.io.FixedLenFeature([], tf.int64)}
        
        if split == 'test':
            del features['class']
            
        if split == 'ox102':
            del features['id']
        
        example = tf.io.parse_single_example(example, features)
        example['image'] = tf.image.decode_jpeg(example['image'], channels=3)
        
        if split != 'test':
            example['label'] = tf.cast(example['class'], tf.int32)
            example['class'] = CLASSES[example['label']]
        return example

    return parse_fn

def get_ds(split, img_size=224, batch_size=128, shuffle=False):
    file_pat = f'{TFRECORD_DIR}/tfrecords-jpeg-{SIZES[img_size]}/{split}/*.tfrec'
    
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    
    ds = (tf.data.Dataset.list_files(file_pat, shuffle=shuffle)
          .with_options(options)
          .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
          .map(get_parse_fn(split), num_parallel_calls=AUTO)
         )
    
    if split == 'train':
        ds = ds.repeat().shuffle(2048).batch(batch_size).map(augment, num_parallel_calls=AUTO)
        
    else:
        ds = ds.batch(batch_size)
    
    return ds.prefetch(AUTO)

def get_ds_all(splits, img_size=224, batch_size=128, shuffle=True):
    """splits is a dict of {split: weight} for each split to include"""
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    
    def get_split_ds(split):
        return get_ds_for_train(split, img_size=img_size, batch_size=batch_size, shuffle=shuffle)
    
    ds_dict = {split: get_split_ds(split).repeat() for split in splits}

    ds = tf.data.experimental.sample_from_datasets([ds_dict[s] for s in splits], [splits[s] for s in splits])
        
    ds = ds.shuffle(2048).batch(batch_size).map(augment, num_parallel_calls=AUTO)
    
    return ds.prefetch(AUTO)

def get_ds_for_train(split, img_size=224, batch_size=128, shuffle=True):        
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    
    ds = (tf.data.Dataset.list_files(get_file_pat(split, img_size), shuffle=shuffle)
          .with_options(options)
          .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
          .map(get_parse_fn(split), num_parallel_calls=AUTO)
         )
    
    if split == 'ox102':
        ds = ds.map(add_blank_id, num_parallel_calls=AUTO)

    if split == 'test':
        ds = ds.filter(filter_test).map(add_test_label, num_parallel_calls=AUTO)
        
    return ds


df_test_thresh = pd.read_csv(DATASET_DIR/'df_test_thresh.csv')
TEST_IDS = tf.constant(df_test_thresh.id.values.squeeze(), tf.string)
TEST_LABELS = tf.constant(df_test_thresh.label.values.squeeze(), tf.int32)

def filter_test(example):
    id_check = tf.math.reduce_sum(tf.cast(TEST_IDS == example['id'], tf.int32), axis=0)
    return id_check > 0

def add_test_label(example):
    label_idx = tf.argmax(tf.cast(example['id'] == TEST_IDS, tf.int32))
    example['label'] = TEST_LABELS[label_idx]
    example['class'] = CLASSES[example['label']]
    return example

def add_blank_id(example):
    example['id'] = tf.constant('unknown', tf.string)
    return example

def get_file_pat(split, img_size):
    if split == 'ox102':
        if KAGGLE:
            file_pat = f'{GCS_DATASET_DIR}/extra/ox102/tfrecords-jpeg-{SIZES[img_size]}/*.tfrec'    
        else:
            file_pat = f'{TFRECORD_DIR}/extra/ox102/tfrecords-jpeg-{SIZES[img_size]}/*.tfrec'
    else:
        file_pat = f'{TFRECORD_DIR}/tfrecords-jpeg-{SIZES[img_size]}/{split}/*.tfrec'
    return file_pat


# In[11]:


ds = get_ds('val').unbatch().batch(16)
ds_iter = iter(ds)


# In[12]:


b = next(ds_iter)
b_aug = augment(b)
show_images(b['image'].numpy(), b['class'].numpy().tolist(), hw=(2,2), rc=(2,8))
show_images(b_aug['image'].numpy(), b_aug['class'].numpy().tolist(), hw=(2,2), rc=(2,8))


# # Split Distributions

# In[34]:


splits = ['train', 'val', 'ox102']

def split_classes(split):
    return [b['label'].numpy() for b in get_ds_for_train(split)]

df_split_stats = pd.concat([pd.Series(split_classes(s)).value_counts() for s in splits], axis=1).fillna(0)
df_split_stats.columns = splits
df_split_stats['weight'] = df_split_stats.sum(axis=1).max() / df_split_stats.sum(axis=1)


# The training and validation splits have similar distributions with respect to label. The extra ox102 samples don't have the same distribution. The class weights are calculated based on the distribution of the combined splits.

# In[35]:


(df_split_stats[splits].apply(lambda s: s/df_split_stats[splits].sum(axis=1))
.plot(kind='area', figsize=(10,5)));


# This shows the ratio of validation samples to training samples on a per class basis indexed to the overall ratio of validation samples to training samples. It looks like the distribution of samples across classes is pretty similar between the training and validation sets.

# In[36]:


((df_split_stats.val / df_split_stats.train) / (df_split_stats.val.sum() / df_split_stats.train.sum())).plot();


# While we're at it, we'll also calculate class weights to try in training. The graph in the error analysis below shows that the f1-score is lower on the smaller classes.

# In[37]:


df_split_stats['weight'].plot()
plt.title('Class Weights')
plt.show()


# # Model 

# In[143]:


img_size = 512 
input_size = (512, 512)
batch_size = 16 * strategy.num_replicas_in_sync
weights = 'imagenet'

ds_train = get_ds('train', img_size=img_size, batch_size=batch_size, shuffle=True)
ds_valid = get_ds('val', img_size=img_size, batch_size=batch_size)

splits_all = {'train': 12.7, 'ox102': 2.7}
ds_train_all = get_ds_all(splits_all, img_size=img_size, batch_size=batch_size, shuffle=True)

preprocess = get_preprocess_fn(batch_size=batch_size,
                               input_size=input_size, norm=weights)

ds_train_fit = ds_train.map(preprocess, num_parallel_calls=AUTO)
ds_train_all_fit = ds_train_all.map(preprocess, num_parallel_calls=AUTO)
ds_valid_fit = ds_valid.map(preprocess, num_parallel_calls=AUTO)


# In[125]:


# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync 
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .7

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
rng = range(20)
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[131]:


model_prefix = 'model_efnb7_512_08'
model_dir = f'gs://{MODEL_BUCKET}/{model_prefix}'
checkpoint_dir = f'{model_dir}/checkpoints'
checkpoint_fn = checkpoint_dir + '/' + 'cp-{epoch:04d}.ckpt'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir, write_graph=False, profile_batch=0)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_fn, save_weights_only=True)
lr_cb = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
latest_cp


# In[139]:


dataset_cp = 'cp-0011.ckpt'
if False:
    if not (DATASET_DIR/model_prefix).exists():
        (DATASET_DIR/model_prefix).mkdir()
        (DATASET_DIR/model_prefix/'checkpoints').mkdir()
    
    for b in bucket.list_blobs(prefix=f'{model_prefix}/checkpoints/{dataset_cp}'):
        print(b.name, b.size / 1e6)
        b.download_to_filename(DATASET_DIR/b.name)


# In[130]:


if False:
    print_output(run_command('rm -rf flowers-caleb/model_efnb7_512_08'))


# In[113]:


model_desc = {'model_efnb7_512_06': {'model': 'efnb7, concat avg and max pool',
                                             'size': 512,
                                             'data': 'ds_train',
                                             'aug': 'horizontal flip, vertical flip, zoom-crop 0.2',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 99,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0391,
                                             'train_accuracy': 0.9897,
                                             'val_loss': 0.2817,
                                             'val_accuracy': 0.9440
                                    },
              'model_efnb7_512_07': {'model': 'efnb7, concat avg and max pool',
                                             'size': 512,
                                             'data': 'ds_train',
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 99,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0212,
                                             'train_accuracy': 0.9949,
                                             'val_loss': 0.2891,
                                             'val_accuracy': 0.9448
                                    },
              'model_efnb7_512_08': {'model': 'efnb7, avg pool, dropout 0.2',
                                             'size': 512,
                                             'data': {'train': 12.7, 'val': 3.7, 'ox102': 2.7},
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 149,
                                             'epochs': 20,
                                             'val_steps': 0,
                                             'train_loss': 0.0073,
                                             'train_accuracy': 0.9987,
                                             'val_loss': 0,
                                             'val_accuracy': 0
                                    },
              'model_efnb7_512_09': {'model': 'efnb7, avg pool',
                                             'size': 512,
                                             'data': 'ds_train',
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 99,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0118,
                                             'train_accuracy': 0.9981,
                                             'val_loss': 0.2140,
                                             'val_accuracy': 0.9555
                                    },
              'model_efnb7_512_10': {'model': 'efnb7, avg pool, dropout 0.2',
                                             'size': 512,
                                             'data': 'ds_train',
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 99,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0132,
                                             'train_accuracy': 0.9973,
                                             'val_loss': 0.2122,
                                             'val_accuracy': 0.9580
                                    },
              'model_efnb7_512_11': {'model': 'efnb7, avg pool, dropout 0.2',
                                             'size': 512,
                                             'data': {'train': 12.7, 'ox102': 2.7},
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 120,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0093,
                                             'train_accuracy': 0.9985,
                                             'val_loss': 0.2101,
                                             'val_accuracy': 0.9599 
                                    },
              'model_efnb7_512_12': {'model': 'efnb7, avg pool, dropout 0.4',
                                             'size': 512,
                                             'data': {'train': 12.7, 'ox102': 2.7},
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 120,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0.0094,
                                             'train_accuracy': 0.9986,
                                             'val_loss': 0.2173,
                                             'val_accuracy': 0.9582 
                                    },
              'model_efnb7_512_13': {'model': 'efnb7, avg pool, dropout 0.6',
                                             'size': 512,
                                             'data': {'train': 12.7, 'ox102': 2.7},
                                             'aug': 'horizontal flip',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 100,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0,
                                             'train_accuracy': 0,
                                             'val_loss': 0,
                                             'val_accuracy': 0 
                                    }
              'model_efnb7_512_14': {'model': 'efnb7, avg pool, dropout 0.2',
                                             'size': 512,
                                             'data': {'train': 12.7, 'ox102': 2.7},
                                             'aug': 'horizontal flip, zoom crop 0.2',
                                             'lr_sched': [1e-5, 5e-5 * 16, 1e-5, 5, 0, 0.8],
                                             'steps': 100,
                                             'epochs': 20,
                                             'val_steps': 29,
                                             'train_loss': 0,
                                             'train_accuracy': 0,
                                             'val_loss': 0,
                                             'val_accuracy': 0 

                     }


# ## Single Model

# In[144]:


if True:
    cp_to_load = f'{checkpoint_dir}/{dataset_cp}' if True else None
    weights_to_load = weights if cp_to_load is None else None

    with strategy.scope():

        opt = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

        cnn = efn.EfficientNetB7(weights=weights_to_load,include_top=False,
                                 pooling=None,input_shape=(*input_size, 3))
        
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(cnn.output)
        output = avg_pool
#         max_pool = tf.keras.layers.GlobalMaxPool2D()(cnn.output)
#         output = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(output)
        
        model = tf.keras.Model(cnn.input, output)
    
        if cp_to_load is not None:
            model.load_weights(cp_to_load)

        model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)

    model.summary()


# In[63]:


print('split', '\t', 'items', '\t',  'steps')
for split in ['train', 'val', 'test', 'ox102']:
    items = 0
    prefix = '/'.join(get_file_pat(split, img_size).split('/')[3:-1])
    for b in bucket.list_blobs(prefix=prefix):
        items += int(b.name.split('.')[0][-3:])
#         print(b.name)
    print(split, '\t', items, '\t',  items // batch_size)


# In[44]:


if False:
    history = model.fit(ds_train_all_fit,
                        steps_per_epoch=120,
                        epochs=20,
                        initial_epoch=0,
                        validation_data=ds_valid_fit,
                        validation_steps=29,
                        callbacks=[checkpoint_cb, lr_cb],
                        class_weight=df_split_stats.weight.to_list()
                       )


# ## Ensemble

# In[ ]:


model0_prefix = 'model_efnb6_512_02'
model0_dir = f'gs://{MODEL_BUCKET}/{model0_prefix}'
cp0 = f'{model_dir}/checkpoints/cp-0030.ckpt'

model_prefix_1 = 'model_efnb6_512_03'
model_dir_1 = f'gs://{MODEL_BUCKET}/{model_prefix_1}'
cp1 = f'{model_dir_1}/checkpoints/cp-0020.ckpt'


# In[ ]:


class EnsembleModels(tf.keras.Model):

    def __init__(self, cp=None, cp1=None, load_weights=False):
        super(EnsembleModels, self).__init__()
        self.cnn = efn.EfficientNetB7(weights=None, include_top=False, pooling='avg')
        self.cnn1 = efn.EfficientNetB7(weights=None, include_top=False, pooling='avg')
        self.dense = tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        self.dense1 = tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        self.model = tf.keras.Sequential([self.cnn, self.dense], name='model')
        self.model1 = tf.keras.Sequential([self.cnn1, self.dense1], name='model1')
        self.concat = tf.keras.layers.Concatenate(name='concat')
        self.post_cat_dense  = tf.keras.layers.Dense(256, activation='relu', name='post_cat_dense')
        self.final = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='final')
        
        if load_weights:
            self.model.load_weights(cp)
            self.model1.load_weights(cp1)
        
        self.model.trainable = False
        self.model1.trainable = False
        
    def call(self, inputs):
        model_output  = self.model(inputs)
        model1_output = self.model1(inputs)
        output        = self.concat([model_output, model1_output])
        output        = self.post_cat_dense(output)
        output        = self.final(output)

        return output


# In[ ]:


if False:
    with strategy.scope():

        opt = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

        model = EnsembleModels(cp, cp1)

        model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)

        model.build((batch_size, *input_size, 3))

    model.summary()


# In[ ]:


if False:
    history = model.fit(ds_train_fit,
                        steps_per_epoch=99,
                        epochs=10,
                        initial_epoch=0,
                        validation_data=ds_valid_fit,
                        validation_steps=29,
                        callbacks=[checkpoint_cb, tensorboard_cb, lr_cb],
                        class_weight=df_split_stats.weight.to_list()
                       )


# In[ ]:


if False:
    with strategy.scope():
        model = EnsembleModels()
        model.load_weights(f'{model_dir}/checkpoints/cp-0010.ckpt')


# # Predictions

# In[110]:


split = 'test'

ds_pred = get_ds(split, img_size=img_size, batch_size=batch_size)

preprocess = get_preprocess_fn(batch_size=batch_size,
                               input_size=input_size, norm=weights, test=(split == 'test'))

ds_pred_pp = ds_pred.map(preprocess, num_parallel_calls=AUTO)


# In[67]:


# make sure example order is deterministic so we can line up training data with predictions
assert np.array_equal(np.concatenate([b['id'] for b in ds_pred.as_numpy_iterator()]),
               np.concatenate([b['id'] for b in ds_pred.as_numpy_iterator()]))


# In[111]:


id_list = []
img_list = []
label_list = []
class_list = []

# TTA = 2
# if TTA is not None:
#     predictions = []
#     for b in tqdm(ds_pred.take(1)):
#         id_list.extend(b['id'].numpy().squeeze())
#         if split == 'val':
#             label_list.extend(b['label'].numpy().squeeze())
#             class_list.extend(b['class'].numpy().squeeze())
#         avg_preds = []
#         for i in range(TTA):
#             b_aug = (tf.data.Dataset.from_tensors(b).unbatch()
#                      .map(augment, num_parallel_calls=AUTO).batch(batch_size)
#                      .map(preprocess, num_parallel_calls=AUTO))
#             preds = model.predict(b_aug)
#             avg_preds.append(preds)
#         predictions.extend(np.mean(np.stack(avg_preds), axis=0))
#     predictions = np.stack(predictions, axis=0)
# else:

predictions = model.predict(ds_pred_pp)
for b in ds_pred.as_numpy_iterator():
    id_list.extend(b['id'].squeeze())
    img_list.extend(tf.cast(tf.image.resize(b['image'], (192,192)), tf.uint8).numpy().squeeze())
    if split == 'val':
        label_list.extend(b['label'].squeeze())
        class_list.extend(b['class'].squeeze())


# In[112]:


df_pred = pd.DataFrame({'id': [n.decode() for n in id_list]})

df_pred['label'] = np.argmax(predictions, axis=1)
df_pred['class'] = [n.decode() for n in np.tile(np.expand_dims(CLASSES.numpy(), axis=0),
                                (len(df_pred.label),))[:,df_pred.label].squeeze()]
df_pred['pred_prob'] = np.take_along_axis(predictions, np.expand_dims(df_pred.label, axis=1), axis=1)
    
if split == 'val':
    df_pred['actual_class'] = [n.decode() for n in class_list]
    df_pred['actual_label'] = label_list

if len(img_list) > 0:
    df_pred['image'] = img_list
    
df_pred[['id', 'label']].to_csv('submission.csv', index=False)


# # Error Analysis 

# The purpose of this analysis is to try to determine where the model can be improved. The macro F1-score is interesting in that it is an unweighted average of the class scores, meaning that the smaller classes are especially important. If your model makes errors on only a small number of images from the smaller classes, that can have a big impact on your overall score, even if you've classified many more images correctly in the bigger classes.
# 
# With that in mind, the first table here is the overall F1 score. The second one, shows the class F1 scores, sorted from lowest to highest.
# 
# The graph shows the class F1 scores by the number of examples in each. This model was trained using the validation data and class weights, so there isn't much, if any, bias towards the larger classes, but earlier models without the validation data and class weights indicated that the model didn't do as well on the smaller classes.
# 
# The third table is the list of errors from one of the classes in the class F1 table with the images displayed below.
# 
# And the last table shows the most confident correct predictions of the class along images.
# 
# This version was trained on the validation data as well, so there aren't many images that were predicted incorrectly. I'm not sure there is anything to do on this one - the distinction between black-eyed susans and sunflowers is pretty fine.

# In[119]:


error_index = 1 #change this up here to analyze errors by class, starting with classes with
                #lowest f1-score

if split == 'val':
    class_report = classification_report(df_pred.actual_label, df_pred.label, output_dict=True)
    df_cl_rep = pd.DataFrame(class_report).T
    print(df_cl_rep.iloc[-3:])
    df_cl_rep = df_cl_rep.iloc[:-3]
    df_cl_rep = pd.DataFrame(class_report).T.iloc[:103]
    df_cl_rep = df_cl_rep.sort_values('f1-score')
    df_pred_g = pd.DataFrame(df_pred.groupby(['actual_label', 'label']).count()['id'])
    print('\n',df_cl_rep.head(10))

    plt.scatter(df_cl_rep.support, df_cl_rep['f1-score'])
    plt.title("Support by F1-Score")
    plt.xlabel("support")
    plt.ylabel("f1-score")
        
    plt.show()
    
    error_label = int(df_cl_rep.index[error_index])
    
    df_errors = df_pred[(df_pred.actual_label == error_label) & (df_pred.label != df_pred.actual_label)].copy()
    df_errors['n_class_err'] = df_errors.label.map(df_errors.groupby('label').count()['id'])
    df_errors = df_errors.sort_values(['n_class_err', 'label'], ascending=[False, False])
    print('\n',df_errors[[c for c in df_errors.columns if c not in ['image', 'n_class_err']]])
    
    if len(img_list) > 0:
        show_n = min(len(df_errors), 6)
        show_images(df_errors.image.iloc[:show_n].to_list(),
                    df_errors['class'].iloc[:show_n].to_list(),
                    hw=(2.5,2.5), rc=(1,show_n))
        
        
    df_correct = df_pred[(df_pred.actual_label == error_label) & (df_pred.label == df_pred.actual_label)].copy()
    df_correct = df_correct.sort_values('pred_prob', ascending=False)
    print('\n',df_correct[[c for c in df_correct.columns if c not in ['image']]].head(10))

    if len(img_list) > 0:
        show_n = min(len(df_correct), 6)
        show_images(df_correct.image.iloc[:show_n].to_list(),
                    df_correct['pred_prob'].iloc[:show_n].to_list(),
                    hw=(2.5,2.5), rc=(1,show_n))


# In[122]:


test_index = 1 #change this up here to analyze test classes

if split == 'test':
    df_pred_class = df_pred.groupby('label').agg({'class': 'max', 'pred_prob': 'mean', 'id': 'count'}).sort_values('pred_prob')
    print(df_pred_class.head(10))
    
    # change up the index here to see different classes
    pred_label = df_pred_class.index[test_index]
    
    df_preds_for_label = df_pred[df_pred['label'] == pred_label].sort_values('pred_prob')
    df_preds_for_label[['id', 'pred_prob']].set_index('id').plot(figsize=(11,5))
    plt.show()
    
    show_n = min(6, len(df_preds_for_label))
    print('low prob')
    show_images(df_preds_for_label.image.to_list()[:show_n],
                df_preds_for_label.pred_prob.to_list()[:show_n],
                hw=(2.5,2.5),
                rc=(1, show_n))
    print('high prob')
    show_images(df_preds_for_label.sort_values('pred_prob', ascending=False).image.to_list()[:show_n],
                df_preds_for_label.sort_values('pred_prob', ascending=False).pred_prob.to_list()[:show_n],
                hw=(2.5,2.5),
                rc=(1, show_n))


# # Push Kernel

# In[135]:


if False:
    if not KAGGLE:
        print_output(run_command(f'kaggle d version -r tar -p {DATASET_DIR} -m "add model checkpoint"'))


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.save_notebook()')


# In[142]:


if True:
    if not KAGGLE:

        data = {'id': 'calebeverett/efficientnetb7-with-transformation',
                      'title': 'EfficientnetB7 with Transformation',
                      'code_file': 'flowers.ipynb',
                      'language': 'python',
                      'kernel_type': 'notebook',
                      'is_private': 'false',
                      'enable_gpu': 'true',
                      'enable_internet': 'true',
                      'dataset_sources': ['calebeverett/flowers-caleb'],
                      'competition_sources': ['flower-classification-with-tpus'],
                     ' kernel_sources': []}
        
        with open('kernel-metadata.json', 'w') as f:
            json.dump(data, f)

        print_output(run_command('kaggle k push'))

