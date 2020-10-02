#!/usr/bin/env python
# coding: utf-8

# ## Explanation
# 
# This project is made for our computer vision's project in Institut Sains dan Teknologi Terpadu Surabaya

# > Please check Version 37 for best submission
# 
# ## Forked from :
# 
# * [Getting started with 100+ flowers on TPU](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu) version 35
# * [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96) version 1
# * [GridMask data augmentation with tensorflow](https://www.kaggle.com/xiejialun/gridmask-data-augmentation-with-tensorflow) version 1
# 
# ## This notebook in nutshell
# 
# * Image augmentation
#     * GridMask image augmentation (https://arxiv.org/abs/2001.04086)
#     * Rotate, shear, zoom, shift from [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96)
#     * [tf.image](https://www.tensorflow.org/api_docs/python/tf/image) functions
# * Ensemble
#     * EfficientNet B7 (https://arxiv.org/abs/1905.11946)
#     * DenseNet 201 (https://arxiv.org/abs/1608.06993)
# * LAMB optimizer (https://arxiv.org/abs/1904.00962)
# * Global Average Pooling (https://arxiv.org/abs/1312.4400)
# * Split transfer learning into warm up and fine tuning phase
# * TPU
# 
# ## Some things i tried
# 
# * Optimizer (Reference from https://github.com/jettify/pytorch-optimizer)
#     * Adam optimizer
#     * Adaboost optimizer
# * Pretrained model
#     * Inception ResNet V2
#     * ResNet 152 V2
# * Activation (for MLP)
#     * ReLU
#     * Leaky ReLU
# * Fully connected
#     * MLP
#     * GAP + MLP
#     * Local Average Pooling
# * Image augmentation
#     * No augmentation
#     * Use all available augmentation
#     * Use some selected augmentation
#     * Change intensity (brightness range, zoom in, etc.) of augmentation
#     * Use CutOut (https://arxiv.org/abs/1708.04552)
# * Static learning rate
# * Different dynamic learning rate
# * No warm up phase
# * Freeze few layers (which assummed to extract low/medium level feature)
# * Use validation data **only** for validation
# * Use external data (which later found out have data leakage)
# 
# ## Some things i wanted to try
# 
# * Optimizer
#     * Adabound (https://arxiv.org/abs/1902.09843)
#     * Diffgrad (https://arxiv.org/abs/1909.11015)
#     * Yogi (https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization)
#     * RAdam optimizer (https://arxiv.org/abs/1908.03265)
# * Image augmentation
#     * CutMix (https://arxiv.org/abs/1905.04899)
#     * MixUp (https://arxiv.org/abs/1710.09412)
#     * AugMix (https://arxiv.org/abs/1912.02781)
#     * [tf.image.adjust_gamma](https://www.tensorflow.org/api_docs/python/tf/image/adjust_gamma)
#     * [tf.image.per_image_standardization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization)
#     * [tfa.image](https://www.tensorflow.org/addons/api_docs/python/tfa/image) functions
# * FixEfficientNet (https://arxiv.org/abs/2003.08237). Currently pretrained model only available for torch library
# * KFold

# ## Version Note
# 
# * V23 (Version 43) : ?
#     * Replace Inception Resnet V2 with EfficientNetB7 (without image augmentation)
#     * Change epoch for each model during fine tuning phase
#     * Use all available image augmentation
#     * Change order of image augmentation function
#     * Tweak LR schedule
#     * Remove unused code
# * V22 (Version 37) : 0.96032
#     * Use ensemble
#     * For warm up phase :
#         * EfficientNet B7 unfreeze layer block7* and below
#         * DensetNet 201 unfreeze layer [conv/pool]5*
#         * Inception-ResNet V2 freeze all
#     * Add DenseNet 201 pre-trained model
#     * Add Inception ResNetV2 pre-trained model
#     * Disable `tf.image` image adjustment function
#     * Tweak LR schedule
# * V21 (Version 33) : 0.93536
#     * Change fine tuning epoch to 50
#     * Disable `tf.image` rotate 90 and random flip
#     * Enable `tf.image` image adjustment function
# * V20 (Version 32) : 0.91509
#     * Enable `tf.image` rotate 90 and random flip
#     * Tweak Early Stopping parameter
#     * Change fine tuning epoch to 30
# * V19 (Version 31) : 0.95642
#     * Remove MLP from model
#     * Make block6* and below of B7 is trainable during warm up phasae
#     * Implement GridMask from [GridMask data augmentation with tensorflow](https://www.kaggle.com/xiejialun/gridmask-data-augmentation-with-tensorflow)
#     * Tweak LR schedule
#     * Enable image augmentation (only GridMask)
#     * Change warm up epoch to 3 and fine tune epoch to 35
# * V18 (Version 29) : 0.91739
#     * Back to Global Average Pooling with MLP
#     * Change Early Stopping parameter
# * V17 (Version 27) : 0.94896
#     * Change Local Average Pooling parameter `pool_size` to `(8,8)` and `strides` to `8`
#     * Change Early Stopping parameter
# * V16 (Version 26) : 0.93765
#     * Back to Local Average Pooling
#     * Use early stopping callback
#     * Change Early Stopping parameter
# * V15 (Version 24) : **0.96230**
#     * Disable image augmentation
# * V14 (Version 23) : 0.94278
#     * Disable cutout
#     * Use validation for both warm up and fine tuning
#     * Tweak LR schedule
# * V13 (Version ~~21~~ 22) : 0.93404
#     * Tone down LR schedule for fine tuning phase
#     * Add sklearn classification report
#     * Only use training data for warm up and fine tuning phase
#     * Add random hue for image augmentation
#     * Disable augmentation function from [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96)
#     * Add callback to ensure 3 hours session limit used as much as possible
#     * Implement Cutout, modified from https://github.com/tensorflow/tpu/blob/f8b2febcb2f558c8ed45de65cf806cad2b3fcb62/models/official/efficientnet/imagenet_input.py#L130
# * V12 (Version 20) : 0.94971
#     * Fix swapped warm up and fine tune epoch count
#     * Warm up stage use training and validation image without augmentation
#     * Fine tuning stage use training image with augmentation
#     * 40 epochs for fine tuning
#     * Use custom LR schedule
# * V11 (Version 19) : 0.88383
#     * Remove all external dataset
#     * Use LAMB optimizer
#     * Use Global Average Pooling
#     * Add rot90 and flip to image augmentation
#     * Tone down range of some image augmentation function
#     * Replace 'imagenet' with 'noisy-student'
#     * Add some description
#     * Add show confusion matrix function from [Getting started with 100+ flowers on TPU]
#     * 5 epochs for warm-up & 35 epochs for fine tune
# * V11 (Preview 4) : 0.82457
#     * Use GPU w/ 192 * 192
#     * Use 20 epoch
# * ~~V10 (Version 12) : Timeout Exceeded~~
#     * Use all dataset
#     * No data augmentation
#     * Change epoch to 30
# * ~~V9 (Version 11) : **0.95480**~~
#     * Fix LR schedule
#     * Change `AveragePooling2D` parameter
#     * Add `Dropout` with *0.125* rate after `AveragePooling2D`
#     * Change Image Augmentation scheme
#         * Each augmentation have 1/3 or 1/2 chance to be executed. `np.random.randint()` don't have equal distribution
#         * Change insensity of each augmentation
# * ~~V8 (Version 10) : 0.93736~~
#     * Stop using validation data for training
#     * Use external dataset from [tf_flower_photo_tfrec](https://www.kaggle.com/kirillblinov/tf-flower-photo-tfrec), excluding *oxford_102* directory
#     * Change LR schedule
#     * Fix **critical** mistake (didn't update `STEPS_PER_EPOCH` value)
#     * Change epoch to 16
# * ~~V7 (Version 9) : 0.94523~~
#     * Add some description on markdown cells
#     * Use external dataset from [Oxford Flowers TFRecords](https://www.kaggle.com/cdeotte/oxford-flowers-tfrecords)
#     * Change LR schedule
#     * Change epoch to 70
# * V6 (Version 8) : **0.95009**
#     * Train all B7 layers
#     * Remove dropout
#     * Use adam optimizer
#     * Change LR schedule
#     * Use both train and val. data for training
#     * Fix **plt** function
#     * Change epoch to 50
# * V5 (Version 7) : 0.79954
#     * Train layer block5* and below of B7
#     * Data Augmentation only zoom out
#     * Use **adadelta** optimizer
#     * Change LR schedule
#     * Change epoch to 100
# * V4 (Version 6) : 0.90568
#     * Increase patience for Early Stopping
#     * Replace Local Average Pooling with Global Average Pooling
#     * Change optimizer to Adam
#     * Change epoch to 100
# * V3 (Version 5) : 0.90801
#     * Add bias to Dense (classification) layer
#     * Add fully-connected layers
#     * Use Leaky Relu for other Dense layers
#     * Increase patience for Early Stopping
# * V2 (Version 4) : **0.92266**
#     * Fix markdown heading
#     * Train all layers of B7
#     * Use 50 epoch
#     * More aggresive Image Augmentation
#     * Remove B7+ summary
#     * Reduce Early stopping val acc./loss by 90%
#     * Change Local Average Pooling to `AveragePooling2D(pool_size=(4,4), strides=3, padding='valid')`
# * V1 (Version 3) : **0.91153**
#     * Train layers block6* and below of B7
#     * Use 10 epoch
#     * MXU usage is quite low with high idle time on training model. Will try [Data Augmentation using GPU/TPU for Maximum Speed!](https://www.kaggle.com/yihdarshieh/make-chris-deotte-s-data-augmentation-faster)

# ## Install, load and configure library

# In[ ]:


get_ipython().system('pip install efficientnet tensorflow_addons==0.9.1 tensorflow==2.1.0')


# In[ ]:


import math
import re
import random
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from matplotlib import pyplot as plt
from datetime import datetime


# In[ ]:


print(f'Numpy version : {np.__version__}')
print(f'Tensorflow version {tf.__version__}')

AUTO = tf.data.experimental.AUTOTUNE
START_TIME = datetime.now()

# this won't make result reproducible, unless you use CPU
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ## TPU or GPU detection

# In[ ]:


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


# ## Competition data access
# 
# TPUs read data directly from Google Cloud Storage (GCS). This Kaggle utility will copy the dataset to a GCS bucket co-located with the TPU. If you have multiple datasets attached to the notebook, you can pass the name of a specific dataset to the get_gcs_path function. The name of the dataset is the name of the directory it is mounted in. Use `!ls /kaggle/input/` to list attached datasets.

# In[ ]:


get_ipython().system('ls -lha /kaggle/input/')


# In[ ]:


from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
GCS_DS_PATH


# # Configuration

# In[ ]:


IMAGE_SIZE = [512, 512]

VALIDATE_WARMUP = False
EPOCHS_WARMUP = 3
DO_AUG_WARMUP = False

VALIDATE = False
EPOCHS = 30
DO_AUG = True

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102


# # Datasets functions

# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_train_val_dataset(do_aug=True):
    dataset = load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES, labeled=True)
    if do_aug:
        dataset = dataset.map(image_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset(do_aug=True):
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    if do_aug:
        dataset = dataset.map(image_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(f'Total training image: {NUM_TRAINING_IMAGES}')
print(f'Total validation image: {NUM_VALIDATION_IMAGES}')
print(f'Total test image: {NUM_TEST_IMAGES}')
print(f'Steps per epoch : {STEPS_PER_EPOCH}')


# # Image augmentation functions
# 
# | Function   | Chance | Range                             |
# | ---------- | ------ | --------------------------------- |
# | Flip       | 50%    | Only Left to right                |
# | Brightness | 50%    | 0.9 to 1.1                        |
# | Contrast   | 50%    | 0.9 to 1.1                        |
# | Saturation | 50%    | 0.9 to 1.1                        |
# | Hue        | 50%    | 0.05                              |
# | Rotate     | 50%    | 17 degrees * normal distribution  |
# | Shear      | 50%    | 5.5 degrees * normal distribution |
# | Zoom Out   | 33%    | 1.0 - (normal distribution / 8.5) |
# | Shift      | 33%    | 18 pixel * normal distribution    |
# | GridMask   | 50%    | 100 - 160 pixel black rectangle   |
# |            |        | with same pixel range for gap     |

# ## Rotate, shear, zoom, shift

# In[ ]:


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
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
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def transform(image):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SIZE[0]
    XDIM = DIM%2 #fix for size 331
    
    if np.random.randint(0, 2) == 0: # 50% chance
        rot = 17. * tf.random.normal([1],dtype='float32')
    else:
        rot = tf.constant([0],dtype='float32')
    
    if np.random.randint(0, 2) == 0: # 50% chance
        shr = 5.5 * tf.random.normal([1],dtype='float32') 
    else:
        shr = tf.constant([0],dtype='float32')
    
    if np.random.randint(0, 3) == 0: # 33% chance
        h_zoom = tf.random.normal([1],dtype='float32')/8.5
        if h_zoom > 0:
            h_zoom = 1.0 + h_zoom * -1
        else:
            h_zoom = 1.0 + h_zoom
    else:
        h_zoom = tf.constant([1],dtype='float32')
    
    if np.random.randint(0, 3) == 0: # 33% chance
        w_zoom = tf.random.normal([1],dtype='float32')/8.5
        if w_zoom > 0:
            w_zoom = 1.0 + w_zoom * -1
        else:
            w_zoom = 1.0 + w_zoom
    else:
        w_zoom = tf.constant([1],dtype='float32')
    
    if np.random.randint(0, 3) == 0: # 33% chance
        h_shift = 18. * tf.random.normal([1],dtype='float32') 
    else:
        h_shift = tf.constant([0],dtype='float32')
    
    if np.random.randint(0, 3) == 0: # 33% chance
        w_shift = 18. * tf.random.normal([1],dtype='float32') 
    else:
        w_shift = tf.constant([0],dtype='float32')
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3])


# ## GridMask

# In[ ]:


def transform_grid_mark(image, inv_mat, image_shape):
    h, w, c = image_shape
    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:,i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])

def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
          #transform to radian
        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero,
                                     -sin_val, cos_val, zero,
                                     zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform_grid_mark(image, rot_mat_inv, image_shape)


def GridMask():
    h, w = IMAGE_SIZE[0], IMAGE_SIZE[1]
    image_height, image_width = (h, w)
    d1 = 100
    d2 = 160
    rotate_angle = 45 # 1
    ratio = 0.5

    hh = int(np.ceil(np.sqrt(h*h+w*w)))
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh//d+1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges <0 , x_ranges > hh-1)
    y_clip_mask = tf.logical_or(y_ranges <0 , y_ranges > hh-1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)

    return mask

def apply_grid_mask(image):
    mask = GridMask()
    mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask, 'float32')


# ## Augmentation function & tf.image functions (flip, brightness, contrast, saturation, hue)

# In[ ]:


def image_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    if np.random.randint(0, 2) == 0: # 50% chance
        image = tf.image.random_brightness(image, 0.1)
    if np.random.randint(0, 2) == 0: # 50% chance
        image = tf.image.random_contrast(image, 0.9, 1.1)
    if np.random.randint(0, 2) == 0: # 50% chance
        image = tf.image.random_saturation(image, 0.9, 1.1)
    if np.random.randint(0, 2) == 0: # 50% chance
        image = tf.image.random_hue(image, 0.05)

    image = transform(image)

    if np.random.randint(0, 2) == 0:
        image = apply_grid_mask(image)

    return image, label


# # Show augmentated image

# In[ ]:


def show_augmented_image(same_image=True):
    row, col = 3, 5
    if same_image:
        all_elements = get_training_dataset(do_aug=False).unbatch()
        one_element = tf.data.Dataset.from_tensors( next(iter(all_elements)) )
        augmented_element = one_element.repeat().map(image_augmentation).batch(row*col)
        for img, label in augmented_element:
            plt.figure(figsize=(15,int(15*row/col)))
            for j in range(row*col):
                plt.subplot(row,col,j+1)
                plt.axis('off')
                plt.imshow(img[j,])
            plt.suptitle(CLASSES[label[0]])
            plt.show()
            break
    else:
        all_elements = get_training_dataset().unbatch()
        augmented_element = all_elements.batch(row*col)

        for img, label in augmented_element:
            plt.figure(figsize=(15,int(15*row/col)))
            for j in range(row*col):
                plt.subplot(row,col,j+1)
                plt.title(CLASSES[label[j]])
                plt.axis('off')
                plt.imshow(img[j,])
            plt.show()
            break


# In[ ]:


# run again to see different batch of image
show_augmented_image()


# In[ ]:


# run again to see different image
show_augmented_image(same_image=False)


# # Functions for model training

# In[ ]:


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D, GlobalAveragePooling2D, SpatialDropout2D


# In[ ]:


def plt_lr(epoch_count):
    if epoch_count > 50:
        epoch_count = 50
    
    rng = [i for i in range(epoch_count)]

    plt.figure()
    y = [lrfn(x) for x in rng]
    plt.title(f'Learning rate schedule: {y[0]} to {y[epoch_count-1]}')
    plt.plot(rng, y)

def plt_acc(h):
    plt.figure()
    plt.plot(h.history["sparse_categorical_accuracy"])
    if 'val_sparse_categorical_accuracy' in h.history:
        plt.plot(h.history["val_sparse_categorical_accuracy"]) 
        plt.legend(["training","validation"])       
    else:
        plt.legend(["training"])
    plt.xlabel("epoch")
    plt.title("Sparse Categorical Accuracy")
    plt.show()

def plt_loss(h):
    plt.figure()
    plt.plot(h.history["loss"])
    if 'val_loss' in h.history:
        plt.plot(h.history["val_loss"]) 
        plt.legend(["training","validation"])       
    else:
        plt.legend(["training"])
    plt.legend(["training","validation"])
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.show()


# In[ ]:


es_val_acc = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=0.001, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)

es_val_loss = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)

es_acc = tf.keras.callbacks.EarlyStopping(
    monitor='sparse_categorical_accuracy', min_delta=0.001, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)

es_loss = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)


# # Create model

# ## EfficientNetB7 model
# 
# | Layer     | Layer Type                   |
# | --------- | ---------------------------- |
# | 0         | input_1 (InputLayer)         |
# | 1         | stem_conv (Conv2D)           |
# | 2         | stem_bn (BatchNormalization) |
# | 3         | stem_activation (Activation) |
# | 4-49      | block1*                      |
# | 50 - 152  | block2*                      |
# | 153 - 255 | block3*                      |
# | 256 - 403 | block4*                      |
# | 404 - 551 | block5*                      |
# | 552 - 744 | block6*                      |
# | 745 - 802 | block7*                      |
# | 803       | top_conv (Conv2D)            |
# | 804       | top_bn (BatchNormalization)  |
# | 805       | top_activation (Activation)  |

# In[ ]:


with strategy.scope():
    efn7 = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    for layer in efn7.layers[:745]:
        layer.trainable = False
    for layer in efn7.layers[745:]:
        layer.trainable = True

    model = Sequential([
        efn7,
        GlobalAveragePooling2D(),
#         AveragePooling2D(pool_size=(8,8), strides=8, padding='valid'), # valid -> drop leftover pixel, same -> add padding
#         Flatten(),
#         Dense(512, activation='relu'),
        Dense(len(CLASSES), activation='softmax')
    ], name='b7-flower')
    
    from tensorflow.keras.applications.densenet import DenseNet201
    densenet201 = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
model.compile(optimizer=tfa.optimizers.LAMB(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# ## DenseNet 201 model
# 
# | Layer     | Layer Type                      |
# | --------- | ------------------------------- |
# | 0         | input_5 (InputLayer)            |
# | 1         | zero_padding2d (ZeroPadding2D)  |
# | 2 - 6     | [conv/pool]1*                   |
# | 7 - 52    | [conv/pool]2*                   |
# | 53 - 140  | [conv/pool]3*                   |
# | 141 - 480 | [conv/pool]4*                   |
# | 481 - 704 | [conv/pool]5*                   |
# | 705       | bn (BatchNormalization)         |
# | 706       | relu (Activation)               |

# In[ ]:


with strategy.scope():
    densenet201 = tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    for layer in densenet201.layers[:481]:
        layer.trainable = False
    for layer in densenet201.layers[481:]:
        layer.trainable = True

    model2 = Sequential([
        densenet201,
        GlobalAveragePooling2D(),
#         AveragePooling2D(pool_size=(8,8), strides=8, padding='valid'), # valid -> drop leftover pixel, same -> add padding
#         Flatten(),
#         Dense(512, activation='relu'),
        Dense(len(CLASSES), activation='softmax')
    ], name='dn201-flower')
    
model2.compile(optimizer=tfa.optimizers.LAMB(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model2.summary()


# # Warm up model

# In[ ]:


def lrfn(epoch):
    initial_lr = 0.0001
    current_lr = initial_lr + 0.00015 * epoch

    return current_lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
plt_lr(EPOCHS_WARMUP)


# In[ ]:


if VALIDATE_WARMUP:
    model.fit(
        get_training_dataset(do_aug=DO_AUG_WARMUP), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS_WARMUP,
        validation_data=get_validation_dataset(), callbacks=[lr_schedule], verbose=1
    )
else:
    model.fit(
        get_train_val_dataset(do_aug=DO_AUG_WARMUP), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS_WARMUP,
        callbacks=[lr_schedule], verbose=1
    )


# In[ ]:


h = model.history
plt_acc(h)
plt_loss(h)


# In[ ]:


if VALIDATE_WARMUP:
    model2.fit(
        get_training_dataset(do_aug=DO_AUG_WARMUP), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS_WARMUP,
        validation_data=get_validation_dataset(), callbacks=[lr_schedule], verbose=1
    )
else:
    model2.fit(
        get_train_val_dataset(do_aug=DO_AUG_WARMUP), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS_WARMUP,
        callbacks=[lr_schedule], verbose=1
    )


# In[ ]:


h = model2.history
plt_acc(h)
plt_loss(h)


# # Fine tune model

# In[ ]:


LR_START = 0.0006
LR_MAX = 0.003 #strategy.num_replicas_in_sync
LR_MIN = 0.0003
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.91

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

plt_lr(EPOCHS)


# In[ ]:


for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=tfa.optimizers.LAMB(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# In[ ]:


for layer in model2.layers:
    layer.trainable = True
model2.compile(optimizer=tfa.optimizers.LAMB(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model2.summary()


# In[ ]:


if VALIDATE:
    model.fit(
        get_training_dataset(do_aug=DO_AUG), steps_per_epoch=STEPS_PER_EPOCH, epochs=35,
        validation_data=get_validation_dataset(), callbacks=[es_acc, lr_schedule], verbose=1
    )
else:
    model.fit(
        get_train_val_dataset(do_aug=DO_AUG), steps_per_epoch=STEPS_PER_EPOCH, epochs=35,
        callbacks=[es_val_acc, lr_schedule], verbose=1
    )


# In[ ]:


h = model.history
plt_acc(h)
plt_loss(h)


# In[ ]:


if VALIDATE:
    model2.fit(
        get_training_dataset(do_aug=DO_AUG), steps_per_epoch=STEPS_PER_EPOCH, epochs=30,
        validation_data=get_validation_dataset(), callbacks=[es_acc, lr_schedule], verbose=1
    )
else:
    model2.fit(
        get_train_val_dataset(do_aug=DO_AUG), steps_per_epoch=STEPS_PER_EPOCH, epochs=30,
        callbacks=[es_val_acc, lr_schedule], verbose=1
    )


# In[ ]:


h = model2.history
plt_acc(h)
plt_loss(h)


# In[ ]:


POST_TRAINING_TIME_START = datetime.now()


# # Evaluate functions
# 
# Note : Evalation is useless when you include validation data for training

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
def display_confusion_matrix(cmat, score, precision, recall, acc):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.4f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.4f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.4f} '.format(recall)
    if recall is not None:
        titlestring += '\naccuracy = {:.4f} '.format(acc)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()

def evaluate(model):
    cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
    
    if type(model) == list:
        cm_model_pred = model[0].predict(images_ds)
        cm_model2_pred = model[1].predict(images_ds)
        cm_probabilities = (cm_model_pred * 0.56) + (cm_model2_pred * 0.44)
    else:
        cm_probabilities = model.predict(images_ds)
    cm_predictions = np.argmax(cm_probabilities, axis=-1)
    print(f'Correct   labels: {cm_correct_labels.shape} {cm_correct_labels}')
    print(f'Predicated labels: {cm_predictions.shape} {cm_predictions}')

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    acc = accuracy_score(cm_correct_labels, cm_predictions)
    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
    
    print(classification_report(cm_correct_labels, cm_predictions, labels=range(len(CLASSES))))

    print(f'F1 score: {score}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {acc}')
    
    return cmat, score, precision, recall, acc


# # Evaluate Model

# ## Evaluate Model 1 (EfficientNetB7)

# In[ ]:


if VALIDATE:
    cmat, score, precision, recall, acc = evaluate(model)


# In[ ]:


if VALIDATE:
    display_confusion_matrix(cmat, score, precision, recall, acc)


# ## Evaluate Model 2 (DensetNet 201)

# In[ ]:


if VALIDATE:
    cmat, score, precision, recall, acc = evaluate(model2)


# In[ ]:


if VALIDATE:
    display_confusion_matrix(cmat, score, precision, recall, acc)


# ## Evaluate Ensemble model

# In[ ]:


if VALIDATE:
    cmat, score, precision, recall, acc = evaluate([model, model2])


# In[ ]:


if VALIDATE:
    display_confusion_matrix(cmat, score, precision, recall, acc)


# # Visual Model Evaluation

# In[ ]:


def show_predict_val(m):
    row, col = 3, 5
    
    dataset = get_validation_dataset()
    dataset = dataset.unbatch().batch(row * col)
    images, labels = next(iter(dataset))

    probabilities = m.predict(images)
    predictions = np.argmax(probabilities, axis=-1)
    
    plt.figure(figsize=(15,int(15*row/col)))
    for i in range(row*col):        
        plt.subplot(row,col,i+1)
        
        pred = CLASSES[predictions[i]]
        real = CLASSES[labels[i]]
        if pred == real:
            plt.title(f'{pred} [OK]')
        else:
            plt.title(f'{pred} [NO -> {real}]', color='red')

        plt.axis('off')
        plt.imshow(images[i,])
    
    plt.show()


# In[ ]:


if VALIDATE:
    show_predict_val(model)


# In[ ]:


if VALIDATE:
    show_predict_val(model2)


# # Submit Result

# In[ ]:


test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
model_pred = model.predict(test_images_ds)
model2_pred = model2.predict(test_images_ds)

probabilities = (model_pred * 0.56) + (model2_pred * 0.44)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
get_ipython().system('head submission.csv')


# In[ ]:


print(f'Post training time : {(datetime.now() - POST_TRAINING_TIME_START).total_seconds()} seconds')

