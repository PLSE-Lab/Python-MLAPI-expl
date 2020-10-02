#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 

# In[ ]:


import math, re, os, time
import datetime
import tensorflow as tf

from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)
tf.config.optimizer.set_jit(True)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

import numpy as np
from collections import namedtuple
from collections import Counter
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

get_ipython().system('pip install tensorflow-addons')
import tensorflow_addons as tfa

get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn

import gc
gc.enable()


# In[ ]:


ls -l /kaggle/input/keras-pretrained-models


# In[ ]:


ls -l /kaggle/input/


# # TPU or GPU detection

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


# # Competition data access
# TPUs read data directly from Google Cloud Storage (GCS). This Kaggle utility will copy the dataset to a GCS bucket co-located with the TPU. If you have multiple datasets attached to the notebook, you can pass the name of a specific dataset to the get_gcs_path function. The name of the dataset is the name of the directory it is mounted in. Use `!ls /kaggle/input/` to list attached datasets.

# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# # Configuration

# In[ ]:


EPOCHS = 3

EPOCH_START_TRAIN_ALL = 2
EPOCH_END_TRAIN_ALL = 2
EPOCH_SAVING_START = 2

include_additional_files = True

image_size = 192
IMAGE_SIZE = [image_size, image_size]

OVERSAMPLE = True
AUGUMENTATION = True

# We want each class occur at least (approximately) `TARGET_MIN_COUNTING` times
TARGET_MIN_COUNTING = 1000
if not OVERSAMPLE:
    TARGET_MIN_COUNTING = 1

backend_names = [
    "EfficientNetB7",
    "DenseNet201",
    "ResNet152V2",
    "Xception"
]

backend_mapping = {
    "EfficientNetB7": efn.EfficientNetB7,
    "DenseNet201": tf.keras.applications.DenseNet201,
    "ResNet152V2": tf.keras.applications.ResNet152V2,    
    "Xception": tf.keras.applications.Xception
}    
    
# backend_name = "Xception"
backend_name = "EfficientNetB7"
backend = backend_mapping[backend_name]
    

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

# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 1e-5
LR_MAX = 5e-5 * strategy.num_replicas_in_sync
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.85
        
@tf.function
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# ## More Images

# In[ ]:


MORE_IMAGES_GCS_DS_PATH = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')

MOREIMAGES_PATH_SELECT = {
    192: '/tfrecords-jpeg-192x192',
    224: '/tfrecords-jpeg-224x224',
    331: '/tfrecords-jpeg-331x331',
    512: '/tfrecords-jpeg-512x512'
}
MOREIMAGES_PATH = MOREIMAGES_PATH_SELECT[IMAGE_SIZE[0]]

IMAGENET_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/imagenet' + MOREIMAGES_PATH + '/*.tfrec')
INATURELIST_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/inaturalist' + MOREIMAGES_PATH + '/*.tfrec')
OPENIMAGE_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/openimage' + MOREIMAGES_PATH + '/*.tfrec')
OXFORD_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/oxford_102' + MOREIMAGES_PATH + '/*.tfrec')
TENSORFLOW_FILES = tf.io.gfile.glob(MORE_IMAGES_GCS_DS_PATH + '/tf_flowers' + MOREIMAGES_PATH + '/*.tfrec')
ADDITIONAL_TRAINING_FILENAMES = IMAGENET_FILES + INATURELIST_FILES + OPENIMAGE_FILES + OXFORD_FILES + TENSORFLOW_FILES

if include_additional_files:
    TRAINING_FILENAMES = TRAINING_FILENAMES + ADDITIONAL_TRAINING_FILENAMES


# ## Visualization utilities
# data -> pixels, nothing of much interest for the machine learning practitioner in this section.

# In[ ]:


# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        ### title = '' if label is None else CLASSES[label]
        title = ""
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
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
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
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


# # Datasets

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

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset(batch_size):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    ### dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(12753)
    dataset = dataset.batch(batch_size, drop_remainder=True)  # slighly faster with fixed tensor sizes
    dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset

def get_validation_dataset(batch_size, ordered=False, repeated=False):
    
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    
    if repeated:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(3712)
        
    dataset = dataset.batch(batch_size, drop_remainder=repeated) # slighly faster with fixed tensor sizes
    
    # dataset = dataset.cache()  # seems this is problematic in the setting of this kernel
    
    dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset

def get_test_dataset(batch_size, ordered=True):
    
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))


# ## Oversample
# 
# Also include Chris Deotte's data augmentation

# ## 1 - Get labels and their countings

# In[ ]:


# Get labels and their countings

def get_training_dataset_raw():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
    return dataset


raw_training_dataset = get_training_dataset_raw()

label_counter = Counter()
for images, labels in raw_training_dataset:
    label_counter.update([labels.numpy()])

del raw_training_dataset    
    
label_counting_sorted = label_counter.most_common()

NUM_TRAINING_IMAGES = sum([x[1] for x in label_counting_sorted])
print("number of examples in the original training dataset: {}".format(NUM_TRAINING_IMAGES))

print("labels in the original training dataset, sorted by occurrence")
label_counting_sorted


# ## 2 - Define the number of repetitions for each class
# 

# In[ ]:


def get_num_of_repetition_for_class(class_id):
    
    counting = label_counter[class_id]
    if counting >= TARGET_MIN_COUNTING:
        return 1.0
    
    num_to_repeat = TARGET_MIN_COUNTING / counting
    
    return num_to_repeat

numbers_of_repetition_for_classes = {class_id: get_num_of_repetition_for_class(class_id) for class_id in range(104)}

print("number of repetitions for each class (if > 1)")
{k: v for k, v in sorted(numbers_of_repetition_for_classes.items(), key=lambda item: item[1], reverse=True) if v > 1}


# ## 3 - Define the number of repetitions for each training example

# In[ ]:


# This will be called later in `get_training_dataset_with_oversample()`

keys_tensor = tf.constant([k for k in numbers_of_repetition_for_classes])
vals_tensor = tf.constant([numbers_of_repetition_for_classes[k] for k in numbers_of_repetition_for_classes])
table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

def get_num_of_repetition_for_example(training_example):
    
    _, label = training_example
    
    num_to_repeat = table.lookup(label)
    num_to_repeat_integral = tf.cast(int(num_to_repeat), tf.float32)
    residue = num_to_repeat - num_to_repeat_integral
    
    num_to_repeat = num_to_repeat_integral + tf.cast(tf.random.uniform(shape=()) <= residue, tf.float32)
    
    return tf.cast(num_to_repeat, tf.int64)


# ## 4 - Use data augmentation to avoid (exactly) same images appear too many times

# ## Transform labels

# In[ ]:


def label_transform(images, labels):

    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, len(CLASSES))

    return images, labels


# ## Rotation

# In[ ]:


def get_batch_transformatioin_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """Returns a tf.Tensor of shape (batch_size, 3, 3) with each element along the 1st axis being
       an image transformation matrix (which transforms indicies).

    Args:
        rotation: 1-D Tensor with shape [batch_size].
        shear: 1-D Tensor with shape [batch_size].
        height_zoom: 1-D Tensor with shape [batch_size].
        width_zoom: 1-D Tensor with shape [batch_size].
        height_shift: 1-D Tensor with shape [batch_size].
        width_shift: 1-D Tensor with shape [batch_size].
        
    Returns:
        A 3-D Tensor with shape [batch_size, 3, 3].
    """    

    # A trick to get batch_size
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(rotation)), tf.int32)    
    
    # CONVERT DEGREES TO RADIANS
    rotation = tf.constant(math.pi) * rotation / 180.0
    shear = tf.constant(math.pi) * shear / 180.0

    # shape = (batch_size,)
    one = tf.ones_like(rotation, dtype=tf.float32)
    zero = tf.zeros_like(rotation, dtype=tf.float32)
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation) # shape = (batch_size,)
    s1 = tf.math.sin(rotation) # shape = (batch_size,)

    # Intermediate matrix for rotation, shape = (9, batch_size) 
    rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    rotation_matrix_temp = tf.transpose(rotation_matrix_temp)
    # Fianl rotation matrix, shape = (batch_size, 3, 3)
    rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear) # shape = (batch_size,)
    s2 = tf.math.sin(shear) # shape = (batch_size,)
    
    # Intermediate matrix for shear, shape = (9, batch_size) 
    shear_matrix_temp = tf.stack([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    shear_matrix_temp = tf.transpose(shear_matrix_temp)
    # Fianl shear matrix, shape = (batch_size, 3, 3)
    shear_matrix = tf.reshape(shear_matrix_temp, shape=(batch_size, 3, 3))    
    

    # ZOOM MATRIX
    
    # Intermediate matrix for zoom, shape = (9, batch_size) 
    zoom_matrix_temp = tf.stack([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    zoom_matrix_temp = tf.transpose(zoom_matrix_temp)
    # Fianl zoom matrix, shape = (batch_size, 3, 3)
    zoom_matrix = tf.reshape(zoom_matrix_temp, shape=(batch_size, 3, 3))
    
    # SHIFT MATRIX
    
    # Intermediate matrix for shift, shape = (9, batch_size) 
    shift_matrix_temp = tf.stack([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    shift_matrix_temp = tf.transpose(shift_matrix_temp)
    # Fianl shift matrix, shape = (batch_size, 3, 3)
    shift_matrix = tf.reshape(shift_matrix_temp, shape=(batch_size, 3, 3))    
        
    return tf.linalg.matmul(tf.linalg.matmul(rotation_matrix, shear_matrix), tf.linalg.matmul(zoom_matrix, shift_matrix))


def basic_transform(images, labels):
    """Returns a tf.Tensor of the same shape as `images`, represented a batch of randomly transformed images.

    Args:
        images: 4-D Tensor with shape (batch_size, width, hight, depth).
            Currently, `depth` can only be 3.
        
    Returns:
        A 4-D Tensor with the same shape as `images`.
    """ 
    
    # input `images`: a batch of images [batch_size, dim, dim, 3]
    # output: images randomly rotated, sheared, zoomed, and shifted
    DIM = images.shape[1]
    XDIM = DIM % 2  # fix for size 331
    
    # A trick to get batch_size
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)
    
    rot = 15.0 * tf.random.normal([batch_size], dtype='float32')
    shr = 5.0 * tf.random.normal([batch_size], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([batch_size], dtype='float32') / 10.0
    w_zoom = 1.0 + tf.random.normal([batch_size], dtype='float32') / 10.0
    h_shift = 16.0 * tf.random.normal([batch_size], dtype='float32') 
    w_shift = 16.0 * tf.random.normal([batch_size], dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    # shape = (batch_size, 3, 3)
    m = get_batch_transformatioin_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)  # shape = (DIM * DIM,)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])  # shape = (DIM * DIM,)
    z = tf.ones([DIM * DIM], dtype='int32')  # shape = (DIM * DIM,)
    idx = tf.stack([x, y, z])  # shape = (3, DIM * DIM)
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.linalg.matmul(m, tf.cast(idx, dtype='float32'))  # shape = (batch_size, 3, DIM ** 2)
    idx2 = K.cast(idx2, dtype='int32')  # shape = (batch_size, 3, DIM ** 2)
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)  # shape = (batch_size, 3, DIM ** 2)
    
    # FIND ORIGIN PIXEL VALUES
    # shape = (batch_size, 2, DIM ** 2)
    idx3 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 - 1 + idx2[:, 1, ]], axis=1)  
    
    # shape = (batch_size, DIM ** 2, 3)
    d = tf.gather_nd(images, tf.cast(tf.transpose(idx3, perm=[0, 2, 1]), dtype=tf.int64), batch_dims=1)
        
    # shape = (batch_size, DIM, DIM, 3)
    new_images = tf.reshape(d, (batch_size, DIM, DIM, 3))

    return new_images, labels


# ## CutMix + MixUp

# In[ ]:


def batch_cutmix(images, labels):
    
    PROBABILITY = 0.1
    
    # A trick to get batch_size
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)  
    
    DIM = IMAGE_SIZE[0]
    
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    
    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.random.uniform([batch_size], minval=0, maxval=batch_size, dtype=tf.int32)
    
    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.random.uniform([batch_size], minval=0, maxval=DIM, dtype=tf.int32)
    new_y = tf.random.uniform([batch_size], minval=0, maxval=DIM, dtype=tf.int32)
    
    # Random width for new images, shape = [batch_size]
    b = tf.random.uniform([batch_size], 0, 1) # this is beta dist with alpha=1.0
    new_width = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32) * do_cutmix
    
    # shape = [batch_size]
    new_y0 = tf.math.maximum(0, new_y - new_width // 2)
    new_y1 = tf.math.minimum(DIM, new_y + new_width // 2)
    new_x0 = tf.math.maximum(0, new_x - new_width // 2)
    new_x1 = tf.math.minimum(DIM, new_x + new_width // 2)
    
    # shape = [batch_size, DIM]
    target = tf.broadcast_to(tf.range(DIM), shape=(batch_size, DIM))
    
    # shape = [batch_size, DIM]
    mask_y = tf.math.logical_and(new_y0[:, tf.newaxis] <= target, target <= new_y1[:, tf.newaxis])
    
    # shape = [batch_size, DIM]
    mask_x = tf.math.logical_and(new_x0[:, tf.newaxis] <= target, target <= new_x1[:, tf.newaxis])    
    
    # shape = [batch_size, DIM, DIM]
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

    # All components are of shape [batch_size, DIM, DIM, 3]
    new_images =  images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3]) +                     tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3])

    a = tf.cast(new_width ** 2 / DIM ** 2, tf.float32)    
        
    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, len(CLASSES))
        
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)        
        
    return new_images, new_labels


def batch_mixup(images, labels):

    PROBABILITY = 0.1
    
    # A trick to get batch_size
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)  
    
    DIM = IMAGE_SIZE[0]

    # Do `batch_mixup` with a probability = `PROBABILITY`
    # This is a tensor containing 0 or 1 -- 0: no mixup.
    # shape = [batch_size]
    do_mixup = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.random.uniform([batch_size], minval=0, maxval=batch_size, dtype=tf.int32)
    
    # ratio of importance of the 2 images to be mixed up
    # shape = [batch_size]
    a = tf.random.uniform([batch_size], 0, 1) * tf.cast(do_mixup, tf.float32)  # this is beta dist with alpha=1.0
                
    # The second part corresponds to the images to be added to the original images `images`.
    new_images =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new_image_indices)

    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, len(CLASSES))
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)

    return new_images, new_labels


# ## Perspective Transformation

# In[ ]:


def random_4_points_2D_batch(height, width, batch_size):
    """Generate `batch_size * 4` random 2-D points.
    
    Each 4 points are inside a rectangle with the same center as the above rectangle but with side length being approximately 1.5 times.
    This choice is to avoid the image being transformed too disruptively.

    Each point is created first by making it close to the corresponding corner points determined by the rectangle, i.e
    [0, 0], [0, width], [height, width] and [height, 0] respectively. Then the 4 points are randomly shifted module 4.
    
    Args:
        height: 0-D tensor, height of a reference rectangle.
        width: 0-D tensor, width of a reference rectangle.
        batch_size: 0-D tensor, the number of 4 points to be generated
        
    Returns:
        points: 3-D tensor of shape [batch_size, 4, 2]
    """

    sy = height // 4
    sx = width // 4
        
    h, w = height, width
    
    y1 = tf.random.uniform(minval = -sy, maxval = sy, shape=[batch_size], dtype=tf.int64)
    x1 = tf.random.uniform(minval = -sx, maxval = sx, shape=[batch_size], dtype=tf.int64)
    
    y2 = tf.random.uniform(minval = -sy, maxval = sy, shape=[batch_size], dtype=tf.int64)
    x2 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[batch_size], dtype=tf.int64)

    y3 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[batch_size], dtype=tf.int64)
    x3 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[batch_size], dtype=tf.int64)    
    
    y4 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[batch_size], dtype=tf.int64)
    x4 = tf.random.uniform(minval = -sx, maxval = sx, shape=[batch_size], dtype=tf.int64)
            
    # shape = [4, 2, batch_size]
    points = tf.convert_to_tensor([[y1, x1], [y2, x2], [y3, x3], [y4, x4], [y1, x1], [y2, x2], [y3, x3], [y4, x4]])
    
    # shape = [batch_size, 4, 2]
    points = tf.transpose(points, perm=[2, 0, 1])
    
    # Trick to get random rotation
    # shape = [batch_size, 8, 2]
    points = tf.tile(points, multiples=[1, 2, 1])    
    # shape = [batch_size]
    start_indices = tf.random.uniform(minval=0, maxval=4, shape=[batch_size], dtype=tf.int64)
    # shape = [batch_size, 4]
    indices = start_indices[:, tf.newaxis] + tf.range(4, dtype=tf.int64)[tf.newaxis, :]
    # shape = [batch_size, 4, 2]
    indices = tf.stack([tf.broadcast_to(tf.range(batch_size, dtype=tf.int64)[:, tf.newaxis], shape=[batch_size, 4]), indices], axis=2)    
    
    # shape = [batch_size, 4, 2]
    points = tf.gather_nd(points, tf.cast(indices, dtype=tf.int64))
        
    return points


def random_4_point_transform_2D_batch(images):
    """Apply 4 point transformation on 2-D images `images` with randomly generated 4 points on target spaces.
    
    On source space, the 4 points are the corner points, i.e [0, 0], [0, width], [height, width] and [height, 0].
    On target space, the 4 points are randomly generated by `random_4_points_2D_batch()`.
    """

    batch_size, height, width = images.shape[:3]

    # 4 corner points in source image
    # shape = [batch_size, 4, 2]
    src_pts = tf.convert_to_tensor([[0, 0], [0, width], [height, width], [height, 0]])
    src_pts = tf.broadcast_to(src_pts, shape=[batch_size, 4, 2])

    # 4 points in target image
    # shape = [batch_size, 4, 2]
    tgt_pts = random_4_points_2D_batch(height, width, batch_size)
    
    tgt_images = four_point_transform_2D_batch(images, src_pts, tgt_pts)

    return tgt_images


def four_point_transform_2D_batch(images, src_pts, tgt_pts):
    """Apply 4 point transformation determined by `src_pts` and `tgt_pts` on 2-D images `images`.
    
    Args:
        images: 3-D tensor of shape [batch_size, height, width], or 4-D tensor of shape [batch_size, height, width, channels]
        src_pts: 3-D tensor of shape [batch_size, 4, 2]
        tgt_pts: 3-D tensor of shape [batch_size, 4, 2]
        
    Returns:
        A tensor with the same shape as `images`.
    """
    
    src_to_tgt_mat = get_src_to_tgt_mat_2D_batch(src_pts, tgt_pts)
    
    tgt_images = transform_by_perspective_matrix_2D_batch(images, src_to_tgt_mat)
    
    return tgt_images


def transform_by_perspective_matrix_2D_batch(images, src_to_tgt_mat):
    """Transform 2-D images by prespective transformation matrices
    
    Args:
        images: 3-D tensor of shape [batch_size, height, width], or 4-D tensor of shape [batch_size, height, width, channels]
        src_to_tgt_mat: 3-D tensor of shape [batch_size, 3, 3]. This is the transformation matrix mapping the source space to the target space.
        
    Returns:
        A tensor with the same shape as `image`.        
    """

    batch_size, height, width = images.shape[:3]

    # shape = (3, 3)
    tgt_to_src_mat = tf.linalg.inv(src_to_tgt_mat)
        
    # prepare y coordinates
    # shape = [height * width]
    ys = tf.repeat(tf.range(height), width) 
    
    # prepare x coordinates
    # shape = [height * width]
    xs = tf.tile(tf.range(width), [height])

    # prepare indices in target space
    # shape = [2, height * width]
    tgt_indices = tf.stack([ys, xs], axis=0)
    
    # Change to projective coordinates in the target space by adding ones
    # shape = [3, height * width]
    tgt_indices_homo = tf.concat([tgt_indices, tf.ones(shape=[1, height * width], dtype=tf.int32)], axis=0)
    
    # Get the corresponding projective coordinate in the source space
    # shape = [batch_size, 3, height * width]
    src_indices_homo = tf.linalg.matmul(tgt_to_src_mat, tf.cast(tgt_indices_homo, dtype=tf.float64))
    
    # normalize the projective coordinates
    # shape = [batch_size, 3, height * width]
    src_indices_normalized = src_indices_homo[:, :3, :] / src_indices_homo[:, 2:, :]
    
    # Get the affine coordinate by removing ones
    # shape = [batch_size, 2, height * width]
    src_indices_affine = tf.cast(src_indices_normalized, dtype=tf.int64)[:, :2, :]
    
    # Mask the points outside the range
    # shape = [batch_size, height * width]
    y_mask = tf.logical_and(src_indices_affine[:, 0] >= 0, src_indices_affine[:, 0] <= height - 1)
    x_mask = tf.logical_and(src_indices_affine[:, 1] >= 0, src_indices_affine[:, 1] <= width - 1)
    mask = tf.logical_and(y_mask, x_mask)
    
    # clip the coordinates
    # shape = [batch_size, 2, height * width]
    src_indices = tf.clip_by_value(src_indices_affine, clip_value_min=0, clip_value_max=[[height - 1], [width - 1]])
    
    # Get a collection of (y_coord, x_coord)
    # shape = [batch_size, height * width, 2]
    src_indices = tf.transpose(src_indices, perm=[0, 2, 1])
    
    # shape = [batch_size, height * width, channels]
    tgt_images = tf.gather_nd(images, tf.cast(src_indices, dtype=tf.int64), batch_dims=1)
    
    # Set pixel to 0 by using the mask
    tgt_images = tgt_images * tf.cast(mask[:, :, tf.newaxis], tf.float32)
    
    # reshape to [height, width, channels]
    tgt_images = tf.reshape(tgt_images, images.shape)

    return tgt_images


def get_src_to_tgt_mat_2D_batch(src_pts, tgt_pts):
    """Get the perspective transformation matrix from the source space to the target space, which maps the 4 source points to the 4 target points.
    
    Args:
        src_pts: 3-D tensor of shape [batch_size, 4, 2]
        tgt_pts: 3-D tensor of shape [batch_size, 4, 2]
        
    Returns:
        2-D tensor of shape [batch_size, 3, 3]
    """
    
    src_pts = tf.cast(src_pts, tf.int64)
    tgt_pts = tf.cast(tgt_pts, tf.int64)
    
    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `src_pts`
    # shape = [batch_size, 3, 3]
    src_mat = get_transformation_mat_2D_batch(src_pts)
    
    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `tgt_pts`
    # shape = [3, 3]
    tgt_mat = get_transformation_mat_2D_batch(tgt_pts)
    
    # The perspective transformation matrix mapping `src_pts` to `tgt_pts`
    # shape = [3, 3]
    src_to_tgt_mat = tf.linalg.matmul(tgt_mat, tf.linalg.inv(src_mat))
    
    return src_to_tgt_mat
  
    
def get_transformation_mat_2D_batch(four_pts):
    """Get the perspective transformation matrix from a space to another space, which maps the basis vectors and (1, 1, 1) to the 4 points defined by `four_pts`.
    
    Args:
        four_pts: 3-D tensor of shape [batch_size, 4, 2]
        
    Returns:
        3-D tensor of shape [batch_size, 3, 3]        
    """
    
    batch_size = four_pts.shape[0]
    
    # Change to projective coordinates by adding ones
    # shape = [batch_size, 3, 4]
    pts_homo = tf.transpose(tf.concat([four_pts, tf.ones(shape=[batch_size, 4, 1], dtype=tf.int64)], axis=-1), perm=[0, 2, 1])
    
    pts_homo = tf.cast(pts_homo, tf.float64)
    
    # Find `scalars` such that: src_pts_homo[:, 3:] * scalars == src_pts_homo[:, 3:]
    # shape = [batch_size 3, 3]
    inv_mat = tf.linalg.inv(pts_homo[:, :, :3])
    # shape = [batch_size, 3, 1]
    scalars = tf.linalg.matmul(inv_mat, pts_homo[:, :, 3:])
    
    # Get the matrix transforming unit vectors to the 4 source points
    # shape = [batch_size, 3, 3]    
    mat = tf.transpose(tf.transpose(pts_homo[:, :, :3], perm=[0, 2, 1]) * scalars, perm=[0, 2, 1])
    
    return mat


def batch_perspective(images, labels):
    
    PROBABILITY = 0.1
    
    # A trick to get batch_size
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)  
    
    # This is a tensor containing 0 or 1 -- 0: no perspective transformation.
    # shape = [batch_size]
    do_perspective = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.float32)
    
    new_images = random_4_point_transform_2D_batch(images)
    
    new_images = images * (1 - do_perspective)[:, tf.newaxis, tf.newaxis, tf.newaxis] + new_images * do_perspective[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    return new_images, labels


# ## 5 - A method to get oversampled training dataset
# 

# In[ ]:


def get_training_dataset_with_oversample(batch_size, shuffle_size=None, repeat_dataset=True, oversample=False, augumentation=False, drop_remainder=False):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    if oversample:
        dataset = dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(get_num_of_repetition_for_example((image, label))))

    if repeat_dataset:
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    
    if shuffle_size is not None:
        dataset = dataset.shuffle(shuffle_size)
    
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    
    if augumentation:
            
        # dataset = dataset.map(label_transform, num_parallel_calls=AUTO)    
        dataset = dataset.map(basic_transform, num_parallel_calls=AUTO)   
        # dataset = dataset.map(batch_cutmix, num_parallel_calls=AUTO)
        # dataset = dataset.map(batch_mixup, num_parallel_calls=AUTO)
        dataset = dataset.map(batch_perspective, num_parallel_calls=AUTO)
    
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset


# ## 6 - Check oversampled dataset

# In[ ]:


oversampled_training_dataset = get_training_dataset_with_oversample(batch_size=8, shuffle_size=None, repeat_dataset=False, oversample=True, augumentation=False)

label_counter_2 = Counter()
for images, labels in oversampled_training_dataset:
    label_counter_2.update(labels.numpy())

del oversampled_training_dataset

label_counting_sorted_2 = label_counter_2.most_common()

NUM_TRAINING_IMAGES_OVERSAMPLED = sum([x[1] for x in label_counting_sorted_2])
print("number of examples in the oversampled training dataset: {}".format(NUM_TRAINING_IMAGES_OVERSAMPLED))

print("labels in the oversampled training dataset, sorted by occurrence")
label_counting_sorted_2


# # Dataset visualizations

# In[ ]:


# Peek at training data
train_ds = get_training_dataset_with_oversample(batch_size=8, shuffle_size=NUM_TRAINING_IMAGES_OVERSAMPLED, repeat_dataset=True, oversample=OVERSAMPLE, augumentation=AUGUMENTATION, drop_remainder=True)
training_dataset = train_ds.unbatch().batch(8)
train_batch = iter(training_dataset)


# In[ ]:


# run this cell again for next set of images
display_batch_of_images(next(train_batch))


# In[ ]:


# Peek at training data
training_dataset = get_training_dataset_with_oversample(batch_size=8)
training_dataset = training_dataset.unbatch().batch(8)
train_batch = iter(training_dataset)

# Used below
for images, labels in training_dataset:
    dummy_images = images
    print(dummy_images)
    print(dummy_images.shape)
    break


# In[ ]:


# run this cell again for next set of images
display_batch_of_images(next(train_batch))


# In[ ]:


# peer at test data
test_dataset = get_test_dataset(batch_size=8)
test_dataset = test_dataset.unbatch().batch(8)
test_batch = iter(test_dataset)


# In[ ]:


# run this cell again for next set of images
display_batch_of_images(next(test_batch))


# # Batch Configuration

# In[ ]:


def set_batch_configuration(batch_size_per_replica, batches_per_update):

    with strategy.scope():

        # The number of examples for which the training procedure running on a single replica will compute the gradients in order to accumulate them.
        BATCH_SIZE_PER_REPLICA = batch_size_per_replica

        # The total number of examples for which the training procedure will compute the gradients in order to accumulate them.
        # This is also used for validation step.
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

        # Accumulate `BATCHES_PER_UPDATE` of gradients before updating the model's parameters.
        BATCHES_PER_UPDATE = batches_per_update

        # The number of examples for which the training procedure will update the model's parameters once.
        # This is the `effective` batch size, which will be used in tf.data.Dataset. 
        UPDATE_SIZE = BATCH_SIZE * BATCHES_PER_UPDATE

        # The number of parameter updates in 1 epoch
        UPDATES_PER_EPOCH = NUM_TRAINING_IMAGES_OVERSAMPLED // UPDATE_SIZE
        
        # The number of batches for a validation step.
        VALID_BATCHES_PER_EPOCH = NUM_VALIDATION_IMAGES // BATCH_SIZE
        
        return BATCH_SIZE_PER_REPLICA, BATCH_SIZE, BATCHES_PER_UPDATE, UPDATE_SIZE, UPDATES_PER_EPOCH, VALID_BATCHES_PER_EPOCH


# # Optimized custom training loop
# Optimized by calling the TPU less often and performing more steps per call
# 

# ## Soft Macro F1 Loss

# In[ ]:


def soft_f1_fn(labels_1_hot, prob_dist):

    tp = tf.math.reduce_sum(labels_1_hot * prob_dist, axis=0)
    fn = tf.math.reduce_sum(labels_1_hot * (1 - prob_dist), axis=0)
    fp = tf.math.reduce_sum((1 - labels_1_hot) * prob_dist, axis=0)
    
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-30)
    recall = tp / (tp + fn + 1e-30)
    precision = tp / (tp + fp + 1e-30)
    
    return f1, recall, precision


def soft_f1_from_probs(labels, prob_dist, n_classes):

    labels_1_hot = tf.one_hot(labels, depth=n_classes)

    return soft_f1_fn(labels_1_hot, prob_dist)


def soft_f1_from_probs_with_1_hot_labels(labels, prob_dist, n_classes):

    labels_1_hot = labels
    
    return soft_f1_fn(labels_1_hot, prob_dist)


def soft_f1_from_logits(labels, logits, n_classes):

    prob_dist = tf.math.softmax(logits, axis=-1)

    return soft_f1_from_probs(labels, prob_dist, n_classes)


def hard_f1_from_logits(labels, logits, n_classes):

    pred_labels = tf.math.argmax(logits, axis=-1)

    pred_labels_1_hot = tf.one_hot(pred_labels, depth=n_classes)

    return soft_f1_from_probs(labels, pred_labels_1_hot, n_classes)


def hard_f1_from_probs(labels, prob_dist, n_classes):

    pred_labels = tf.math.argmax(prob_dist, axis=-1)

    pred_labels_1_hot = tf.one_hot(pred_labels, depth=n_classes)

    return soft_f1_from_probs(labels, pred_labels_1_hot, n_classes)


def soft_f1_loss_from_logits(labels, logits, n_classes):
    
    f1_scores, recalls, precisions = soft_f1_from_logits(labels, logits, n_classes)
    f1_score = tf.math.reduce_sum(f1_scores)
    
    return 1 - f1_score

def soft_f1_loss_from_probs(labels, prob_dist, n_classes):
    
    f1_scores, recalls, precisions = soft_f1_from_probs(labels, prob_dist, n_classes)
    f1_score = tf.math.reduce_mean(f1_scores)
    
    return 1 - f1_score


def hard_f1_from_probs_with_1_hot_labels(labels, prob_dist, n_classes):

    pred_labels = tf.math.argmax(prob_dist, axis=-1)

    pred_labels_1_hot = tf.one_hot(pred_labels, depth=n_classes)

    return soft_f1_from_probs_with_1_hot_labels(labels, pred_labels_1_hot, n_classes)


def soft_f1_loss_from_probs_with_1_hot_labels(labels, prob_dist, n_classes):
    
    f1_scores, recalls, precisions = soft_f1_from_probs_with_1_hot_labels(labels, prob_dist, n_classes)
    f1_score = tf.math.reduce_mean(f1_scores)
    
    return 1 - f1_score


# ## Model

# In[ ]:


def set_model(learning_rate_scaling=1):

    with strategy.scope():

        weights = 'imagenet'
        ### weights = '/kaggle/input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pretrained_model = backend(weights=weights, include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = True  # False = transfer learning, True = fine-tuning

        class Flower_Classifier(tf.keras.models.Model):
            
            def __init__(self, backend):
                
                super(Flower_Classifier, self).__init__()

                self.backend = backend
                self.pooling = tf.keras.layers.GlobalAveragePooling2D(name='flower/pooling')
                ### self.linear = tf.keras.layers.Dense(len(CLASSES), name='linear', activation='relu')
                self.logit = tf.keras.layers.Dense(len(CLASSES), name='logit')
                self.prediction = tf.keras.layers.Softmax(dtype='float32', name='prediction')
                
            def train_call(self, images, training=False):
                
                embeddings = self.backend(images, training=training)
                pooling = self.pooling(embeddings)
                ### linear = self.linear(pooling)
                linear = pooling
                logit = self.logit(linear)
                prediction = self.prediction(logit)
                
                return prediction, pooling
                
            def call(self, images, training=False):
                
                prediction, pooling = self.train_call(images, training=training)
                return prediction
                
        model = Flower_Classifier(backend=pretrained_model)
        
        model(dummy_images)
        model.summary()
        
        embedding_dim = model.backend(dummy_images).shape[-1]

        # Instiate optimizer with learning rate schedule
        class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            
            def __init__(self, scaling):
                
                self.scaling = scaling
            
            def __call__(self, step):
                
                return self.scaling * lrfn(epoch=step // (UPDATES_PER_EPOCH))

        optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule(scaling=learning_rate_scaling))
        optimizer_final = tf.keras.optimizers.Adam(learning_rate=LRSchedule(scaling=1))

        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        optimizer_final = mixed_precision.LossScaleOptimizer(optimizer_final, loss_scale='dynamic')        
        
        # Instantiate metrics
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        train_loss = tf.keras.metrics.Sum()
        valid_loss = tf.keras.metrics.Sum()
        
        train_focal_loss = tf.keras.metrics.Sum()
        valid_focal_loss = tf.keras.metrics.Sum()
        
        train_hard_f1 = tf.keras.metrics.Sum()
        train_hard_recall = tf.keras.metrics.Sum()
        train_hard_precision = tf.keras.metrics.Sum()
        
        train_soft_f1 = tf.keras.metrics.Sum()
        train_soft_recall = tf.keras.metrics.Sum()
        train_soft_precision = tf.keras.metrics.Sum()        
        
        train_soft_f1_loss = tf.keras.metrics.Sum()
        
        # Loss
        # The recommendation from the Tensorflow custom training loop documentation is:
        # loss_fn = lambda a,b: tf.nn.compute_average_loss(tf.keras.losses.sparse_categorical_crossentropy(a,b), global_batch_size=BATCH_SIZE)
        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
        # This works too and shifts all the averaging to the training loop which is easier:
        
        loss_fn = tf.keras.losses.categorical_crossentropy
        
        loss_fn_focal = tfa.losses.sigmoid_focal_crossentropy
        
        loss_fn_sparse = tf.keras.losses.sparse_categorical_crossentropy
        
        return model, loss_fn_sparse, loss_fn, loss_fn_focal, optimizer, train_accuracy, train_loss, valid_accuracy, valid_loss, train_hard_f1, train_hard_recall, train_hard_precision, train_soft_f1, train_soft_recall, train_soft_precision, train_soft_f1_loss, optimizer_final, embedding_dim, train_focal_loss, valid_focal_loss


# ## Training routines

# In[ ]:


def set_routines():

    with strategy.scope():
        
        def train_step_1_forward(images, labels):
            
            probabilities, pooling = model.train_call(images, training=True)
            loss = loss_fn_sparse(labels, probabilities)
            loss_value = tf.math.reduce_sum(loss)
            
            focal_loss = loss_fn_focal(tf.one_hot(labels, len(CLASSES)), probabilities)
            focal_loss_value = tf.math.reduce_sum(focal_loss)
            
            loss_scaled = loss_value / UPDATE_SIZE
            focal_loss_scaled = focal_loss_value / UPDATE_SIZE
                        
            train_focal_loss.update_state(focal_loss_value)            
            
            loss_scaled += focal_loss_scaled
            
            return loss_scaled, pooling, probabilities, loss_value

        def train_step_1_forward_backward(images, labels, epoch):

            if epoch >= EPOCH_START_TRAIN_ALL and epoch <= EPOCH_END_TRAIN_ALL:
            
                with tf.GradientTape() as tape:
                    loss_scaled, pooling, probabilities, loss_value = train_step_1_forward(images, labels)
                    loss_dynamically_scaled = optimizer.get_scaled_loss(loss_scaled)
                dynamically_scaled_gradients = tape.gradient(loss_dynamically_scaled, model.trainable_variables)   
                grads = optimizer.get_unscaled_gradients(dynamically_scaled_gradients) 
        
            else:
                loss_scaled, pooling, probabilities, loss_value = train_step_1_forward(images, labels)
                grads = [tf.zeros_like(var, dtype=tf.float32) for var in model.trainable_variables]

            # update metrics
            train_accuracy.update_state(labels, probabilities)
            train_loss.update_state(loss_value)                
                
            return grads, pooling

        def train_step_1_update(batch, epoch):
            """
            """

            images, labels = batch
            
            accumulated_grads = [tf.zeros_like(var, dtype=tf.float32) for var in model.trainable_variables]
            
            # shape = [BATCH_SIZE_PER_REPLICA * BATCHES_PER_UPDATE, 2048]
            total_pooling = tf.constant(0.0, shape=(BATCH_SIZE_PER_REPLICA * BATCHES_PER_UPDATE, embedding_dim))
                                
            for batch_idx in tf.range(BATCHES_PER_UPDATE):

                # Take the 1st `BATCH_SIZE_PER_REPLICA` examples.
                small_images = images[:BATCH_SIZE_PER_REPLICA]
                small_labels = labels[:BATCH_SIZE_PER_REPLICA]      
                
                grads, pooling = train_step_1_forward_backward(small_images, small_labels, epoch)

                if epoch >= EPOCH_START_TRAIN_ALL and epoch <= EPOCH_END_TRAIN_ALL:
                    accumulated_grads = [x + y for x, y in zip(accumulated_grads, grads)]

                # Move the leading part to the end, so the shape is not changed.
                images = tf.concat([images[BATCH_SIZE_PER_REPLICA:], small_images], axis=0)
                labels = tf.concat([labels[BATCH_SIZE_PER_REPLICA:], small_labels], axis=0)
                                
                total_pooling = tf.concat([total_pooling[BATCH_SIZE_PER_REPLICA:], tf.cast(pooling, tf.float32)], axis=0)
                
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                    
                tape.watch(final_variables)
                    
                ### linear = model.linear(total_pooling)
                linear = total_pooling
                logit = model.logit(linear)
                
                # shape = [BATCH_SIZE_PER_REPLICA * BATCHES_PER_UPDATE, len(CLASSES)]
                probabilities = model.prediction(logit)

                soft_f1_loss = soft_f1_loss_from_probs(labels, probabilities, len(CLASSES))
                soft_f1_loss_scaled = soft_f1_loss / strategy.num_replicas_in_sync
                
                soft_f1_loss_dynamically_scaled = optimizer_final.get_scaled_loss(soft_f1_loss_scaled)

            dynamically_scaled_gradients = tape.gradient(soft_f1_loss_dynamically_scaled, final_variables)   
            final_grads = optimizer_final.get_unscaled_gradients(dynamically_scaled_gradients)  
            
            # hard_f1, hard_recall, hard_precision = hard_f1_from_probs_with_1_hot_labels(labels, probabilities, len(CLASSES))
            hard_f1, hard_recall, hard_precision = hard_f1_from_probs(labels, probabilities, len(CLASSES))
            
            hard_f1 = tf.reduce_mean(hard_f1)
            hard_recall = tf.reduce_mean(hard_recall)
            hard_precision = tf.reduce_mean(hard_precision)
            
            train_hard_f1.update_state(hard_f1)
            train_hard_recall.update_state(hard_recall)
            train_hard_precision.update_state(hard_precision)            
            
            # soft_f1, soft_recall, soft_precision = soft_f1_from_probs_with_1_hot_labels(labels, probabilities, len(CLASSES))
            soft_f1, soft_recall, soft_precision = soft_f1_from_probs(labels, probabilities, len(CLASSES))
            
            soft_f1 = tf.reduce_mean(soft_f1)
            soft_recall = tf.reduce_mean(soft_recall)
            soft_precision = tf.reduce_mean(soft_precision)            
            
            train_soft_f1.update_state(soft_f1)
            train_soft_recall.update_state(soft_recall)
            train_soft_precision.update_state(soft_precision)
            
            train_soft_f1_loss.update_state(soft_f1_loss_scaled)
            
            # Update the model's parameters.
            
            if epoch >= EPOCH_START_TRAIN_ALL and epoch <= EPOCH_END_TRAIN_ALL:
                optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
        
            optimizer_final.apply_gradients(zip(final_grads, final_variables))

            
        @tf.function
        def train_step_1_epoch(data_iter, epoch):

            for _ in tf.range(UPDATES_PER_EPOCH):  
                strategy.experimental_run_v2(train_step_1_update, args=(next(data_iter), epoch))
                
                
        @tf.function
        def valid_step(data_iter):
            
            def valid_step_fn(images, labels):
                
                probabilities, pooling = model.train_call(images, training=False)
                loss = tf.math.reduce_sum(loss_fn_sparse(labels, probabilities))
                focal_loss = tf.math.reduce_sum(loss_fn_focal(tf.one_hot(labels, len(CLASSES)), probabilities))                
                
                loss_scaled = loss / UPDATE_SIZE
                focal_loss_scaled = focal_loss / UPDATE_SIZE
                
                # update metrics
                valid_accuracy.update_state(labels, probabilities)
                
                valid_loss.update_state(loss)
                valid_focal_loss.update_state(focal_loss)

            for _ in tf.range(VALID_BATCHES_PER_EPOCH):
                strategy.experimental_run_v2(valid_step_fn, next(data_iter))                
                
    return train_step_1_epoch, valid_step


# ## Training with gradient accumulation with even higer effective batch size + larger learning rate

# In[ ]:


BATCH_SIZE_PER_REPLICA, BATCH_SIZE, BATCHES_PER_UPDATE, UPDATE_SIZE, UPDATES_PER_EPOCH, VALID_BATCHES_PER_EPOCH = set_batch_configuration(batch_size_per_replica=8, batches_per_update=64)
model, loss_fn_sparse, loss_fn, loss_fn_focal, optimizer, train_accuracy, train_loss, valid_accuracy, valid_loss, train_hard_f1, train_hard_recall, train_hard_precision, train_soft_f1, train_soft_recall, train_soft_precision, train_soft_f1_loss, optimizer_final, embedding_dim, train_focal_loss, valid_focal_loss = set_model(learning_rate_scaling=16)
train_step_1_epoch, valid_step = set_routines()

final_variables = []

for layer in model.layers:
    if layer.name in ['logit', 'linear']:
        for variable in layer.variables:
            final_variables.append(variable)

print("BATCH_SIZE_PER_REPLICA: {}".format(BATCH_SIZE_PER_REPLICA))
print("BATCH_SIZE: {}".format(BATCH_SIZE))
print("BATCHES_PER_UPDATE: {}".format(BATCHES_PER_UPDATE))
print("UPDATE_SIZE: {}".format(UPDATE_SIZE))
print("UPDATES_PER_EPOCH: {}".format(UPDATES_PER_EPOCH))
print("VALID_BATCHES_PER_EPOCH: {}".format(VALID_BATCHES_PER_EPOCH))


# In[ ]:


train_ds = get_training_dataset_with_oversample(batch_size=UPDATE_SIZE, shuffle_size=NUM_TRAINING_IMAGES_OVERSAMPLED, repeat_dataset=True, oversample=OVERSAMPLE, augumentation=AUGUMENTATION, drop_remainder=True)
train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
train_data_iter = iter(train_dist_ds)

# valid_ds = get_validation_dataset(batch_size=BATCH_SIZE, repeated=True)
# valid_dist_ds = strategy.experimental_distribute_dataset(valid_ds)
# valid_data_iter = iter(valid_dist_ds)

# valid_ds_2 = get_validation_dataset(batch_size=BATCH_SIZE, ordered=True)
# valid_images_ds = valid_ds_2.map(lambda image, label: image)
# valid_labels_ds = valid_ds_2.map(lambda image, label: label).unbatch()
# valid_labels = next(iter(valid_labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy().astype('U') # all in one batch
# valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.int32)

test_ds = get_test_dataset(batch_size=BATCH_SIZE, ordered=True)
test_images_ds = test_ds.map(lambda image, idnum: image)
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

# valid_steps = NUM_VALIDATION_IMAGES // BATCH_SIZE
# if NUM_VALIDATION_IMAGES % BATCH_SIZE > 0:
#     valid_steps += 1

test_steps = NUM_TEST_IMAGES // BATCH_SIZE
if NUM_TEST_IMAGES % BATCH_SIZE > 0:
    test_steps += 1

for epoch_idx in range(EPOCHS):
    
    s = datetime.datetime.now()
    
    epoch = tf.constant(epoch_idx + 1, dtype=tf.int32)
    
    train_step_1_epoch(train_data_iter, epoch)
    
    loss = train_loss.result() / (UPDATES_PER_EPOCH * UPDATE_SIZE)
    acc = train_accuracy.result()
    
    focal_loss = train_focal_loss.result() / (UPDATES_PER_EPOCH * UPDATE_SIZE)
    
    hard_f1 = train_hard_f1.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    hard_recall = train_hard_recall.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    hard_precision = train_hard_precision.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    
    soft_f1 = train_soft_f1.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    soft_recall = train_soft_recall.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    soft_precision = train_soft_precision.result() / (UPDATES_PER_EPOCH * strategy.num_replicas_in_sync)
    
    soft_f1_loss = train_soft_f1_loss.result() / UPDATES_PER_EPOCH
    
    print("epoch: {}".format(epoch_idx + 1))

    print("train loss: {}".format(loss))
    print("train accuracy: {}".format(acc))
    
    print("train focal loss: {}".format(focal_loss))    
    
    print("train hard f1: {}".format(hard_f1))
    print("train hard recall: {}".format(hard_recall))
    print("train hard precision: {}".format(hard_precision))    
    
    print("train soft f1: {}".format(soft_f1))
    print("train soft recall: {}".format(soft_recall))
    print("train soft precision: {}".format(soft_precision))        
    
    print("train soft f1 loss: {}".format(soft_f1_loss))
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    train_focal_loss.reset_states()
    
    train_hard_f1.reset_states()
    train_hard_recall.reset_states()
    train_hard_precision.reset_states()    
    
    train_soft_f1.reset_states()
    train_soft_recall.reset_states()
    train_soft_precision.reset_states()     
    
    train_soft_f1_loss.reset_states()

    e = datetime.datetime.now()
    print("elapsed: {}".format((e-s).total_seconds()))
    
#     valid_step(valid_data_iter)
    
#     val_loss = valid_loss.result() / (VALID_BATCHES_PER_EPOCH * BATCH_SIZE)
#     val_acc = valid_accuracy.result()    
    
#     val_focal_loss = valid_focal_loss.result() / (VALID_BATCHES_PER_EPOCH * BATCH_SIZE)

#     print("valid loss: {}".format(val_loss))
#     print("valid accuracy: {}".format(val_acc))
    
#     print("valid focal loss: {}".format(val_focal_loss))
     
#     valid_loss.reset_states()
#     valid_accuracy.reset_states()
    
#     valid_focal_loss.reset_states()

#     valid_probs = model.predict(valid_images_ds, steps=valid_steps)
#     valid_preds = np.argmax(valid_probs, axis=-1)

#     valid_hard_f1, valid_hard_recall, valid_hard_precision = hard_f1_from_probs(valid_labels, valid_probs, len(CLASSES))
    
#     valid_hard_f1 = tf.reduce_mean(valid_hard_f1)
#     valid_hard_recall = tf.reduce_mean(valid_hard_recall)
#     valid_hard_precision = tf.reduce_mean(valid_hard_precision)    
    
#     valid_soft_f1, valid_soft_recall, valid_soft_precision = soft_f1_from_probs(valid_labels, valid_probs, len(CLASSES))    
    
#     valid_soft_f1 = tf.reduce_mean(valid_soft_f1)
#     valid_soft_recall = tf.reduce_mean(valid_soft_recall)
#     valid_soft_precision = tf.reduce_mean(valid_soft_precision)
    
#     valid_soft_f1_loss = 1 - valid_soft_f1
    
#     print("valid hard f1: {}".format(valid_hard_f1))
#     print("valid hard recall: {}".format(valid_hard_recall))
#     print("valid hard precision: {}".format(valid_hard_precision))    
    
#     print("valid soft f1: {}".format(valid_soft_f1))
#     print("valid soft recall: {}".format(valid_soft_recall))
#     print("valid soft precision: {}".format(valid_soft_precision))    
    
#     print("valid soft f1 loss: {}".format(valid_soft_f1_loss))
    
    if (epoch_idx + 1) >= EPOCH_SAVING_START:
        
        ### model.save_weights("{}_epoch_{}.ckpt".format(backend_name, epoch_idx + 1))

        test_probs = model.predict(test_images_ds, steps=test_steps)
        test_preds = np.argmax(test_probs, axis=-1)

#         print('Generating valid_probs.txt file...')
#         with open('valid_probs_epoch_{}.txt'.format(epoch_idx + 1), "w", encoding="UTF-8") as fp:
#             np.savetxt(fp, valid_probs, delimiter=',', fmt='%10.6f')

        print('Generating test_probs.txt file...')
        with open('test_probs_epoch_{}.txt'.format(epoch_idx + 1), "w", encoding="UTF-8") as fp:
            np.savetxt(fp, test_probs, delimiter=',', fmt='%10.6f')

#         print('Generating valid_preds.csv file...')
#         np.savetxt('valid_preds_epoch_{}.csv'.format(epoch_idx + 1), np.rec.fromarrays([valid_labels, valid_preds]), fmt=['%d', '%d'], delimiter=',', header='label,pred', comments='')

        print('Generating submission.csv file...')
        np.savetxt('submission_epoch_{}.csv'.format(epoch_idx + 1), np.rec.fromarrays([test_ids, test_preds]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')    
    
    print("-" * 80)
    
del optimizer
del model
del train_step_1_epoch
gc.collect()
tf.keras.backend.clear_session()

