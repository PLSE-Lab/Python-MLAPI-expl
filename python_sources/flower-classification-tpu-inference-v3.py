#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('pip install -U --pre efficientnet')


# In[ ]:


import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.applications import DenseNet201
print("Tensorflow version " + tf.__version__)


# # Configurations

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


GCS_DS_PATH = 'gs://kds-a44744780cb28b908f65e7fa3b4a473ef199bb9ffe8048dbbf29ffeb'#'gs://flowers-public' 
#GCS_DS_PATH = KaggleDatasets().get_gcs_path() # 


# Configuration
IMAGE_SIZE = [512, 512]
EPOCHS1=25
EPOCHS2=25
BATCH_SIZE = 1 * 16 * strategy.num_replicas_in_sync


# # Custom LR schedule

# In[ ]:


LR_START = 0.0001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 5
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = np.random.random_sample() * LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


# In[ ]:


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# In[ ]:


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

# watch out for overfitting!
SKIP_VALIDATION = False
if SKIP_VALIDATION:
    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES


# In[ ]:


VALIDATION_FILENAMES_1 = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')[:int(len(VALIDATION_FILENAMES)/2)]
VALIDATION_FILENAMES_2 = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')[int(len(VALIDATION_FILENAMES)/2):]

TRAINING_FILENAMES_1 = TRAINING_FILENAMES + VALIDATION_FILENAMES_1
TRAINING_FILENAMES_2 = TRAINING_FILENAMES + VALIDATION_FILENAMES_2

NUM_TRAINING_IMAGES_1 = count_data_items(TRAINING_FILENAMES_1)
NUM_TRAINING_IMAGES_2 = count_data_items(TRAINING_FILENAMES_2)

NUM_VALIDATION_IMAGES = (1 - SKIP_VALIDATION) * count_data_items(VALIDATION_FILENAMES)
NUM_VALIDATION_IMAGES_1 = (1 - SKIP_VALIDATION) * count_data_items(VALIDATION_FILENAMES_1)
NUM_VALIDATION_IMAGES_2 = (1 - SKIP_VALIDATION) * count_data_items(VALIDATION_FILENAMES_2)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH_1 = NUM_TRAINING_IMAGES_1 // BATCH_SIZE
STEPS_PER_EPOCH_2 = NUM_TRAINING_IMAGES_2 // BATCH_SIZE

print('Dataset: {} training_1 images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES_1, NUM_VALIDATION_IMAGES_1, NUM_TEST_IMAGES))
print('Dataset: {} training_2 images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES_2, NUM_VALIDATION_IMAGES_2, NUM_TEST_IMAGES))


# In[ ]:


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


# ## Datasets Functions

# In[ ]:


img_size=512
def random_blockout(img, sl=0.1, sh=0.2, rl=0.4):

    h, w, c = img_size, img_size, 3
    origin_area = tf.cast(h*w, tf.float32)

    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

    e_height_h = tf.minimum(e_size_h, h)
    e_width_h = tf.minimum(e_size_h, w)

    erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
    erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)

    erase_area = tf.zeros(shape=[erase_height, erase_width, c])
    erase_area = tf.cast(erase_area, tf.uint8)

    pad_h = h - erase_height
    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
    pad_bottom = pad_h - pad_top

    pad_w = w - erase_width
    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
    pad_right = pad_w - pad_left

    erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
    erase_mask = tf.squeeze(erase_mask, axis=0)
    erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

    return tf.cast(erased_img, img.dtype)


# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    #image = tf.io.decode_image(image_data, channels=3)
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
    #label = tf.one_hot(label, len(CLASSES))
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
    image= random_blockout(image)
    return image, label   


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    #dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2345)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset1():
    dataset = load_dataset(TRAINING_FILENAMES_1, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2345)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset2():
    dataset = load_dataset(TRAINING_FILENAMES_2, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2345)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset1(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES_1, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset2(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES_2, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def data_augment2(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    #image= random_blockout(image)
    return image, label   

def get_test_dataset2(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.map(data_augment2, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset1a(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES_1, labeled=True, ordered=ordered)
    dataset = dataset.map(data_augment2, num_parallel_calls=AUTO)        
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset2a(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES_2, labeled=True, ordered=ordered)
    dataset = dataset.map(data_augment2, num_parallel_calls=AUTO)    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


# # Training Model

# ## Load Model into TPU

# In[ ]:


# Need this line so Google will recite some incantations
# for Turing to magically load the model onto the TPU
with strategy.scope():
    enet = efn.EfficientNetB7(
        input_shape=(512, 512, 3),
        weights=None,#'noisy-student',
        include_top=False
    )

    model1 = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    
    
model1.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model1.load_weights('/kaggle/input/flower-class-tpu-efficientnet-training-v2/EfficientNetB7_validation2_35.h5')


# In[ ]:


with strategy.scope():
    enet = efn.EfficientNetB7(
        input_shape=(512, 512, 3),
        weights=None,#'noisy-student',
        include_top=False
    )

    model1b = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    
    
model1b.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model1b.load_weights('/kaggle/input/flower-class-tpu-efficientnet-training-v3/EfficientNetB7_validation2_40.h5')


# In[ ]:


with strategy.scope():
    enet = efn.EfficientNetB7(
        input_shape=(512, 512, 3),
        weights=None,#'noisy-student',
        include_top=False
    )

    model11 = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    
    
model11.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model11.load_weights('/kaggle/input/flower-class-tpu-efficientnet-training-v2/EfficientNetB7_validation1_35.h5')


# In[ ]:


with strategy.scope():
    enet = efn.EfficientNetB7(
        input_shape=(512, 512, 3),
        weights=None,#'noisy-student',
        include_top=False
    )

    model11b = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    
    
model11b.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model11b.load_weights('/kaggle/input/flower-class-tpu-efficientnet-training-v3/EfficientNetB7_validation1_40.h5')


# In[ ]:


# Need this line so Google will recite some incantations
# for Turing to magically load the model onto the TPU
with strategy.scope():
    rnet = DenseNet201(
        input_shape=(512, 512, 3),
        weights=None,#'imagenet',
        include_top=False
    )

    model2 = tf.keras.Sequential([
        rnet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
model2.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model2.load_weights('/kaggle/input/flower-class-tpu-densenet-training-v2/DenseNet201_validation2_35.h5')


# In[ ]:


with strategy.scope():
    rnet = DenseNet201(
        input_shape=(512, 512, 3),
        weights=None,#'imagenet',
        include_top=False
    )

    model2b = tf.keras.Sequential([
        rnet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
model2b.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model2b.load_weights('/kaggle/input/flower-class-tpu-densenet-training-v3/DenseNet201_validation2_50.h5')


# In[ ]:


with strategy.scope():
    rnet = DenseNet201(
        input_shape=(512, 512, 3),
        weights=None,#'imagenet',
        include_top=False
    )

    model22 = tf.keras.Sequential([
        rnet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
model22.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model22.load_weights('/kaggle/input/flower-class-tpu-densenet-training-v2/DenseNet201_validation1_35.h5')


# In[ ]:


with strategy.scope():
    rnet = DenseNet201(
        input_shape=(512, 512, 3),
        weights=None,#'imagenet',
        include_top=False
    )

    model22b = tf.keras.Sequential([
        rnet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
model22b.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
    loss = 'sparse_categorical_crossentropy',
    metrics=[]
)

model22b.load_weights('/kaggle/input/flower-class-tpu-densenet-training-v3/DenseNet201_validation1_50.h5')


# In[ ]:


from scipy.special import logit, expit
import pandas as pd


# In[ ]:


SKIP_VALIDATION = False

if not SKIP_VALIDATION:
    cmdataset1a = get_validation_dataset1a(ordered=True)
    cmdataset2a = get_validation_dataset2a(ordered=True)
    images_ds1a = cmdataset1a.map(lambda image, label: image)
    images_ds2a = cmdataset2a.map(lambda image, label: image)
    
    cmdataset1 = get_validation_dataset1(ordered=True)
    cmdataset2 = get_validation_dataset2(ordered=True)
    images_ds1 = cmdataset1.map(lambda image, label: image)
    images_ds2 = cmdataset2.map(lambda image, label: image)
    labels_ds1 = cmdataset1.map(lambda image, label: label).unbatch()
    labels_ds2 = cmdataset2.map(lambda image, label: label).unbatch()
    
    cm_correct_labels1 = next(iter(labels_ds1.batch(NUM_VALIDATION_IMAGES))).numpy()
    cm_correct_labels2 = next(iter(labels_ds2.batch(NUM_VALIDATION_IMAGES))).numpy()    
    cm_correct_labels = np.concatenate((cm_correct_labels1,cm_correct_labels2),axis=0)
    
    valid_1 = model11.predict(images_ds1)  
    valid_2 = model1.predict(images_ds2)    
    m = np.concatenate((valid_1,valid_2),axis=0) 
    
    valid_1 = model11.predict(images_ds1a)  
    valid_2 = model1.predict(images_ds2a)    
    m2 = np.concatenate((valid_1,valid_2),axis=0)
    
    for i in range(104):
        m[:,i] = np.nan_to_num(logit(m[:,i]))
        m2[:,i] = np.nan_to_num(logit(m2[:,i]))
        
    valid_1 = model11b.predict(images_ds1)  
    valid_2 = model1b.predict(images_ds2)    
    mb = np.concatenate((valid_1,valid_2),axis=0)
    
    valid_1 = model11b.predict(images_ds1a)  
    valid_2 = model1b.predict(images_ds2a)    
    m2b = np.concatenate((valid_1,valid_2),axis=0)

    for i in range(104):
        mb[:,i] = np.nan_to_num(logit(mb[:,i]))
        m2b[:,i] = np.nan_to_num(logit(m2b[:,i]))
        
        
    m = 0.5*m+0.5*mb        
    m2 = 0.5*m2+0.5*m2b 
    del mb,m2b    
    
    alpha_best = 0
    score_best = 0
    for alpha in np.linspace(0,1,101):
        alpha = np.round(alpha,2)
        cm_probabilities = alpha*m+(1-alpha)*m2
        if alpha == 0:
            probabilities = cm_probabilities
        cm_predictions = np.argmax(cm_probabilities, axis=-1)
        auxi = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
        print('alpha:',alpha,'--- f1_score:',auxi)
        if auxi > score_best:
            score_best = auxi
            alpha_best = alpha
            probabilities = cm_probabilities
    best_score1 = score_best        
    best_alpha1 = alpha_best
    cm_probabilities1 = probabilities
else:
    best_alpha1 = 0.50


# In[ ]:


print(best_score1)      
print(best_alpha1)


# In[ ]:


if not SKIP_VALIDATION:   
    valid_1 = model22.predict(images_ds1)  
    valid_2 = model2.predict(images_ds2)    
    mm = np.concatenate((valid_1,valid_2),axis=0)
    
    valid_1 = model22.predict(images_ds1a)  
    valid_2 = model2.predict(images_ds2a)    
    mm2 = np.concatenate((valid_1,valid_2),axis=0)

    for i in range(104):
        mm[:,i] = np.nan_to_num(logit(mm[:,i]))
        mm2[:,i] = np.nan_to_num(logit(mm2[:,i]))
        
        
    valid_1 = model22b.predict(images_ds1)  
    valid_2 = model2b.predict(images_ds2)    
    mmb = np.concatenate((valid_1,valid_2),axis=0)
    
    valid_1 = model22b.predict(images_ds1a)  
    valid_2 = model2b.predict(images_ds2a)    
    mm2b = np.concatenate((valid_1,valid_2),axis=0)
    
    for i in range(104):
        mmb[:,i] = np.nan_to_num(logit(mmb[:,i]))
        mm2b[:,i] = np.nan_to_num(logit(mm2b[:,i]))
        
        
    mm = 0.5*mm+0.5*mmb        
    mm2 = 0.5*mm2+0.5*mm2b 
    del mmb,mm2b    
    
    alpha_best = 0
    score_best = 0
    for alpha in np.linspace(0,1,101):
        alpha = np.round(alpha,2)
        cm_probabilities = alpha*mm+(1-alpha)*mm2
        if alpha == 0:
            probabilities = cm_probabilities
        cm_predictions = np.argmax(cm_probabilities, axis=-1)
        auxi = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
        print('alpha:',alpha,'--- f1_score:',auxi)
        if auxi > score_best:
            score_best = auxi
            alpha_best = alpha
            probabilities = cm_probabilities
    best_score2 = score_best        
    best_alpha2 = alpha_best
    cm_probabilities2 = probabilities
else:
    best_alpha2 = 0.50


# In[ ]:


print(best_score2)      
print(best_alpha2)


# In[ ]:


if not SKIP_VALIDATION:
    
    #for i in range(104):
    #    cm_probabilities1[:,i] = expit(cm_probabilities1[:,i])
    #    cm_probabilities2[:,i] = expit(cm_probabilities2[:,i])
    
    alpha_best = 0
    score_best = 0
    for alpha in np.linspace(0,1,101):
        alpha = np.round(alpha,2)
        cm_probabilities = alpha*cm_probabilities1+(1-alpha)*cm_probabilities2
        if alpha == 0:
            probabilities = cm_probabilities
        cm_predictions = np.argmax(cm_probabilities, axis=-1)
        auxi = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
        print('alpha:',alpha,'--- f1_score:',auxi)
        if auxi > score_best:
            score_best = auxi
            alpha_best = alpha
            probabilities = cm_probabilities
    best_score3 = score_best        
    best_alpha3 = alpha_best
    cm_probabilities3 = probabilities
else:
    best_alpha3 = 0.5


# In[ ]:


print(best_score3)      
print(best_alpha3)


# In[ ]:


test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.
test_ds2 = get_test_dataset2(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
test_images_ds2 = test_ds2.map(lambda image, idnum: image)

#############

m1 = model1.predict(test_images_ds) 
m11 = model11.predict(test_images_ds)

m1b = model1.predict(test_images_ds2) 
m11b = model11.predict(test_images_ds2)

m2 = model1b.predict(test_images_ds) 
m22 = model11b.predict(test_images_ds)

m2b = model1b.predict(test_images_ds2) 
m22b = model11b.predict(test_images_ds2)


for i in range(104):
    m1[:,i] = np.nan_to_num(logit(m1[:,i]))
    m11[:,i] = np.nan_to_num(logit(m11[:,i]))
    m1b[:,i] = np.nan_to_num(logit(m1b[:,i]))
    m11b[:,i] = np.nan_to_num(logit(m11b[:,i]))
    m2[:,i] = np.nan_to_num(logit(m2[:,i]))
    m22[:,i] = np.nan_to_num(logit(m22[:,i]))
    m2b[:,i] = np.nan_to_num(logit(m2b[:,i]))
    m22b[:,i] = np.nan_to_num(logit(m22b[:,i]))    
    


m1 = 0.25*m1+0.25*m11+0.25*m1b+0.25*m11b
m2 = 0.25*m2+0.25*m22+0.25*m2b+0.25*m22b
del m11,m22,m1b,m2b,m11b,m22b
    
cm_probabilities1 = best_alpha1*m1+(1-best_alpha1)*m2


#############

m1 = model2.predict(test_images_ds) 
m11 = model22.predict(test_images_ds)

m1b = model2.predict(test_images_ds2) 
m11b = model22.predict(test_images_ds2)

m2 = model2b.predict(test_images_ds) 
m22 = model22b.predict(test_images_ds)

m2b = model2b.predict(test_images_ds2) 
m22b = model22b.predict(test_images_ds2)


for i in range(104):
    m1[:,i] = np.nan_to_num(logit(m1[:,i]))
    m11[:,i] = np.nan_to_num(logit(m11[:,i]))
    m1b[:,i] = np.nan_to_num(logit(m1b[:,i]))
    m11b[:,i] = np.nan_to_num(logit(m11b[:,i]))
    m2[:,i] = np.nan_to_num(logit(m2[:,i]))
    m22[:,i] = np.nan_to_num(logit(m22[:,i]))
    m2b[:,i] = np.nan_to_num(logit(m2b[:,i]))
    m22b[:,i] = np.nan_to_num(logit(m22b[:,i]))    
    


m1 = 0.25*m1+0.25*m11+0.25*m1b+0.25*m11b
m2 = 0.25*m2+0.25*m22+0.25*m2b+0.25*m22b
del m11,m22,m1b,m2b,m11b,m22b

    
cm_probabilities2 = best_alpha2*m1+(1-best_alpha2)*m2

#############

cm_probabilities3 = best_alpha3*cm_probabilities1+(1-best_alpha3)*cm_probabilities2

predictions = np.argmax(cm_probabilities3, axis=-1)
print(predictions)

for i in range(104):
    cm_probabilities3[:,i] = expit(cm_probabilities3[:,i])
    #cm_probabilities2[:,i] = expit(cm_probabilities2[:,i])
    #cm_probabilities1[:,i] = expit(cm_probabilities1[:,i])    
    
    
pd.DataFrame(cm_probabilities3).to_csv('probabilities.csv', index=False)
#pd.DataFrame(cm_probabilities2).to_csv('probabilities2.csv', index=False)
#pd.DataFrame(cm_probabilities2).to_csv('probabilities1.csv', index=False)


print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')


# In[ ]:




