#!/usr/bin/env python
# coding: utf-8

# 
# The purpose of this notebook is to show the building blocks of ResNet50 and how to train it on TPU. For the purpose of demonstration this shows only for 10 epoch. In competition scenario, one would either train for longer epochs, handle class imbalance or maybe use Transfer Learning for ResNet50 with imagenet weights.
# 
# This notebooks builds upon the work of two popular notebooks in this competition. Please upvote these notebooks for the amazing work the authors have done there:
# 
# Credits:
# 
# https://www.kaggle.com/philculliton/a-simple-petals-tf-2-2-notebook
# 
# 
# https://www.kaggle.com/dimitreoliveira/flower-classification-with-tpus-eda-and-baseline/data

# In[ ]:


import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import numpy as np
import cv2
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

print("Tensorflow version " + tf.__version__)


# **Set up the TPU**

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


# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [192, 192] 
EPOCHS = 10
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
NUM_TRAINING_IMAGES = 12753
NUM_TEST_IMAGES = 7382
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE


# **Setting up some Helper Functions**
# 

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

    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec'), labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()


# **Creating the ResNet50 from scratch**
# 
# 
# Credits: Prof. Andrew Ng's course on Deep learning
# 
# The ResNet50 Architecture can be shown at a high level as shown by the diagram below. This has two main building blocks:
# 1. Identity Block
# 2. Convolutional Block

# ![](data:image/png;base64,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)

# The biggest strength of ResNet lies in the skip connection or shortcut. The skip connection can be shown as follows:
# 
# Image Credit: deeplearning.ai

# ![](data:image/png;base64,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)

# **Let's build these blocks one by one and bring them together**

# In[ ]:


def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
        
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
        
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
        
    return X
def convolutional_block(X, f, filters, stage, block, s = 2):
        
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
   
    return X
def ResNet50(input_shape = (64, 64, 3), classes = 104):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model


# In[ ]:


with strategy.scope():    
    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    #pretrained_model.trainable = False # tramsfer learning
    
    model = ResNet50(input_shape = [*IMAGE_SIZE, 3], classes = 104)
        
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)


# In[ ]:


model.summary()


# In[ ]:


historical = model.fit(training_dataset, 
          steps_per_epoch=STEPS_PER_EPOCH, 
          epochs=EPOCHS, 
          validation_data=validation_dataset)


# **Generate Submissions**

# In[ ]:


test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

