#!/usr/bin/env python
# coding: utf-8

# # Flowers Classification Step 2 with TPU
# 
# In the first step we did the below steps. (STEP 1)
# 
# 1. Reading the Data
# 2. Splitting data to train, val and test test
# 3. Displaying some of the sample images
# 4. Get the pretrained model VGG
# 5. Put the data in the model
# 6. Get the prediction
# 
# In the second Step, we will be adding the support of TPU and doing data Augmentation (Step 2)

# In[ ]:


# import required packages

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
AUTO = tf.data.experimental.AUTOTUNE


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


# # Constant
# Adding some of the constants so that can be used later in the jupyter notebook.
# Please note I am using low resolution images as I will be creating the entire pipeline on CPU and then train it on TPU.

# In[ ]:


from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# In[ ]:


IMAGE_SIZE = [512, 512] # the size of the images
EPOCS = 30
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
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
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']


# In[ ]:


GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]


# In[ ]:


# Splitting the data to train val and test set. Please note the dataset is in tfrec for serial processing.

train_set = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
val_set = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
test_set = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')


# List of all the tfrec files in the train set.
print('List of records for trainset')
train_set


# # Extracting data from tfRecord
# (https://www.tensorflow.org/tutorials/load_data/tfrecord)
# 
# As the dataset is in tfRec, we have to extract the features from the record.
# So in our tfRecord, we have combination of image and class.

# In[ ]:


# Function to parse the tf record and give us image, label combination.
def read_labeled_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    
    return image, label

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


# In[ ]:


# Load the dataset
def load_dataset(filesnames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filesnames, num_parallel_reads=AUTO)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset


# # Load Train, Val and Test dataset from TFRec

# In[ ]:


def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    return image, label

def data_augment(image,label):
    image,label = convert(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.adjust_saturation(image, 3)
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness

    return image,label

# Training Dataset
def get_training_dataset():
    dataset = load_dataset(train_set, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# Validation Dataset
def get_val_dataset(ordered=False):
    dataset = load_dataset(val_set, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

# Test dataset
def get_test_dataset(ordered=False):
    dataset = load_dataset(test_set, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# Convert images to numpy
def batch_to_numpy(data):
    images, labels = data;
    
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    
    return numpy_images, numpy_labels

def display_batch_of_images(databatch):
    images, labels = batch_to_numpy(databatch)
    
    plt.figure(figsize=(50, 50))
    for i, (image, label) in enumerate(zip(images, labels), start=1):
        plt.subplot(10, 10, i)
        fig = plt.imshow(image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.title(label, fontsize=28)
    plt.show()
    
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image


# In[ ]:


#Load Train, Val and Test dataset from TFRec

training_dataset = get_training_dataset()
validation_datset = get_val_dataset()
test_dataset = get_test_dataset()


# # Plot my dataset

# In[ ]:


# Fetch 20 images for the training set
training_dataset = training_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)

# Plotting single image
images, labels = batch_to_numpy(next(train_batch));

# get the first image
image = images[0].squeeze()
plt.imshow(image)


# In[ ]:


# display multiple images
display_batch_of_images(next(train_batch))


# In[ ]:


import re
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(train_set)
NUM_VALIDATION_IMAGES = count_data_items(val_set)
NUM_TEST_IMAGES = count_data_items(test_set)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))


# In[ ]:


with strategy.scope():
    pretrained_model = tf.keras.applications.DenseNet201(
        weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    
    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = True
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])


# In[ ]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
 )

model.summary()


# In[ ]:


history = model.fit(
    get_training_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCS,
    validation_data=get_val_dataset()
)


# In[ ]:


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


test_ds = get_test_dataset() # since we are splitting the dataset and iterating separately on images and ids, order matters.

test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)


# In[ ]:


print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
get_ipython().system('head submission.csv')


# Credits: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

# In[ ]:




