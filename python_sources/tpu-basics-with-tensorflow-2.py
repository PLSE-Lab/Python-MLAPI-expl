#!/usr/bin/env python
# coding: utf-8

# I am writing this kernel mostly to explain myself how TPUs work. TPUs or Tensor Processing Units are accelerators that are specifically designed to carry out tasks related to deep learning with ease. However, just setting up TPU as the accelerator in one's Kaggle kernel is not sufficient. I, thus, try to perform a simple image classification problem on the Flowers dataset, as suggested by [Phil Culliton](https://www.kaggle.com/philculliton).

# In[ ]:


import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import numpy as np
import matplotlib.pyplot as plt

print("Tensorflow version " + tf.__version__)


# The following code requests the system for a TPU in the `try` block. If the TPU is available, well and good. If not, the system allocated either GPU or CPU to work with the code. Thus, it is essential to check or detect which accelerator is being used.

# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# `REPLICAS: 8` lets us know that this kernel will be using something similar to 8 physical TPUs accross the board. Phil explains something extremely important for handling TPUs. It is that they require the data to be stored in a separate location, sort of co-mapped with the hardware of the TPU. In other words, they need to be stored in a container or "bucket", quite close to the TPU. So the dataset is shipped to the bucket next to the TPU. And hence, `KaggleDatasets()` transfers the dataset to the GCS (Google Cloud Services) bucket.

# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path()


# Now that all pre-requisites are ready, we can carry out the deep learning tasks efficiently. The image size is kept to be `[192,192]`, at which a GPU will easily run out of memory. I set the number of generations to be 50 and the batch size to be 16 per hardware. This means that a specific number of pieces of data get sent to the TPU as a particular instance of time.

# In[ ]:


IMAGE_SIZE = [192, 192]
EPOCHS = 50
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

NUM_TRAINING_IMAGES = 12753
NUM_TEST_IMAGES = 7382
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
AUTO = tf.data.experimental.AUTOTUNE


# It is important to know that a single TPU board contains 4 TPU chips and each TPU chip contains 2 TPU cores. These cores are where all the matrix multiplications happen with regard to neural networks. These hardware structures are designed to carry out matrix multiplications on extremely large datasets, but at the same time, very fast. This is why they are ideal for using in Deep Learning since we can thus iterate our models for more time if we train them quite fast.

# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_training_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec'), labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()


# However, that does not mean that we can take all data and dump it onto the TPU. There ought to be a meticulous handling of this data and a proper way in which data is delivered. This is made possible thanks to `tfrecords` which takes, in this case, the key pixels from the image and shoves it into a file for the TPU and sends it from the bucket like so. This helps in the maximum utilisation of the TPU, mostly because, if the way in which data is delivered is not optimised, it becomes a bottleneck for the efficient usage of the TPU.

# Now, we train the model using a pre-trained model which further helps in saving unnecessary time.

# In[ ]:


with strategy.scope():    
    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(104, activation='softmax')
    ])
        
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

historical = model.fit(training_dataset, 
          steps_per_epoch=STEPS_PER_EPOCH, 
          epochs=EPOCHS, 
          validation_data=validation_dataset)


# In[ ]:


plt.plot(historical.history['loss'])
plt.plot(historical.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])


# In[ ]:


plt.plot(historical.history['sparse_categorical_accuracy'])
plt.plot(historical.history['val_sparse_categorical_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Sparse Categorical Accuracy')
plt.legend(['Accuracy', 'Validation Accuracy'])


# And so, there we go, in around 200 seconds, the model loss and accuracy are converged.

# In[ ]:


test_ds = get_test_dataset(ordered=True)

print('Computing predictions')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

