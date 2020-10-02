#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import os
import argparse
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras import layers as L


import efficientnet.tfkeras as efn


# In[ ]:


def normalize(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])
    return image


# In[ ]:


def get_model(input_size, backbone='efficientnet-b0', weights='imagenet', tta=False):
    print(f'Using backbone {backbone} and weights {weights}')
    x = L.Input(shape=input_size, name='imgs', dtype='float32')
    y = normalize(x)
    
    if backbone.startswith('efficientnet'):
        model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')
        
    y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)
    y = L.GlobalAveragePooling2D()(y)
    y = L.Dropout(0.05)(y)
    # 1292 out 1295 are present at bottom layer
    y = L.BatchNormalization()(y)
    y = L.Dense(1292, activation='softmax')(y)
    model = tf.keras.Model(x, y)
    
    if tta:
        assert False, 'This doesn not make sense'
        x_flip = tf.reverse(x, [2])  # 'NHWC'
        y_tta = tf.add(model(x), model(x_flip)) / 2.0
        tta_model = tf.keras.Model(x, y_tta)
        return model, tta_model

    return model


# In[ ]:


# Read mixup paper
def mixup(img_batch, label_batch, bs):
    weight = tf.random.uniform([bs])
    x_weight = tf.reshape(weight, [bs,1,1,1])
    y_weight = tf.reshape(weight, [bs,1])
    index = tf.random.shuffle(tf.range(bs, dtype=tf.int32))
    x1, x2 = img_batch, tf.gather(img_batch, index)
    img_batch = x1* x_weight + x2*(1. - x_weight)
    y1, y2 = label_batch, tf.gather(label_batch, index)
    label_batch = y1* y_weight + y2*(1. - y_weight)
    return img_batch, label_batch


# In[ ]:


def get_strategy():
  # Detect hardware, return appropriate distribution strategy
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
  except ValueError:
    tpu = None

  if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
  else:
    strategy = tf.distribute.get_strategy()

  print('REPLICAS: ', strategy.num_replicas_in_sync)
  return strategy



# In[ ]:


def one_hot(image, label):
  label = tf.one_hot(label, 1292)
  return image, label


# In[ ]:


def read_tfrecords(example, input_size):
  features = {
      'img': tf.io.FixedLenFeature([], tf.string),
      'image_id': tf.io.FixedLenFeature([], tf.int64),
      'grapheme_root': tf.io.FixedLenFeature([], tf.int64),
      'vowel_diacritic': tf.io.FixedLenFeature([], tf.int64),
      'consonant_diacritic': tf.io.FixedLenFeature([], tf.int64),
      'unique_tuple': tf.io.FixedLenFeature([], tf.int64),
  }
  example = tf.io.parse_single_example(example, features)
  img = tf.image.decode_image(example['img'])
  img = tf.reshape(img, input_size + (1, ))
  img = tf.cast(img, tf.float32)
  # grayscale -> RGB
  img = tf.repeat(img, 3, -1)

  # image_id = tf.cast(example['image_id'], tf.int32)
  # grapheme_root = tf.cast(example['grapheme_root'], tf.int32)
  # vowel_diacritic = tf.cast(example['vowel_diacritic'], tf.int32)
  # consonant_diacritic = tf.cast(example['consonant_diacritic'], tf.int32)
  unique_tuple = tf.cast(example['unique_tuple'], tf.int32)
  return img, unique_tuple


# In[ ]:


# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery
	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)
		# return the learning rate
		return float(alpha)


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_size', type=str, default='160,256')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--backbone', type=str, default='efficientnet-b4')
    parser.add_argument('--weights', type=str, default='imagenet')
    args, _ = parser.parse_known_args()

    args.input_size = tuple(int(x) for x in args.input_size.split(','))
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

  # build the model
    strategy = get_strategy()
    with strategy.scope():
        model = get_model(input_size=args.input_size + (3, ), backbone=args.backbone,
        weights=args.weights)
        schedule = StepDecay(initAlpha=1e-4, factor=0.25, dropEvery=15)
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=[categorical_accuracy, top_k_categorical_accuracy])

    callbacks = [LearningRateScheduler(schedule)]
    AUTO = tf.data.experimental.AUTOTUNE #
    ignore_order = tf.data.Options() #
    ignore_order.experimental_deterministic = False #
    
    ds_path = KaggleDatasets().get_gcs_path('bengali-tfrecords-v010')
    train_fns = tf.io.gfile.glob(os.path.join(ds_path, 'records/train*.tfrec'))
    train_ds = tf.data.TFRecordDataset(train_fns, num_parallel_reads=AUTO)
#     train_ds = train_ds.with_optional(ignore_order)
    train_ds = train_ds.map(lambda e: read_tfrecords(e, args.input_size), num_parallel_calls=AUTO)
    train_ds = train_ds.repeat().batch(args.batch_size)
    train_ds = train_ds.map(one_hot)
    train_ds = train_ds.map(lambda a, b: mixup(a, b, args.batch_size), num_parallel_calls=AUTO)
    
    val_fns = tf.io.gfile.glob(os.path.join(ds_path, 'records/val*.tfrec'))
    val_ds = tf.data.TFRecordDataset(val_fns, num_parallel_reads=AUTO)
#     val_ds = val_ds.with_optional(ignore_order)
    val_ds = val_ds.map(lambda e: read_tfrecords(e, args.input_size), num_parallel_calls=AUTO)
    val_ds = val_ds.batch(args.batch_size)
    val_ds = val_ds.map(one_hot)

  # train
    num_train_samples = sum(int(fn.split('_')[2]) for fn in train_fns)
     # num_val_samples = sum(int(fn.split('_')[2]) for fn in val_fns)
    steps_per_epoch = num_train_samples // args.batch_size
    print(f'Training on {num_train_samples} samples. Each epochs requires {steps_per_epoch} steps')
    h = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=args.epochs, verbose=1,
          validation_data=val_ds,callbacks=callbacks)
    print(h)
    weight_fn = 'model-%04d.h5' % args.model_id
    model.save_weights(weight_fn)
    model.save('Enet-final.h5')
    print(f'Saved weights to: {weight_fn}')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'main()')


# In[ ]:




