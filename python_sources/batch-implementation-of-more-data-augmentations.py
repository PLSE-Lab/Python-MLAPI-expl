#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# * Implement CutMix, MixUp and GridMask in batch.
# * Comparing timings - Batch operations are about 5x - 10x faster.
# * The batch implemention for GridMask is kind partial. For a batch, grid width are fixed in that batch, but rotation angle can be random.
# * References: 
# 
#     * [CutMix and MixUp on GPU/TPU (by Chris Deotte)](https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu)
#     * [GridMask data augmentation with tensorflow (by Xie29)](https://www.kaggle.com/xiejialun/gridmask-data-augmentation-with-tensorflow)
# 

# # Common

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import re
import math
import datetime
from kaggle_datasets import KaggleDatasets
import tensorflow.keras.backend as K
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


image_size = 192
IMAGE_SIZE = [image_size, image_size]
BATCH_SIZE = 64
AUG_BATCH = BATCH_SIZE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')


GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192'

}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

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

def load_dataset(filenames, labeled = True, ordered = False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # Diregarding data order. Order does not matter since we will be shuffling the data anyway
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))

print('Dataset: {} training images'.format(NUM_TRAINING_IMAGES))


# In[ ]:


def batch_to_numpy_images_and_labels(data):
    
    images = data
    numpy_images = images.numpy()

    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images


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
    """
    
    # data
    images = batch_to_numpy_images_and_labels(databatch)
    labels = None
    
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
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


# In[ ]:


def get_training_dataset(dataset, batch_size=None, advanced_aug=True, repeat=True, with_labels=True, drop_remainder=False):
    
    if not with_labels:
        dataset = dataset.map(lambda image, label: image, num_parallel_calls=AUTO)
    
    if advanced_aug:
        dataset = dataset.map(transform, num_parallel_calls=AUTO)
    
    if type(repeat) == bool and repeat:
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    elif type(repeat) == int and repeat > 0:
        dataset = dataset.repeat(repeat)
    
    dataset = dataset.shuffle(2048)
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset


# In[ ]:


dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
training_dataset = get_training_dataset(dataset, advanced_aug=False, repeat=1, with_labels=True)

images, labels = next(iter(training_dataset.take(1)))
print(images.shape)

display_batch_of_images(images)


# # Original CutMix

# In[ ]:


def cutmix(images, label, PROBABILITY = 1.0):
    
    # input images - is a batch of imagess of size [n,dim,dim,3] not a single images of [dim,dim,3]
    # output - a batch of imagess with cutmix applied
    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)
        # CHOOSE RANDOM images TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # MAKE CUTMIX images
        one = images[j,ya:yb,0:xa,:]
        two = images[k,ya:yb,xa:xb,:]
        three = images[j,ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([images[j,0:ya,:,:],middle,images[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    images2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return images2,label2


# In[ ]:


new_images, new_labels = cutmix(images, labels, PROBABILITY=1.0)
display_batch_of_images(new_images)


# # Batch CutMix

# In[ ]:


def batch_cutmix(images, labels, PROBABILITY=1.0, batch_size=0):
    
    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    if batch_size == 0:
        batch_size = BATCH_SIZE
    
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    # This is a tensor containing 0 or 1 -- 0: no cutmix.
    # shape = [batch_size]
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    
    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # Choose random location in the original image to put the new images
    # shape = [batch_size]
    new_x = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    new_y = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    
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
        labels = tf.one_hot(labels, CLASSES)
        
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)        
        
    return new_images, new_labels


# In[ ]:


new_images, new_labels = batch_cutmix(images, labels, PROBABILITY=1.0)
display_batch_of_images(new_images)


# # Compare timing for CutMix

# In[ ]:


n_iter = 1000

start = datetime.datetime.now()
for i in range(n_iter):
    batch_cutmix(images, labels, PROBABILITY=1.0) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"batch_cutmix: {timing}")


# In[ ]:


start = datetime.datetime.now()
for i in range(n_iter):
    cutmix(images, labels, PROBABILITY=1.0) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"cutmix: {timing}")


# # Original MixUP

# In[ ]:


def mixup(image, label, PROBABILITY = 1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image2,label2


# In[ ]:


new_images, new_labels = mixup(images, labels, PROBABILITY=1.0)
display_batch_of_images(new_images)


# # Batch MixUP

# In[ ]:


def batch_mixup(images, labels, PROBABILITY=1.0, batch_size=0):

    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    if batch_size == 0:
        batch_size = BATCH_SIZE
    
    # Do `batch_mixup` with a probability = `PROBABILITY`
    # This is a tensor containing 0 or 1 -- 0: no mixup.
    # shape = [batch_size]
    do_mixup = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

    # Choose random images in the batch for cutmix
    # shape = [batch_size]
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    
    # ratio of importance of the 2 images to be mixed up
    # shape = [batch_size]
    a = tf.random.uniform([batch_size], 0, 1) * tf.cast(do_mixup, tf.float32)  # this is beta dist with alpha=1.0
                
    # The second part corresponds to the images to be added to the original images `images`.
    new_images =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new_image_indices)

    # Make labels
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, CLASSES)
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)

    return new_images, new_labels


# In[ ]:


new_images, new_labels = batch_mixup(images, labels, PROBABILITY=1.0)
display_batch_of_images(new_images)


# # Compare timing for MixUp

# In[ ]:


start = datetime.datetime.now()
for i in range(n_iter):
    batch_mixup(images, labels, PROBABILITY=1.0) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"batch_mixup: {timing}")


# In[ ]:


start = datetime.datetime.now()
for i in range(n_iter):
    mixup(images, labels, PROBABILITY=1.0) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"mixup: {timing}")


# # Original GridMask

# In[ ]:


def transform(image, inv_mat, image_shape):

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
    angle = float(angle) * tf.constant(1, shape=[1], dtype=tf.float32) # tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform(image, rot_mat_inv, image_shape)


def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    h, w = image_height, image_width
    hh = int(np.ceil(np.sqrt(h*h+w*w)))
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    
    st_h = 0
    st_w = 0

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

def apply_grid_mask(image, image_shape):
    mask = GridMask(image_shape[0],
                    image_shape[1],
                    AugParams['d1'],
                    AugParams['d2'],
                    AugParams['rotate'],
                    AugParams['ratio'])
    
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask, tf.float32)


AugParams = {
    'd1' : 10,
    'd2': 40 ,
    'rotate' : 45,
    'ratio' : 0.5
}


# In[ ]:


new_image = apply_grid_mask(images[0], images[0].shape)
new_images = tf.convert_to_tensor([new_image])
display_batch_of_images(new_images)


# # (Partial) Batch GridMask

# In[ ]:


def get_batch_rotation_matrix(angles, batch_size=0):
    """Returns a tf.Tensor of shape (batch_size, 3, 3) with each element along the 1st axis being
       an image rotation matrix (which transforms indicies).

    Args:
        angles: 1-D Tensor with shape [batch_size].
        
    Returns:
        A 3-D Tensor with shape [batch_size, 3, 3].
    """    

    if batch_size == 0:
        batch_size = BATCH_SIZE
    
    # CONVERT DEGREES TO RADIANS
    angles = tf.constant(math.pi) * angles / 180.0

    # shape = (batch_size,)
    one = tf.ones_like(angles, dtype=tf.float32)
    zero = tf.zeros_like(angles, dtype=tf.float32)
    
    # ROTATION MATRIX
    c1 = tf.math.cos(angles) # shape = (batch_size,)
    s1 = tf.math.sin(angles) # shape = (batch_size,)

    # Intermediate matrix for rotation, shape = (9, batch_size) 
    rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    rotation_matrix_temp = tf.transpose(rotation_matrix_temp)
    # Fianl rotation matrix, shape = (batch_size, 3, 3)
    rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))
        
    return rotation_matrix


def batch_random_rotate(images, max_angles, batch_size=0):
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
    
    if batch_size == 0:
        batch_size = BATCH_SIZE
    
    angles = max_angles * tf.random.normal([batch_size], dtype='float32')

  
    # GET TRANSFORMATION MATRIX
    # shape = (batch_size, 3, 3)
    m = get_batch_rotation_matrix(angles, batch_size) 

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
    d = tf.gather_nd(images, tf.transpose(idx3, perm=[0, 2, 1]), batch_dims=1)
        
    # shape = (batch_size, DIM, DIM, 3)
    new_images = tf.reshape(d, (batch_size, DIM, DIM, 3))

    return new_images


def batch_get_grid_mask(d1, d2, ratio=0.5, max_angle=90, batch_size=0):
        
    # ratio: the ratio of black region

    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    if batch_size == 0:
        batch_size = BATCH_SIZE

    # Length of diagonal
    hh = tf.cast((tf.math.ceil(tf.math.sqrt(2.0) * DIM)), tf.int64)
    hh = hh + tf.math.floormod(hh, 2)
    
    # We look squares of size dxd inside each image
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int64)
    
    # Inside each square of size dxd, we mask a square of size LxL (L <= d)
    l = tf.cast(tf.cast(d, tf.float32) * ratio + 0.5, tf.int64)

    lower_limit = -1
    upper_limit = tf.math.floordiv(hh, d) + 1
    indices = tf.range(lower_limit, upper_limit)  # shape = [upper_limit + 1]
    
    # The 1st component has shape [upper_limit + 1, 1]
    # The 2nd component has shae [1: L]
    # The addition has shape [upper_limit + 1: L]
    # The final output has sahpe [upper_limit + 1 * L]
    ranges = tf.reshape((d * indices)[:, tf.newaxis] + tf.range(l, dtype=tf.int64)[tf.newaxis, :], shape=[-1])
    shift = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int64)
    
    ranges = shift + ranges

    clip_mask = tf.logical_or(ranges < 0 , ranges > hh - 1)
    ranges = tf.boolean_mask(ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(ranges)), tf.int64)])
    
    ranges = tf.repeat(ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)
    
    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)
    
    mask = batch_random_rotate(tf.broadcast_to(mask[tf.newaxis, :, :, :], shape=[batch_size, mask.shape[0], mask.shape[1], 3]), max_angle, batch_size)
    
    mask = tf.image.crop_to_bounding_box(mask, (hh - DIM) // 2, (hh - DIM) // 2, tf.cast(DIM, dtype=tf.int64), tf.cast(DIM, dtype=tf.int64))

    return mask


def batch_grid_mask(images, batch_size=0):
    
    if batch_size == 0:
        batch_size = BATCH_SIZE
    
    # d1, d2 determined the width of the grid
    d1 = 35
    d2 = d1 + 1 + tf.cast(35 * tf.random.uniform(shape=[]), dtype=tf.int64)
    ratio = 0.25 + 0.25 * tf.random.uniform(shape=[])
    max_angle = 90
        
    mask = batch_get_grid_mask(d1, d2, ratio, max_angle, batch_size)
    
    return images * tf.cast(mask, tf.float32)


# In[ ]:



new_images = batch_grid_mask(images)
display_batch_of_images(new_images)


# # Compare timing for GridMask

# In[ ]:


n_iter = 100

start = datetime.datetime.now()
for i in range(n_iter):
    batch_grid_mask(images) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"batch_grid_mask: {timing}")


# In[ ]:


start = datetime.datetime.now()
for i in range(n_iter):
    for image in images:
        apply_grid_mask(image, image.shape) 
end = datetime.datetime.now()
timing = (end - start).total_seconds() / n_iter
print(f"grid_mask: {timing}")

