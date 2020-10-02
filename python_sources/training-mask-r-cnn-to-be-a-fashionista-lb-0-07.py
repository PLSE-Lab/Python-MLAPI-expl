#!/usr/bin/env python
# coding: utf-8

# Welcome to the world where fashion meets computer vision! This is a starter kernel that applies Mask R-CNN with COCO pretrained weights to the task of [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6).

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==1.13.1')


# In[ ]:


import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow


# In[ ]:


get_ipython().system('pip3 show tensorflow-gpu')


# In[ ]:


import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))


# In[ ]:


with tf.Session() as sess:
  devices = sess.list_devices()
devices


# In[ ]:


DATA_DIR = Path(r'/kaggle/working/Mask_RCNN/fashion_dataset')
ROOT_DIR = Path(r'/kaggle/working/Mask_RCNN')

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 46
IMAGE_SIZE = 512


# # Dowload Libraries and Pretrained Weights

# In[ ]:


get_ipython().system('git clone https://github.com/IITGuwahati-AI/Mask_RCNN.git')


get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')


# In[ ]:


pwd


# In[ ]:


sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils_for_FGC
import mrcnn.model_for_FGC as modellib
from mrcnn import visualize
from mrcnn.model_for_FGC import log


# In[ ]:


# !wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
# !ls -lh mask_rcnn_coco.h5


COCO_WEIGHTS_PATH = Path(r'/kaggle/working/Mask_RCNN/mask_rcnn_coco.h5')
NUM_CATS = 46
IMAGE_SIZE = 1024


# # Set Config

# Mask R-CNN has a load of hyperparameters. I only adjust some of them.

# In[ ]:


class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 # a memory error occurs when IMAGES_PER_GPU is too high
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

    ## My changes CA
    BACKBONE = 'resnet101'
    
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024    
    IMAGE_RESIZE_MODE = 'square'

    MINI_MASK_SHAPE = (112, 112)  # (height, width) of the mini-mask

    NUM_ATTR = 294

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_attr_loss":1.
    }


    
config = FashionConfig()
config.display()


# # Make Datasets

# In[ ]:


with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

class_names = [x['name'] for x in label_descriptions['categories']]
attr_names = [x['name'] for x in label_descriptions['attributes']]


# In[ ]:


print(len(class_names),len(attr_names))


# In[ ]:



segment_df = pd.read_csv(DATA_DIR/"train_small.csv")
segment_df['AttributesIds'] = segment_df['AttributesIds'].apply(lambda x:tuple([int(i) for i in x.split(',')]))


# Segments that contain attributes are only 3.46% of data, and [according to the host](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/90643#523135), 80% of images have no attribute. So, in the first step, we can only deal with categories to reduce the complexity of the task.

# In[ ]:


def pad_tuple_attrs(x):
    if x!=x:
        x = []
    else:
        x = list(x)
    for i in range(len(x)):
        if x[i]>=281 and x[i]<284:
            x[i] = x[i]-46
        elif x[i]>284:
            x[i] = x[i]-47
    
    x = tuple(x)
    return x


# In[ ]:



segment_df['AttributesIds'] = segment_df['AttributesIds'].apply(pad_tuple_attrs)


# In[ ]:


def get_one_hot(targets):
    targets = np.array(list(targets), dtype = np.int32)
    nb_classes = 294
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes]).sum(axis=0)


# Rows with the same image are grouped together because the subsequent operations perform in an image level.

# In[ ]:


segment_df['AttributesIds'] = segment_df['AttributesIds'].apply(get_one_hot)


# In[ ]:


segment_df['AttributesIds'].head()


# In[ ]:


image_df = segment_df.groupby('ImageId')['EncodedPixels', 'ClassId', 'AttributesIds'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

print("Total images: ", len(image_df))
image_df.head()


# Here is the custom function that resizes an image.

# In[ ]:


def resize_image(image_path):
    image_path = image_path + ".jpg"
    # print("image_path", image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img


# The crucial part is to create a dataset for this task.

# In[ ]:


class FashionDataset(utils_for_FGC.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(class_names):
            self.add_class("fashion", i+1, name)
        
        for i, name in enumerate(attr_names):
            self.add_attribute("fashion", i, name)
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(DATA_DIR/'train'/row.name), 
                           labels=row['ClassId'],
                           attributes=row['AttributesIds'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    

    def image_reference(self, image_id):
        # attr_sublist=[]
        attr_list=[]
        info = self.image_info[image_id]
        for x in info['attributes']:
            attr_sublist=[]
            for i, j in enumerate(x):
                if(j==1):
                    attr_sublist.append(attr_names[i])
            attr_list.append(attr_sublist)
                
            
            
        return info['path'], [class_names[int(x)] for x in info['labels']],attr_list
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        attributes = []
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            attributes.append(list(info['attributes'][m]))
            
        return mask, np.array(labels), np.array([np.array(attr) for attr in attributes])


# Let's visualize some random images and their masks.

# In[ ]:



dataset = FashionDataset(image_df)
dataset.prepare()

for i in range(1):
    image_id = random.choice(dataset.image_ids)
    print(dataset.image_reference(image_id))
    
    image = dataset.load_image(image_id)
    mask, class_ids, attr_ids = dataset.load_mask(image_id)
    # print("class_ids", class_ids)
    # print("attr_ids", attr_ids)
    # print(type(attr_ids))
    visualize.display_top_masks(image, mask, class_ids, attr_ids, dataset.class_names, dataset.attr_names, limit=4)


# Now, the data are partitioned into train and validation sets.

# In[ ]:


# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 3

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()


# Let's visualize class distributions of the train and validation data.

# In[ ]:


train_segments = np.concatenate(train_df['ClassId'].values).astype(int)
print("Total train images: ", len(train_df))
print("Total train segments: ", len(train_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, class_names, rotation='vertical')
plt.show()

valid_segments = np.concatenate(valid_df['ClassId'].values).astype(int)
print("Total train images: ", len(valid_df))
print("Total validation segments: ", len(valid_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(valid_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, class_names, rotation='vertical')
plt.show()


train2_segments = np.concatenate(train_df['AttributesIds'].values).astype(int).reshape((-1,))
train2_segments = train2_segments[train2_segments!= -1]
# print("Total train images: ", len(valid_df))
print("Total train segments: ", len(train2_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train2_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, attr_names, rotation='vertical')
plt.show()

train2_segments = np.concatenate(valid_df['AttributesIds'].values).astype(int).reshape((-1,))
train2_segments = train2_segments[train2_segments!= -1]
# print("Total train images: ", len(valid_df))
print("Total train segments: ", len(train2_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train2_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, attr_names, rotation='vertical')
plt.show()


# # Train

# In[ ]:


# # Note that any hyperparameters here, such as LR, may still not be optimal
LR = 1e-4


import warnings 
warnings.filterwarnings("ignore")


# This section creates a Mask R-CNN model and specifies augmentations to be used.

# 

# 

# In[ ]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights(str(COCO_WEIGHTS_PATH), by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


# In[ ]:


augmentation = iaa.Sequential([
    iaa.Fliplr(0.5) # only horizontal flip here
])


# First, we train only the heads.

# In[ ]:


get_ipython().run_cell_magic('time', '', "model.train(train_dataset, valid_dataset,\n            learning_rate=config.LEARNING_RATE*2, # train heads with higher lr to speedup learning\n            epochs=2,\n            layers='heads',\n            augmentation=None)\n\nhistory = model.keras_model.history.history")


# In[ ]:





# In[ ]:


get_ipython().system('pip3 show tensorflow')


# Then, all layers are trained.

# In[ ]:


get_ipython().run_cell_magic('time', '', "model.train(train_dataset, valid_dataset,\n            learning_rate=LR,\n            epochs=EPOCHS[1],\n            layers='all',\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]")


# Afterwards, we reduce LR and train again.

# In[ ]:


get_ipython().run_cell_magic('time', '', "model.train(train_dataset, valid_dataset,\n            learning_rate=LR/5,\n            epochs=EPOCHS[2],\n            layers='all',\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]")


# Let's visualize training history and choose the best epoch.

# In[ ]:


epochs = range(EPOCHS[-1])

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()


# In[ ]:


best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])


# # Predict

# The final step is to use our model to predict test data.

# In[ ]:


glob_list = glob.glob(f'/kaggle/working/fashion*/mask_rcnn_fashion_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''


# This cell defines InferenceConfig and loads the best trained model.

# In[ ]:


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Then, load the submission data.

# In[ ]:


sample_df = pd.read_csv(DATA_DIR/"sample_submission.csv")
sample_df.head()


# Here is the main prediction steps, along with some helper functions.

# In[ ]:


# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


# In[ ]:


# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_list = []\nmissing_count = 0\nfor i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):\n    image = resize_image(str(DATA_DIR/'test'/row['ImageId']))\n    result = model.detect([image])[0]\n    if result['masks'].size > 0:\n        masks, _ = refine_masks(result['masks'], result['rois'])\n        for m in range(masks.shape[-1]):\n            mask = masks[:, :, m].ravel(order='F')\n            rle = to_rle(mask)\n            label = result['class_ids'][m] - 1\n            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])\n    else:\n        # The system does not allow missing ids, this is an easy way to fill them \n        sub_list.append([row['ImageId'], '1 1', 23])\n        missing_count += 1")


# The submission file is created, when all predictions are ready.

# In[ ]:


submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv", index=False)


# Finally, it's pleasing to visualize the results! Sample images contain both fashion models and predictions from the Mask R-CNN model.

# In[ ]:


for i in range(9):
    image_id = sample_df.sample()['ImageId'].values[0]
    image_path = str(DATA_DIR/'test'/image_id)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = model.detect([resize_image(image_path)])
    r = result[0]
    
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']
        
    visualize.display_instances(img, rois, masks, r['class_ids'], 
                                ['bg']+label_names, r['scores'],
                                title=image_id, figsize=(12, 12))


# [](http://)My code is largely based on [this Mask-RCNN kernel](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155) and borrowed some ideas from [the U-Net Baseline kernel](https://www.kaggle.com/go1dfish/u-net-baseline-by-pytorch-in-fgvc6-resize). So, I would like to thank the kernel authors for sharing insights and programming techniques. Importantly, an image segmentation task can be accomplished with short code and good accuracy thanks to [Matterport's implementation](https://github.com/matterport/Mask_RCNN) and a deep learning line of researches culminating in [Mask R-CNN](https://arxiv.org/abs/1703.06870).
# 
# I am sorry that I published this kernel quite late, beyond the halfway of a timeline. I just started working for this competition about a week ago, and to my surprise, the score fell in the range of silver medals at that time. I have no dedicated GPU and no time to further tune the model, so I decided to make this kernel public as a starter guide for anyone who is interested to join this delightful competition.
# 
# <img src='https://i.imgur.com/j6LPLQc.png'>

# Hope you guys like this kernel. If there are any bugs, please let me know.
# 
# P.S. When clicking 'Submit to Competition' button, I always run into 404 erros, so I have to save a submission file and upload it to the submission page for submitting. The public LB score of this kernel is around **0.07**.

# In[ ]:




