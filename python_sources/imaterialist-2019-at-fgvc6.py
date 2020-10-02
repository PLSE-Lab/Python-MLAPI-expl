#!/usr/bin/env python
# coding: utf-8

# ## Data Overview
# 
# Check the competition page: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/

# ## Dependencies

# In[ ]:


# Downgrade tf to prevent errors in mrcnn
get_ipython().system('pip install tensorflow==1.14')


# In[ ]:


# Downgrade keras to prevent errors in mrcnn
get_ipython().system('pip install keras==2.2.4')


# In[ ]:


import numpy as np
import pandas as pd

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import gc
import json
import glob
from pathlib import Path
import keras # to prevent error when importing mrcnn.model
import tensorflow as tf

import itertools
from tqdm import tqdm

from sklearn.model_selection import train_test_split

tf.__version__
keras.__version__


# In[ ]:


# Root and data directory of the project
ROOT_DIR = Path('/kaggle/working')

# Download Mask RCNN library
get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')
os.chdir('Mask_RCNN')

get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')


# In[ ]:


# Import Mask RCNN
sys.path.append(ROOT_DIR/'Mask_RCNN')  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
import mrcnn.model as modellib


# In[ ]:


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[ ]:


# Define config class
class MyConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = 46 + 1 # background + 46 classes
    
    EPOCHS = 1
    TRAIN_SIZE = 1/1000
    VAL_SIZE = 1/10000
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    STEPS_PER_EPOCH = 5
    VALIDATION_STEPS = 1
    
    LEARNING_RATE = 0.002

# Create instance for late use in the model
config = MyConfig()
config.display()


# ## Dataset

# In[ ]:


# Load label json
train_dir = '/kaggle/input/imaterialist-fashion-2019-FGVC6/'
# Get class names
with open(train_dir+'label_descriptions.json') as f:
    label_descriptions = json.load(f)
label_names = [x['name'] for x in label_descriptions['categories']]


# In[ ]:


# Load train table
df = pd.read_csv(train_dir+'train.csv')
# Remove attributs from catagory
df['labels'] = df['ClassId'].apply(lambda x: x.split('_')[0])


# In[ ]:


# Group by image id and concatenate EncodedPixels and labels
g1_df = df.groupby('ImageId')['EncodedPixels','labels'].agg(lambda x: list(x))
g2_df = df.groupby('ImageId')['Height', 'Width'].mean()

train_df = g1_df.join(g2_df, on='ImageId')


# In[ ]:


# Train and validation split
df_train,df_val = train_test_split(train_df,train_size=config.TRAIN_SIZE ,test_size=config.VAL_SIZE)


# In[ ]:


# Extend the Dataset class and add load_data() to load the training data. 
# Override the following methods:
#  load_image()
#  load_mask()
#  image_reference()

class MyDataset(utils.Dataset):

    def load_data(self, label_names, df):
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            
            self.add_image("fashion", 
                           image_id=i, 
                           path=train_dir+'train/'+i, 
                           labels=row['labels'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], 
                           width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):       
        img = cv2.imread(self.image_info[image_id]['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interpolation=cv2.INTER_AREA)  
        return img

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)


# In[ ]:


# Create dataset for later use in the model
train_dataset = MyDataset()
train_dataset.load_data(label_names,df_train)
train_dataset.prepare()

valid_dataset = MyDataset()
valid_dataset.load_data(label_names,df_val)
valid_dataset.prepare()

# Visualize samples
for i in range(2):
    image_id = random.choice(train_dataset.image_ids)
    print(train_dataset.image_reference(image_id))
    
    image = train_dataset.load_image(image_id)
    mask, class_ids = train_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=4)


# ## Build Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=ROOT_DIR)

# Load weights trained on MS COCO, but skip layers that are different due to the different number of classes
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


# In[ ]:


# Train the heads
model.train(train_dataset, valid_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=config.EPOCHS, 
            layers='heads')
history = model.keras_model.history.history


# In[ ]:


# Fine tune all layers
model.train(train_dataset, valid_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=config.EPOCHS, 
            layers='all')
new_history = model.keras_model.history.history
for k in new_history: 
    history[k] = history[k] + new_history[k]


# ## Detect Objects

# In[ ]:


# Create a configuration for inference
class InferenceConfig(MyConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


# Test on a random image
image_id = random.choice(valid_dataset.image_ids)

# Get original image data
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(valid_dataset, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# Visualize original image
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            train_dataset.class_names, figsize=(8, 8))


# In[ ]:


# Predict and visualize on the same image
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            valid_dataset.class_names, r['scores'],figsize=(8, 8))


# ## Evaluation

# In[ ]:



# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(valid_dataset.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(valid_dataset, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))


# In[ ]:




