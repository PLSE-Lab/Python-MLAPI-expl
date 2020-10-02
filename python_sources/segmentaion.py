#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
import sys
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import skimage.io
import matplotlib.pyplot as plt


# In[ ]:


# https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07

get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')
os.chdir('Mask_RCNN')

get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')


# In[ ]:


DATA_DIR = Path('/kaggle/input')
ROOT_DIR = Path('/kaggle/working')


# In[ ]:


sys.path.append(ROOT_DIR/'Mask_RCNN')


# In[ ]:


get_ipython().system('pip install pycocotools')


# In[ ]:


from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[ ]:


get_ipython().system('wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5')


# In[ ]:


COCO_MODEL_PATH = 'mask_rcnn_coco.h5'


# In[ ]:


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  # To find local version
import coco


# In[ ]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
config = InferenceConfig()
config.display()


# In[ ]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# In[ ]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# In[ ]:


os.chdir('/kaggle/')


# In[ ]:


IMAGE_DIR = "/kaggle/input/test/"


# In[ ]:


os.listdir("./input")


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-segmentable.csv')


# In[ ]:


class_lookup_df = pd.read_csv("./challenge-2019-classes-description-segmentable.csv")
emp


# In[ ]:


os.listdir()


# In[ ]:


class_lookup_df=pd.read_csv("./challenge-2019-classes-description-segmentable.csv")


# In[ ]:


# we have to convert coco classes to this competition's one.

class_lookup_df.columns = ["encoded_label","label"]
class_lookup_df['label'] = class_lookup_df['label'].str.lower()
class_lookup_df.head()


# In[ ]:


empty_submission_df.head()


# In[ ]:


sample_image = "80155d58d0ee19bd.jpg"
image = skimage.io.imread(os.path.join(IMAGE_DIR, sample_image))
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print( class_names[r['class_ids'][0]])

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


# In[ ]:


r['masks'].shape


# In[ ]:





# In[ ]:




