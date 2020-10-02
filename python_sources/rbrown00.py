#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import random
random.seed(42)

np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)

# Any results you write to the current directory are saved as output.


# In[2]:


ROOT_DIR = "/kaggle"


# In[3]:


submit = pd.read_csv(ROOT_DIR + "/input/test_sample_submission.csv")
masks = pd.read_csv(ROOT_DIR + "/input/train_masks.csv")


# In[4]:


from zipfile import ZipFile
from PIL import Image
import PIL


# In[5]:


test_dir = ROOT_DIR + "/input/test_images/test_images"
train_dir = ROOT_DIR + "../input/train_images/train_images"


# In[6]:


test_files = [test_dir + "/" + s + "/images/" + s + ".png" for s in submit['ImageId'].unique()]
test_names = [s for s in submit['ImageId'].unique()]
train_files = [train_dir + "/" + s + "/images/" + s + ".png" for s in masks['ImageId'].unique()]
train_names = [s for s in masks['ImageId'].unique()]


# In[7]:


get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')


# In[9]:


print(os.listdir("./"))
os.chdir('Mask_RCNN')


# In[12]:


from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[15]:


class Nucleus(utils.Dataset):
    
    def load_nucleus(self):
        #Add Classes (in this example there's only the "nucleus class)
        self.add_class("train", 1, "train")
        for image_id in train_names:
            self.add_image(
                "train",
                image_id=image_id,
                path= self.image_reference(image_id))

    def load_mask(self, image_id):
        image = Image.open(self.image_reference(image_id)).convert('RGB')
        
        vals = masks[masks['ImageId'] == name]['EncodedPixels']
        mask_vals = ""
        for string in vals:
            mask_vals = mask_vals + " " + string
        mask_vals = mask_vals.split(" ")[1:]
        mask_vals = [int(s) for s in mask_vals]

        mask_array = 255 * np.zeros(img_array.shape, dtype=np.uint8)

        dims = img_array.shape
        #print(dims)
        for i in range(0, len(mask_vals), 2):
            for j in range(0, mask_vals[i + 1]):
                row = int(mask_vals[i] % dims[0])
                col = int(mask_vals[i] / dims[0])
                mask_array[row][col] = [255, 0, 0]

        mask_img = Image.fromarray(mask_array)
        mask_array = np.array(mask_img)
                
    def image_reference(self, image_id):
        return test_dir + "/" + image_id + "/images/" + image_id + ".png"


# In[16]:


nuc = Nucleus()
nuc.load_nucleus()


# In[17]:


def train(model, dataset_dir, subset):
#Training Dataset
    dataset_train = NucleusDataset()
    dataset_train.load_dataset()  
    dataset_train.prepare()
#Validating Dataset
    dataset_val = CustomDataset()        
    dataset_val.load_dataset()    
    dataset_val.prepare()


# In[22]:


class NucleusConfig(Config):
  
    # Give the configuration a recognizable name    
    NAME = "nucleus"
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657)
    VALIDATION_STEPS = 1
    # Don't exclude based on confidence. Since we have two classes
      # then 0.5 is the minimum anyway as it picks between nucleus and    BG
    DETECTION_MIN_CONFIDENCE = 0
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400
class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1    
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate morepropsals.
    RPN_NMS_THRESHOLD = 0.7

