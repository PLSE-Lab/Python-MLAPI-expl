#!/usr/bin/env python
# coding: utf-8

# # Ship Detection
# This notebook uses the model trained by [Mask R-CNN Ship Ddetection MVM - 1](https://www.kaggle.com/samlin001/mask-r-cnn-ship-detection-minimum-viable-model-1) to detect ships in satellite images from [Airbus Ship Detection Challenge test_v2.zip](https://www.kaggle.com/c/airbus-ship-detection/data).
# 
# 

# # Project Configuration

# In[ ]:


import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
import sys
import time
import errno
from skimage.data import imread
from skimage.morphology import label

# Configurations
PRE_TRAINED_WEIGHT_URL = 'https://github.com/samlin001/Mask_RCNN/releases/download/v2.2-alpha/mask_rcnn_asdc.h5'
WORKING_DIR = '/kaggle/working'
INPUT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/output'
IMAGE_DIR = os.path.join(INPUT_DIR, 'test_v2')
MASK_RCNN_PATH = os.path.join(WORKING_DIR, 'Mask_RCNN-master')
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
CSV_HEADER = 'ImageId,EncodedPixels\n'
SUBMISSION_FILE_NAME = os.path.join(WORKING_DIR, 'submission_v2.csv')

print('Working Dir:', WORKING_DIR, os.listdir(WORKING_DIR))
print('Input Dir:', INPUT_DIR, os.listdir(INPUT_DIR))


# # Import Mask R-CNN code
# Download and import [Mask R-CNN code](https://github.com/samlin001/Mask_RCNN).

# In[ ]:


# if to clone Mask_R-CNN git when it exists 
UPDATE_MASK_RCNN = False

os.chdir(WORKING_DIR)
if UPDATE_MASK_RCNN:
    get_ipython().system('rm -rf {MASK_RCNN_PATH}')

# Downlaod Mask RCNN code to a local folder 
if not os.path.exists(MASK_RCNN_PATH):
    get_ipython().system(' wget https://github.com/samlin001/Mask_RCNN/archive/master.zip -O Mask_RCNN-master.zip')
    get_ipython().system(" unzip Mask_RCNN-master.zip 'Mask_RCNN-master/mrcnn/*'")
    get_ipython().system(' rm Mask_RCNN-master.zip')

# Import Mask RCNN
sys.path.append(MASK_RCNN_PATH)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log    


# ## Model configurations for inference
# 

# In[ ]:


class AirbusShipDetectionChallengeGPUConfig(Config):
    """
    Configuration of Airbus Ship Detection Challenge Dataset 
    Overrides values in the base Config class.
    From https://github.com/samlin001/Mask_RCNN/blob/master/mrcnn/config.py
    """
    # https://www.kaggle.com/docs/kernels#technical-specifications
    NAME = 'ASDC_GPU'
    # NUMBER OF GPUs to use.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 2  # ship or background
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH
    STEPS_PER_EPOCH = 5
    VALIDATION_STEPS = 5
    SAVE_BEST_ONLY = True
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.95

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.05

    
class InferenceConfig(AirbusShipDetectionChallengeGPUConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    # One image for each time at inference
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# # Load the weights
# * The weights are the output files on the training of [Mask R-CNN Ship Ddetection MVM-1a](https://www.kaggle.com/samlin001/mask-r-cnn-ship-detection-minimum-viable-model-1a).
# * Importing the weights via "Add Data" will copy a few GB data over. It is more efficiency to download [mask_rcnn_asdc.h5](https://github.com/samlin001/Mask_RCNN/releases) from github.
# 

# In[ ]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=WORKING_DIR, config=config)
start_time = time.time()
pre_weights = PRE_TRAINED_WEIGHT_URL
weights_path = os.path.join(WORKING_DIR, 'mask_rcnn_asdc.h5')
if not os.path.exists(weights_path):
    get_ipython().system(' wget {pre_weights} -O {weights_path}')

print("Loading weights: ", weights_path)
model.load_weights(weights_path, by_name=True)
end_time = time.time() - start_time
print("loading weights: {}".format(end_time))


# ## Mask Run Length Encoding & transformation functions

# In[ ]:


# ref: https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def create_submission_records(image_id, masks, scores):
    """Creates submission records."""
    no_masks = len(scores)
    # Return 'ImageId,' when there is no ship detected
    if no_masks < 1:
        return '{},'.format(image_id)
    
    # Creates a record for each mask
    records = []
    for i in range(no_masks):
        rle = rle_encode(masks[:,:,i])
        records.append('{},{}'.format(image_id, rle))
    return '\n'.join(records)

def save_csv(filename, header, records):
    # Save to csv file
    content = header + '\n'.join(records)
    with open(filename, 'w') as f:
        f.write(content)

# Load file name list
file_names = next(os.walk(IMAGE_DIR))[2]
print(len(file_names), ' test images found')


# # Inference
# Detect ships for each image in test_v2.zip and encord their masks into the format for submission.

# In[ ]:


# set MAX_INFERENCE_IMAGE_NO < len(file_names) for development
MAX_INFERENCE_IMAGE_NO = 100000
# MAX_INFERENCE_IMAGE_NO = 10
inference_start = time.time()
i = 0
mask_records = []
for image_id in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_id))
    # Detect ships
    result = model.detect([image], verbose=0)[0]
    records = create_submission_records(image_id, result['masks'], result['scores'])
    mask_records.append(records)
    i += 1
    if i >= MAX_INFERENCE_IMAGE_NO:
        break
inference_end = time.time()
inference_time = inference_end - inference_start
print('Inference Time: {:0.2f} minutes for {} images'.format(inference_time/60, i))
save_csv(SUBMISSION_FILE_NAME,CSV_HEADER, mask_records)
print('Detect {:0.2f} images per second'.format(inference_time/i))
save_csv(SUBMISSION_FILE_NAME,CSV_HEADER, mask_records)

print('Save submission to {}'.format(SUBMISSION_FILE_NAME))


# # Result inspections
# Display a few ship deticion examples from the results.

# In[ ]:


# Read mask encording from the input CSV file 
masks = pd.read_csv(SUBMISSION_FILE_NAME)
masks.head()

def rle_decode(mask_rle, shape=SHAPE):
    '''
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return 
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    # gets starts & lengths 1d arrays 
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, shape=SHAPE):
    '''Take the individual ship masks and create a single mask array for all ships
    in_mask_list: pd Series: [idx0] [RLE string0]...
    Returns numpy array as (shape.h, sahpe.w, 1)
    '''
    all_masks = np.zeros(shape, dtype = np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def show_image_mask(image_id, path):
    '''Show image & ship mask
    '''
    fig, axarr = plt.subplots(1, 3, figsize = (20, 5))
    # image
    img_0 = imread(os.path.join(path, image_id))
    axarr[0].imshow(img_0)
    axarr[0].set_title(image_id)
    
    # input mask
    rle_1 = masks.query('ImageId=="{}"'.format(image_id))['EncodedPixels']
    img_1 = masks_as_image(rle_1)
    # takes 2d array (shape.h, sahpe.w)
    axarr[1].imshow(img_1[:, :, 0])
    axarr[1].set_title('Ship Mask')
    
    axarr[2].imshow(img_0)
    axarr[2].imshow(img_1[:, :, 0], alpha=0.3)
    axarr[2].set_title('Encoded & Decoded Mask')
    plt.show()

# inspect a few examples
show_image_mask('c175e03b9.jpg', IMAGE_DIR)    
show_image_mask('8a56c9bdd.jpg', IMAGE_DIR)
show_image_mask('f52f4a484.jpg', IMAGE_DIR)
for i in range(20):
    show_image_mask(random.choice(file_names), IMAGE_DIR)
    


# # Clean up

# In[ ]:


get_ipython().system('rm -rf {MASK_RCNN_PATH}')

