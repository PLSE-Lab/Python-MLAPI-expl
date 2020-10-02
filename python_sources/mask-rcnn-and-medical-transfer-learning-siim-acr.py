#!/usr/bin/env python
# coding: utf-8

# **Mask-RCNN Starter Model for the SIIM-ACR Pneumothorax Segmentation with transfer learning **
# 
# This kernel uses pre-trained weights from a past medical imaging competition on pneumonia identification: https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155
# 
# Basic ideas included here:
# * dataset distribution balancing
# * image augmentation
# * multi-stage training for transfer learning, and model weights analysis
# * multi-mask prediction for submission
# 
# If you please, upvote and leave questions or constructive feedback below (for me and other kagglers learning).
# Cheers!

# In[ ]:


debug = False
# debug = True

get_ipython().system('ls ../input/')


# In[ ]:


import warnings 
warnings.filterwarnings("ignore")

import os, gc
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import glob

sys.path.insert(0, '/kaggle/input/siim-acr-pneumothorax-segmentation')
from mask_functions import rle2mask, mask2rle


# In[ ]:


DATA_DIR = '/kaggle/input/siim-acr-pneumothorax-segmentation-data/pneumothorax'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'

get_ipython().system('ls {DATA_DIR}')


# ## Install Matterport's Mask-RCNN
# A very popular model in github.
# See the [Matterport's implementation of Mask-RCNN](https://github.com/matterport/Mask_RCNN).

# In[ ]:


# !pip install 'keras==2.1.6' --force-reinstall
STAGE_DIR = '/tmp/Mask_RCNN'
get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git {STAGE_DIR}')
os.chdir(STAGE_DIR)
#!python setup.py -q install
get_ipython().system('rm .git samples images assets -rf')
get_ipython().system('pwd; ls')


# In[ ]:


# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[ ]:


train_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-train')
test_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-test')

# count files
get_ipython().system('ls -m {train_dicom_dir} | wc')
get_ipython().system('ls -m {test_dicom_dir} | wc')


# ### Load Pneumonia pre-trained weights

# In[ ]:


# get model with best validation score: https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155/
WEIGHTS_PATH = "mask_rcnn_pneumonia.h5"
get_ipython().system('cp /kaggle/input/mask-rcnn*/pneumonia*/*0013.h5 {WEIGHTS_PATH}')
get_ipython().system('du -sh *.h5')


# ## Setup Mask-RCNN
# 
# - dicom_fps is a list of the dicom image path and filenames 
# - image_annotions is a dictionary of the annotations keyed by the filenames
# - parsing the dataset returns a list of the image filenames and the annotations dictionary

# In[ ]:


# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal

IMAGE_DIM = 512

class DetectorConfig(Config):    
    # Give the configuration a recognizable name  
    NAME = 'Pneumothorax'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 11
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background and pneumothorax classes
    
    IMAGE_MIN_DIM = IMAGE_DIM
    IMAGE_MAX_DIM = IMAGE_DIM
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 15
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.90
    DETECTION_NMS_THRESHOLD = 0.1
    WEIGHT_DECAY = 0.0005

    STEPS_PER_EPOCH = 20 if debug else 350
    VALIDATION_STEPS = 10 if debug else 120
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 12.0,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 2.4
    }

config = DetectorConfig()
config.display()


# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.util import montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

def multi_rle_encode(img, **kwargs):
    ''' Encode disconnected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle2mask(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle2mask(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

def masks_as_image(rle_list, shape):
    # Take the individual masks and create a single mask array
    all_masks = np.zeros(shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str) and mask != '-1':
            all_masks |= rle2mask(mask, shape[0], shape[1]).T.astype(bool)
    return all_masks

def masks_as_color(rle_list, shape):
    # Take the individual masks and create a color mask array
    all_masks = np.zeros(shape, dtype=np.float)
    scale = lambda x: (len(rle_list)+x+1) / (len(rle_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(rle_list):
        if isinstance(mask, str) and mask != '-1':
            all_masks[:,:] += scale(i) * rle2mask(mask, shape[0], shape[1]).T
    return all_masks


# In[ ]:


from PIL import Image
from sklearn.model_selection import train_test_split

train_glob = f'{train_dicom_dir}/*/*/*.dcm'
test_glob = f'{test_dicom_dir}/*/*/*.dcm'

exclude_list = []
train_names = [f for f in sorted(glob.glob(train_glob)) if f not in exclude_list]
test_names = [f for f in sorted(glob.glob(test_glob)) if f not in exclude_list]

print(len(train_names), len(test_names))
# print(train_names[0], test_names[0])
# !ls -l {os.path.join(train_dicom_dir, train_names[0])}


# In[ ]:


# training dataset
SEGMENTATION = DATA_DIR + '/train-rle.csv'
anns = pd.read_csv(SEGMENTATION)
anns.head()


# In[ ]:


# get rid of damn space in column name
anns.columns = ['ImageId', 'EncodedPixels']


# In[ ]:


# over-sample pneumothorax
pneumothorax_anns = anns[anns.EncodedPixels != ' -1'].ImageId.unique().tolist()
print(f'Positive samples: {len(pneumothorax_anns)}/{len(anns.ImageId.unique())} {100*len(pneumothorax_anns)/len(anns.ImageId.unique()):.2f}%')


# In[ ]:


# ## use only pneumothorax images
# pneumothorax_fps_train = [fp for fp in train_names if fp.split('/')[-1][:-4] in pneumothorax_anns]

# image_fps_train, image_fps_val = train_test_split(pneumothorax_fps_train, test_size=0.1, random_state=42)

# test_image_fps = test_names

# if debug:
#     print('DEBUG subsampling from:', len(image_fps_train), len(image_fps_val), len(test_image_fps))
#     image_fps_train = image_fps_train[:150] 
#     image_fps_val = image_fps_val[:150]
# #     test_image_fps = test_names[:150]
    
# print(len(image_fps_train), len(image_fps_val), len(test_image_fps))


# In[ ]:


## split and rebalance dataset
test_size = config.VALIDATION_STEPS * config.IMAGES_PER_GPU
image_fps_train, image_fps_val = train_test_split(train_names, test_size=test_size, random_state=42)
test_image_fps = test_names

pneumothorax_fps_train = [fp for fp in image_fps_train if fp.split('/')[-1][:-4] in pneumothorax_anns]

if debug:
    print('DEBUG subsampling from:', len(image_fps_train), len(image_fps_val), len(test_image_fps))
    image_fps_train = image_fps_train[:100] + pneumothorax_fps_train[:50] 
    image_fps_val = image_fps_val[:150]
#     test_image_fps = test_names[:150]
else:
    image_fps_train += pneumothorax_fps_train*3  # oversample positive cases
    random.shuffle(image_fps_train)
    
print(len(image_fps_train), len(image_fps_val), len(test_image_fps))
pos, total = len([fp for fp in image_fps_train if fp in pneumothorax_fps_train]), len(image_fps_train)
print(f'Positive samples in training: {pos}/{total} {100*pos/total:.2f}%')


# In[ ]:


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumothorax', 1, 'Pneumothorax')
        
        # add images 
        for i, fp in enumerate(image_fps):
            image_id = fp.split('/')[-1][:-4]
            annotations = image_annotations.query(f"ImageId=='{image_id}'")['EncodedPixels']
            self.add_image('pneumothorax', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
#         print(image_id, annotations)
        count = len(annotations)
        if count == 0 or (count == 1 and annotations.values[0] == ' -1'): # empty annotation
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle2mask(a, info['orig_height'], info['orig_width']).T
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


# ## Examine the annotation data
# ..., parse the dataset, and view dicom fields

# In[ ]:


image_fps, image_annotations = train_names, anns


# In[ ]:


ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
# image = ds.pixel_array # get image array
print(ds)
del ds; gc.collect()


# In[ ]:


# Original image size: 1024 x 1024
ORIG_SIZE = 1024


# ## Create and prepare the training dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', '# prepare the training dataset\ndataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)\ndataset_train.prepare()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# prepare the validation dataset\ndataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)\ndataset_val.prepare()')


# ## Display a random image with bounding boxes

# In[ ]:


# Load and display random sample and their bounding boxes

class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_val.image_ids)
    image_fp = dataset_val.image_reference(image_id)
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)

del masked


# ## Image Augmentation
# Try finetuning some variables to custom values

# In[ ]:


# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# test on the same image as above
imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid[:, :, 0], cmap='gray')
del imggrid; del image


# ## Training
# Now it's time to train the model. Note that training even a basic model can take a few hours. 
# 
# Note: the following model is for demonstration purpose only. We have limited the training to a few epochs, and have set nominal values for the Detector Configuration to reduce run-time. 
# 
# - dataset_train and dataset_val are derived from DetectorDataset 
# - DetectorDataset loads images from image filenames and  masks from the annotation data
# - model is Mask-RCNN

# In[ ]:


get_ipython().run_cell_magic('time', '', '# get pixel statistics\nimage_stats = []\nfor image_id in dataset_val.image_ids[:4]:\n    image = dataset_val.load_image(image_id)\n    image_stats.append(image.mean(axis=(0,1)))\n\nconfig.MEAN_PIXEL = np.mean(image_stats, axis=0).tolist()\n# VAR_PIXEL = images.var()\ndel image; del image_stats\ngc.collect()\n\nprint(config.MEAN_PIXEL)')


# In[ ]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

# load all weights as number of classes matches the pre-trained model
model.load_weights(WEIGHTS_PATH, by_name=True)

# # Exclude the last layers because they require a matching number of classes
# model.load_weights(WEIGHTS_PATH, by_name=True,
#                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


# In[ ]:


# Train the Mask-RCNN Model
LEARNING_RATE = 0.0006


# In[ ]:


get_ipython().run_cell_magic('time', '', "## train heads with higher lr to speedup the learning\nmodel.train(dataset_train, dataset_val,\n            learning_rate=LEARNING_RATE*2,\n            epochs=1,\n            layers='heads',\n            augmentation=None)  ## no need to augment yet\n\nhistory = model.keras_model.history.history")


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.train(dataset_train, dataset_val,\n            learning_rate=LEARNING_RATE,\n            epochs=3 if debug else 13,\n            layers='all',\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]")


# In[ ]:


# %%time
# model.train(dataset_train, dataset_val,
#             learning_rate=LEARNING_RATE/2,
#             epochs=4 if debug else 18,
#             layers='all',
#             augmentation=augmentation)

# new_history = model.keras_model.history.history
# for k in new_history: history[k] = history[k] + new_history[k]


# In[ ]:


epochs = range(1, len(history['loss'])+1)
pd.DataFrame(history, index=epochs)


# In[ ]:


plt.figure(figsize=(21,11))

plt.subplot(231)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(232)
plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
plt.legend()
plt.subplot(233)
plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
plt.legend()
plt.subplot(234)
plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
plt.legend()
plt.subplot(235)
plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
plt.legend()
plt.subplot(236)
plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
plt.legend()

plt.show()


# In[ ]:


best_epoch = np.argmin(history["val_loss"])
score = history["val_loss"][best_epoch]
print(f'Best Epoch:{best_epoch+1} val_loss:{score}')


# In[ ]:


# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))

fps = []
# Pick last directory
for d in dir_names: 
    dir_name = os.path.join(model.model_dir, d)
    # Find checkpoints
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        raise Exception(f'No weight files in {dir_name}')
    if best_epoch < len(checkpoints):
        checkpoint = checkpoints[best_epoch]
    else:
        checkpoint = checkpoints[-1]
    fps.append(os.path.join(dir_name, checkpoint))

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))


# In[ ]:


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# ## Validation
# How does the predicted box compared to the expected value? Let's use the validation dataset to check. 

# In[ ]:


# Show few example of ground truth vs. predictions on the validation dataset 
dataset = dataset_val
pneumothorax_ids_val = [fp.split('/')[-1][:-4] for fp in image_fps_val]
pneumothorax_ids_val = [i for i,id in enumerate(pneumothorax_ids_val) if id in pneumothorax_anns]
fig = plt.figure(figsize=(10, 40))

for i in range(8):
    image_id = random.choice(pneumothorax_ids_val)
    
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
#     print(original_image.shape)
    plt.subplot(8, 2, 2*i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    
    plt.subplot(8, 2, 2*i + 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])


# ## Basic Model Weigths Analysis

# In[ ]:


# Show stats of all trainable weights    
visualize.display_weight_stats(model)
### Click to expand output


# In[ ]:


# from https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_weights.ipynb
# Pick layer types to display
LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, 
                     layers))
# Display Histograms
fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)), gridspec_kw={"hspace":1})
for l, layer in enumerate(layers):
    weights = layer.get_weights()
    for w, weight in enumerate(weights):
        tensor = layer.weights[w]
        ax[l, w].set_title(f'Layer:{l}.{w} {tensor.name}')
        _ = ax[l, w].hist(weight[w].flatten(), 50)


# ## Final steps
# Create the submission file

# In[ ]:


# Make predictions on test images, write out submission file
def predict(image_fps, filepath='submission.csv', min_conf=0.97):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write("ImageId,EncodedPixels\n")

        for fp in tqdm_notebook(image_fps):
            image_id = fp.split('/')[-1][:-4]
            maks_written = 0
            
            ds = pydicom.read_file(fp)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            results = model.detect([image])
            r = results[0]

#             assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            num_instances = len(r['rois'])

            for i in range(num_instances):
                if r['scores'][i] > min_conf and np.sum(r['masks'][...,i]) > 1:
                    mask = r['masks'][...,i].T*255
                    mask, _,_,_,_ = utils.resize_image(
                        np.stack((mask,) * 3, -1), # requires 3 channels
                        min_dim=ORIG_SIZE,
                        min_scale=config.IMAGE_MIN_SCALE,
                        max_dim=ORIG_SIZE,
                        mode=config.IMAGE_RESIZE_MODE)
                    mask = (mask[...,0] > 0)*255
#                     print(mask.shape)
#                     plt.imshow(mask, cmap=get_cmap('jet'))
                    file.write(image_id + "," + mask2rle(mask, ORIG_SIZE, ORIG_SIZE) + "\n")
                    maks_written += 1

            if maks_written == 0:
                file.write(image_id + ",-1\n")  ## no pneumothorax


# In[ ]:


submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
predict(test_image_fps, filepath=submission_fp)
print(submission_fp)


# In[ ]:


sub = pd.read_csv(submission_fp, dtype={'ImageId':str, 'EncodedPixels':str})
print((sub.EncodedPixels != '-1').sum(), sub.ImageId.size, sub.ImageId.nunique())
print(sub.EncodedPixels.nunique(), (sub.EncodedPixels != '-1').sum()/sub.ImageId.nunique())

print('Unique samples:\n', sub.EncodedPixels.drop_duplicates()[:6])
sub.head(10)


# In[ ]:


# show a few test image detection example
def visualize_test():
    ids_with_mask = sub[sub.EncodedPixels != '-1'].ImageId.values
    fp = random.choice([fp for fp in test_image_fps if fp.split('/')[-1][:-4] in ids_with_mask])
#     import pdb; pdb.set_trace()
    
    # original image
    image_id = fp.split('/')[-1][:-4]
    ds = pydicom.read_file(fp)
    image = ds.pixel_array
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    
    # assume square image 
    resize_factor = 1 ## ORIG_SIZE / config.IMAGE_SHAPE[0]

    # Detect on full size test images (without resizing)
    results = model.detect([image])
    r = results[0]
    for bbox in r['rois']: 
#         print(bbox)
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2]  * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width = x2 - x1
        height = y2 - y1
#         print("x {} y {} h {} w {}".format(x1, y1, width, height))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title(image_id)
    ax1.imshow(image)
    ax2.set_title(f"{len(r['rois'])} masks predicted again")
    if len(r['rois']) > 0:
        ax2.imshow(r['masks'].max(-1))  # get max (overlap) between all masks in this prediction
    ax3.set_title(f"{np.count_nonzero(image_id == ids_with_mask)} masks in csv")
    ax3.imshow(masks_as_color(sub.query(f"ImageId=='{image_id}'")['EncodedPixels'].values, (ORIG_SIZE, ORIG_SIZE)))
#     print(f"ImageId=='{image_id}'", sub.query(f"ImageId=='{image_id}'")['EncodedPixels'])

for i in range(8):
    visualize_test()


# In[ ]:


os.chdir(ROOT_DIR)
get_ipython().system('ls *')


# In[ ]:


# remove files to allow committing (hit files limit otherwise)
get_ipython().system('rm -rf {STAGE_DIR}  /kaggle/working/*/events*')

