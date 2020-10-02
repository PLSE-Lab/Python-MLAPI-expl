#!/usr/bin/env python
# coding: utf-8

# # Identifikation von Drusenarten in OCT-Images
# ### Unter der Verwendung des matterplot Mask-R-CNN Ansatzes

# **Helpful-Links:**
# * https://github.com/matterport/Mask_RCNN

# # Setup Umgebung

# In[ ]:


import sys
import os
get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git ../Mask_RCNN')
os.chdir('../Mask_RCNN')


# In[ ]:


get_ipython().system('pip install -r requirements.txt')
get_ipython().system('pip install tensorflow-gpu==1.15')
get_ipython().system('pip install keras==2.2.5')


# In[ ]:


get_ipython().system('conda install -c anaconda --yes cudatoolkit')
get_ipython().system('conda install -c anaconda --yes cudnn')


# ## Pfade

# In[ ]:


ROOT_PATH='/kaggle/working'
MASK_PATH='/kaggle/working/Mask_RCNN'
MODEL_PATH='/kaggle/working/Mask_RCNN/logs'
TRAIN_PATH='/kaggle/input/kermany2018/oct2017/OCT2017 /train/DRUSEN'
TEST_PATH='/kaggle/input/kermany2018/oct2017/OCT2017 /test/DRUSEN'
VAL_PATH='/kaggle/input/kermany2018/oct2017/OCT2017 /val/DRUSEN'
JSON_PATH='/kaggle/input/labels'


# # Import

# In[ ]:


import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.abspath(ROOT_PATH)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# #  Configurations

# In[ ]:


class OCTConfig(Config):
   
    NAME = "OCT"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # TODO CHANGE
    NUM_CLASSES = 1 + 3  # Background + ObereSchicht + Membran + UntereSchicht

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 125

    DETECTION_MIN_CONFIDENCE = 0.75

    TRAIN_ROIS_PER_IMAGE = 200


# # Dataset

# In[ ]:


class OCTDataset(utils.Dataset):

    def load_oct(self, subset):

        self.add_class("oct", 1, "pigment")  # adjusted here
        self.add_class("oct", 2, "soft")  # adjusted here
        self.add_class("oct", 3, "hard")  # adjusted here

        dataset_dir = ""
        # Train or validation dataset?
        if subset == "train":
            dataset_dir=TRAIN_PATH
        if subset == "val":
            dataset_dir=VAL_PATH
             
        annotations = json.load(open(os.path.join(JSON_PATH, "via_export_json-"+subset+".json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        numberImages = 0

        # Add images
        for a in annotations:

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            class_names_str = [r['region_attributes']['druse'] for r in a['regions']]
            class_name_nums = []

            for i in class_names_str:
                if i == 'pigment':
                    class_name_nums.append(1)
                if i == 'soft':
                    class_name_nums.append(2)
                if i == 'hard':
                    class_name_nums.append(3)

            image_path = os.path.join(dataset_dir, a['filename'])
            
            # Thanks to MAX
            image_path = image_path.replace("val","train")
            
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            numberImages = numberImages+1

            self.add_image(
                "oct",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_list=np.array(
                    class_name_nums))  # UNSURE IF  I CAN JUST ADD THIS  HERE. OTHERWISE NEED  TO MODIFY DATASET UTIL

            self.dataset_size = numberImages

    def load_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "oct":  # adjusted here
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_array = info['class_list']
        return mask.astype(np.bool), class_array

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "oct":  # adjusted here
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# #  Training

# In[ ]:


weights = "coco"
logs="logs"
command = "train"

# Configurations
if command == "train":
    config = OCTConfig()
config.display()

# Create model
if command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs)

model.keras_model.summary()

# Select weights file to load
if weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
elif weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
else:
    weights_path = weights

# Load weights
print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)


# ## Augmentation

# In[ ]:


augmentation = iaa.SomeOf((0, 2), [
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops

        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)

img=mpimg.imread("/kaggle/input/kermany2018/oct2017/OCT2017 /train/DRUSEN/DRUSEN-9201573-14.jpeg")
imggrid = augmentation.draw_grid(img, cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid.astype(int))


# In[ ]:


# Training dataset.
print('preparing training set')
dataset_train = OCTDataset()
dataset_train.load_oct("train")
dataset_train.prepare()

# Validation dataset
print('preparing val set')
dataset_val = OCTDataset()
dataset_val.load_oct("val")
dataset_val.prepare()


# In[ ]:


# starting_epoch = model.epoch
epoch = int(dataset_train.dataset_size / (config.STEPS_PER_EPOCH * config.BATCH_SIZE) * 100)
epochs_warmup = epoch
epochs_heads = 3 * epoch
epochs_stage4 = 3 * epoch
epochs_all =  epoch
epochs_breakOfDawn = 3 * epoch

print("> Training Schedule:     \nwarmup: {} epochs     \nheads: {} epochs     \nstage4+: {} epochs     \nall layers: {} epochs     \ntill the break of Dawn: {} epochs".format(
    epochs_warmup,epochs_heads,epochs_stage4,epochs_all,epochs_breakOfDawn))


# In[ ]:


get_ipython().run_cell_magic('time', '', '## Training - WarmUp Stage\nprint("> Warm Up all layers")\nmodel.train(dataset_train, dataset_val,\n            learning_rate=config.LEARNING_RATE / 10,\n            epochs=epochs_warmup,\n            layers=\'all\',\n            augmentation=augmentation)\n\nhistory = model.keras_model.history.history')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Training - Stage 1\nprint("> Training network heads")\nmodel.train(dataset_train, dataset_val,\n            learning_rate=config.LEARNING_RATE,\n            epochs= epochs_warmup,\n            layers=\'heads\',\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Training - Stage 2\n# Finetune layers  stage 4 and up\nprint("> Fine tune {} stage 4 and up".format(config.BACKBONE))\nmodel.train(dataset_train, dataset_val,\n            learning_rate=config.LEARNING_RATE,\n            epochs=epochs_stage4,\n            layers="4+",\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Training - Stage 3\n# Fine tune all layers\nprint("> Fine tune all layers")\nmodel.train(dataset_train, dataset_val,\n            learning_rate=config.LEARNING_RATE / 10,\n            epochs=epochs_all,\n            layers=\'all\',\n            augmentation=augmentation)\n\nnew_history = model.keras_model.history.history\nfor k in new_history: history[k] = history[k] + new_history[k]')


# ## Metrics

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

best_epoch = np.argmin(history["val_loss"])
score = history["val_loss"][best_epoch]
print(f'Best Epoch:{best_epoch+1} val_loss:{score}')


# # Result

# ## Load Model

# In[ ]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = MODEL_PATH


# ## Configuration

# In[ ]:


config = OCTConfig()
OCT_DIR = os.path.join(ROOT_DIR, "")
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.DETECTION_MIN_CONFIDENCE = 0.75
config.display()

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Load Validation Dataset
# 

# In[ ]:


# Load validation dataset
dataset = OCTDataset()
dataset.load_oct("val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model

# In[ ]:


with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir='/kaggle/Mask_RCNN/logs/',
                              config=config)

weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# ## Run Detection

# In[ ]:


#Display results
numberPictures=min(len(dataset.image_ids),5)
ax = get_ax(rows=numberPictures,cols=2)

for i in range(0,numberPictures):
    image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           dataset.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=1)

    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax[i,0],
                                title="Predictions ")
    
    image2=mpimg.imread(dataset.image_reference(image_id))
    ax[i,1].set_title("Blank")
    ax[i,1].imshow(image2)
        
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)


# ## RPN Targets

# In[ ]:


# Generate RPN trainig targets
# target_rpn_match is 1 for positive anchors, -1 for negative anchors
# and 0 for neutral anchors.
target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
    image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
log("target_rpn_match", target_rpn_match)
log("target_rpn_bbox", target_rpn_bbox)

positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
positive_anchors = model.anchors[positive_anchor_ix]
negative_anchors = model.anchors[negative_anchor_ix]
neutral_anchors = model.anchors[neutral_anchor_ix]
log("positive_anchors", positive_anchors)
log("negative_anchors", negative_anchors)
log("neutral anchors", neutral_anchors)

# Apply refinement deltas to positive anchors
refined_anchors = utils.apply_box_deltas(
    positive_anchors,
    target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
log("refined_anchors", refined_anchors, )

visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())


# ## RPN Predictions

# In[ ]:


# Run RPN sub-graph
pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

# TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
if nms_node is None:
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
if nms_node is None: #TF 1.9-1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

rpn = model.run_graph([image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", nms_node),
    ("proposals", model.keras_model.get_layer("ROI").output),
])

limit = 100
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())


# In[ ]:


# Show top anchors with refinement. Then with clipping to image boundaries
limit = 50
ax = get_ax(1, 2)
pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                     refined_boxes=refined_anchors[:limit], ax=ax[0])
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])


# In[ ]:


# Show refined anchors after non-max suppression
limit = 50
ixs = rpn["post_nms_anchor_ix"][:limit]
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())


# In[ ]:


# Show final proposals
# These are the same as the previous step (refined anchors 
# after NMS) but with coordinates normalized to [0, 1] range.
limit = 50
# Convert back to image coordinates for display
h, w = config.IMAGE_SHAPE[:2]
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())


# In[ ]:


# Get input and output to classifier and mask heads.
mrcnn = model.run_graph([image], [
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("probs", model.keras_model.get_layer("mrcnn_class").output),
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
])


# In[ ]:


# Get detection class IDs. Trim zero padding.
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]
detections = mrcnn['detections'][0, :det_count]

print("{} detections: {}".format(
    det_count, np.array(dataset.class_names)[det_class_ids]))

captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
            for c, s in zip(detections[:, 4], detections[:, 5])]
visualize.draw_boxes(
    image, 
    refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
    visibilities=[2] * len(detections),
    captions=captions, title="Detections",
    ax=get_ax())


# ##  Proposal Classification

# In[ ]:


# Get input and output to classifier and mask heads.
mrcnn = model.run_graph([image], [
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("probs", model.keras_model.get_layer("mrcnn_class").output),
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
])


# In[ ]:


# Get detection class IDs. Trim zero padding.
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]
detections = mrcnn['detections'][0, :det_count]

print("{} detections: {}".format(
    det_count, np.array(dataset.class_names)[det_class_ids]))

captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
            for c, s in zip(detections[:, 4], detections[:, 5])]
visualize.draw_boxes(
    image, 
    refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
    visibilities=[2] * len(detections),
    captions=captions, title="Detections",
    ax=get_ax())


# ## Step by Step Detection

# In[ ]:


# Proposals are in normalized coordinates. Scale them
# to image coordinates.
h, w = config.IMAGE_SHAPE[:2]
proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

# Class ID, score, and mask per proposal
roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
roi_class_names = np.array(dataset.class_names)[roi_class_ids]
roi_positive_ixs = np.where(roi_class_ids > 0)[0]

# How many ROIs vs empty rows?
print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
print("{} Positive ROIs".format(len(roi_positive_ixs)))

# Class counts
print(list(zip(*np.unique(roi_class_names, return_counts=True))))


# In[ ]:


# Display a random sample of proposals.
# Proposals classified as background are dotted, and
# the rest show their class and confidence score.
limit = 200
ixs = np.random.randint(0, proposals.shape[0], limit)
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
visualize.draw_boxes(image, boxes=proposals[ixs],
                     visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                     captions=captions, title="ROIs Before Refinement",
                     ax=get_ax())


# ## Apply Bounding Box Refinement

# In[ ]:


# Class-specific bounding box shifts.
roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
log("roi_bbox_specific", roi_bbox_specific)

# Apply bounding box transformations
# Shape: [N, (y1, x1, y2, x2)]
refined_proposals = utils.apply_box_deltas(
    proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
log("refined_proposals", refined_proposals)

# Show positive proposals
# ids = np.arange(roi_boxes.shape[0])  # Display all
limit = 5
ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                     refined_boxes=refined_proposals[roi_positive_ixs][ids],
                     visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                     captions=captions, title="ROIs After Refinement",
                     ax=get_ax())


# ## Filter Low Confidence Detections

# In[ ]:


keep = np.where(roi_class_ids > 0)[0]
print("Keep {} detections:\n{}".format(keep.shape[0], keep))

# Remove low confidence detections
keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
print("Remove boxes below {} confidence. Keep {}:\n{}".format(
    config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))


# ## Per-Class Non-Max Suppression

# In[ ]:


# Apply per-class non-max suppression
pre_nms_boxes = refined_proposals[keep]
pre_nms_scores = roi_scores[keep]
pre_nms_class_ids = roi_class_ids[keep]

nms_keep = []
for class_id in np.unique(pre_nms_class_ids):
    # Pick detections of this class
    ixs = np.where(pre_nms_class_ids == class_id)[0]
    # Apply NMS
    class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                            pre_nms_scores[ixs],
                                            config.DETECTION_NMS_THRESHOLD)
    # Map indicies
    class_keep = keep[ixs[class_keep]]
    nms_keep = np.union1d(nms_keep, class_keep)
    print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20], 
                                   keep[ixs], class_keep))

keep = np.intersect1d(keep, nms_keep).astype(np.int32)
print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))


# In[ ]:


# Show final detections
ixs = np.arange(len(keep))  # Display all
# ixs = np.random.randint(0, len(keep), 10)  # Display random sample
captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
visualize.draw_boxes(
    image, boxes=proposals[keep][ixs],
    refined_boxes=refined_proposals[keep][ixs],
    visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
    captions=captions, title="Detections after NMS",
    ax=get_ax())


# ## Generating Masks

# In[ ]:


display_images(np.transpose(gt_mask, [2, 0, 1]), cmap="Blues")


# In[ ]:


# Get predictions of mask head
mrcnn = model.run_graph([image], [
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
])

# Get detection class IDs. Trim zero padding.
det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]

print("{} detections: {}".format(
    det_count, np.array(dataset.class_names)[det_class_ids]))


# In[ ]:


# Masks
det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                              for i, c in enumerate(det_class_ids)])
det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                      for i, m in enumerate(det_mask_specific)])
log("det_mask_specific", det_mask_specific)
log("det_masks", det_masks)


# In[ ]:


display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")


# In[ ]:


display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")


# ## Visualize Activations
# 

# In[ ]:


# Get activations of a few sample layers
activations = model.run_graph([image], [
    ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
    ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
    ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
    ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
    ("roi",                model.keras_model.get_layer("ROI").output),
])


# In[ ]:


# Input image (normalized)
_ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))


# In[ ]:


# Backbone feature map
display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)

