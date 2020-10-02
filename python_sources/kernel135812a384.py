# %% [code]
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

# %% [code]
DATA_DIR = '/kaggle/input'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'

# %% [code]
#!pip install -I keras==2.1.0

# %% [code]
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('/kaggle/working')

# %% [code]
import sys
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

# %% [code]
from xml.etree import ElementTree
from mrcnn.utils import Dataset
import numpy as np
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class DogDataset(Dataset):
    # load the dataset definitions
    def read_class(self):
        list_class = os.listdir('/kaggle/input/stanford-dogs-dataset/annotations/Annotation')
        id = []
        name = []
        for classi in list_class:
            arr_str = classi.split('-')
            id.append(arr_str[0])
            name.append('-'.join(arr_str[1:]))
        return id, name
    def load_dataset(self, is_train=True):
        id_class, name_class = self.read_class()
        for id, name in zip(id_class, name_class):
            self.add_class("dataset", id, name)
            images_dir = '/kaggle/input/stanford-dogs-dataset/images/Images/'+id+'-'+name+'/'
            annotations_dir = '/kaggle/input/stanford-dogs-dataset/annotations/Annotation/'+id+'-'+name+'/'
            num = 0
            # find all images
            for filename in os.listdir(images_dir):
                # extract image id
                num = num + 1
                image_id = filename[:-4]
                img_path = images_dir + filename
                ann_path = annotations_dir + image_id
                # add to dataset
                if is_train and num >= 140:
                    continue
                if not is_train and num < 140:
                    continue
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
    # ...
    # load the masks for an image
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h, class_obj = self.extract_boxes(path)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1,
            class_ids.append(self.class_names.index(class_obj))
        return masks, np.asarray(class_ids, dtype='int32')
    # ...
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    # ...
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        class_obj = root.find('.//object/name').text
        return boxes, width, height, class_obj
# define a configuration for the model
class DogConfig(Config):
    # define the name of the configuration
   NAME = 'dog_cfg'
   #BACKBONE = 'resnet50'
   NUM_CLASSES = 121
   IMAGES_PER_GPU = 2
   GPU_COUNT = 1
   # number of training steps per epoch
    #STEPS_PER_EPOCH = 16680 // (IMAGES_PER_GPU * GPU_COUNT)
    #VALIDATION_STEPS = 3900 // (IMAGES_PER_GPU * GPU_COUNT)
   IMAGE_MIN_DIM = 256
   IMAGE_MAX_DIM = 256
   STEPS_PER_EPOCH = 3000
   VALIDATION_STEPS = 50
   DETECTION_MAX_INSTANCES = 1
   DETECTION_MIN_CONFIDENCE = 0.9
   DETECTION_NMS_THRESHOLD = 0.1
   # ...
# train set
train_set = DogDataset()
train_set.load_dataset( is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# test/val set
test_set = DogDataset()
test_set.load_dataset( is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# %% [code]
config = DogConfig()
config.display()

# %% [code]
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
os.chdir('/kaggle/working')

# %% [code]
# define the model
model = MaskRCNN(mode='training', model_dir='../', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('/kaggle/working/mask_rcnn_coco.h5', by_name=True,
                  exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                           "mrcnn_bbox", "mrcnn_mask"])

# %% [code]
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE/10, epochs=2, layers='all')

# %% [code]
from mrcnn import visualize
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import expand_dims
class InferenceConfig(DogConfig):
   GPU_COUNT = 1
   # 1 image for inference
   IMAGES_PER_GPU = 1
inference_config = InferenceConfig()
# create a model in inference mode
infer_model = MaskRCNN(mode="inference",
                         config=inference_config,
                         model_dir='../')
model_path = infer_model.find_last()
#model_path = '../mask_rcnn_coco.h5'
# Load trained weights
print("Loading weights from ", model_path)
infer_model.load_weights(model_path, by_name=True)
# Test on a random image
image_id = np.random.choice(train_set.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
   load_image_gt(train_set, inference_config,
                          image_id, use_mini_mask=False)
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                           train_set.class_names, figsize=(8, 8))
results = infer_model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                           train_set.class_names, r['scores'], figsize=(8, 8))