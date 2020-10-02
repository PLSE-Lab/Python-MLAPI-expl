#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hi guys, this notebook illustrates how to employ yoloV3 to perform Object Detection in Flickr images and COCO 2014 dataset. 99% of the codes are from this amazing repo https://github.com/zzh8829/yolov3-tf2 please star him.
# 

# In[ ]:


get_ipython().system('git clone https://github.com/zzh8829/yolov3-tf2')


# In[ ]:



get_ipython().run_line_magic('cd', 'yolov3-tf2')
get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls data')
print('coco.names')
get_ipython().system('cat data/coco.names | wc')
print('voc2012.names')
get_ipython().system('cat data/voc2012.names | wc')
get_ipython().system('cat data/coco.names')


# In[ ]:


get_ipython().system('pip install -r requirements-gpu.txt')


# In[ ]:


import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

print(tf.__version__)


# In[ ]:


get_ipython().system('cat convert.py # this function use to convert official yolov3 weights to Keras model')


# In[ ]:


get_ipython().system('wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights')
get_ipython().system('python convert.py')


# In[ ]:


import sys
from absl import app, logging, flags
from absl.flags import FLAGS
import time
import cv2
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from IPython.display import Image, display


# In[ ]:




flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

app._run_init(['yolov3'], app.parse_flags_with_usage)


# In[ ]:


# trick to better allocate GPU memory, otherwise, we will get OOM
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) 


# In[ ]:


FLAGS.tiny, FLAGS.classes


# In[ ]:




if FLAGS.tiny:
    yolo = YoloV3Tiny(classes=FLAGS.num_classes)
else:
    yolo = YoloV3(classes=FLAGS.num_classes)
      
yolo.load_weights(FLAGS.weights).expect_partial() # expect_partial just suppress some loading warning
logging.info('weights loaded')

class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
logging.info('classes loaded')


# In[ ]:


print(class_names)


# # Flickr dataset

# In[ ]:


flickr_path = '/kaggle/input/flickr8k-sau/Flickr_Data/Images/'
paths2 = sorted(os.listdir(flickr_path))
print(len(paths2))


# In[ ]:


'''
girl.png  meme2.jpeg    street.jpg	    
meme.jpg  meme_out.jpg  street_out.jpg 
'''
FLAGS.image = 'data/meme.jpg'
FLAGS.image = 'data/meme2.jpeg'
FLAGS.image = 'data/girl.png'
FLAGS.image = 'data/street.jpg'

for jj in range(20):
    FLAGS.image = flickr_path + paths2[jj]

    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
#     logging.info('time: {}'.format(t2 - t1))

#     logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    
    display(Image(data=bytes(cv2.imencode('.jpg', img)[1]), width=800))


# In[ ]:


nums_np = nums[0].numpy()
print(nums_np)

score_np = scores[0].numpy()
print(score_np[:(nums[0].numpy()+1)])

classes_np = classes[0].numpy().astype(int)
print(classes_np)


# In[ ]:


## see the details of drawing function
get_ipython().system('cat yolov3_tf2/utils.py')


# In[ ]:


for i in range(nums_np):
    print(class_names[classes_np[i]]) # Note that Person = class0


# # COCO dataset

# In[ ]:


annotation_file = '/kaggle/input/coco2014/captions/annotations/captions_train2014.json'
COCOPATH = '/kaggle/input/coco2014/train2014/train2014/'
get_ipython().system('ls {COCOPATH} | wc')


# In[ ]:


with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = {}
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = COCOPATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    
    if all_captions.get(all_img_name_vector[-1]) is None:
        all_captions[all_img_name_vector[-1]] = []
    
    all_captions[all_img_name_vector[-1]].append(caption)


# In[ ]:


len(all_captions), len(all_img_name_vector), image_id
print(all_img_name_vector[:5])
print(all_captions[list(all_captions.keys())[0]])


# In[ ]:




paths2 = sorted(os.listdir(COCOPATH))
print(len(paths2))

for jj in range(20):
    FLAGS.image =  COCOPATH + paths2[jj] # all_img_name_vector[jj] is repeated
    print('\n***',all_captions[COCOPATH + paths2[jj]],'***\n')
    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
#     logging.info('time: {}'.format(t2 - t1))

#     logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    
    display(Image(data=bytes(cv2.imencode('.jpg', img)[1]), width=800))


# In[ ]:




