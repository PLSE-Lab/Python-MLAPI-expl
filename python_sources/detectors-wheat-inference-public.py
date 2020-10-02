#!/usr/bin/env python
# coding: utf-8

# Here is my first public kernel.
# I use DetectoRS and modify some source code.
# I pre-build DetectoRS (mmdetection) in a kernel.
# 
# No other additional training setting.
# 
# Use TTA: horizontal flip.
# 
# Socre: 0.6991
#     
# 
# I've implemented 'mosaic' in mmdetection and is testing this method.

# ## Install package

# In[ ]:


get_ipython().system('pip install /kaggle/input/mmdetection-package/addict-2.2.1-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/mmdetection-package/mmcv-0.5.9-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install /kaggle/input/mmdetection-package/pycocotools-2.0.0-cp37-cp37m-linux_x86_64.whl')


# In[ ]:


get_ipython().system('cp -r /kaggle/input/detectors ./DetectoRS')


# In[ ]:


cd DetectoRS


# In[ ]:


get_ipython().system('pip install -v -e .')


# In[ ]:


cd ..


# ## Import package

# In[ ]:


import argparse
import os
import json
import time
import sys
sys.path.append('/kaggle/working/DetectoRS')
sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes import *

import numpy as np
import pandas as pd

import cv2
import mmcv
import torch
import torchvision
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


# ## Generate Test files

# In[ ]:


CONFIG_FILE = '/kaggle/input/mmdetection-wheat-models/2/config.py'
TEST_IMG_DIR = '/kaggle/input/global-wheat-detection/test'
CHECKPOINT_PATH = '/kaggle/input/mmdetection-wheat-models/2/model.pth'


# ## Config

# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


# In[ ]:


def run_wbf(boxes,scores, image_size=1024, iou_thr=0.33, skip_box_thr=0.34, weights=None):
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes, scores, labels


# In[ ]:


config = mmcv.Config.fromfile(CONFIG_FILE)
config.model['neck']['rfp_pretrained'] = None

# TTA
config.data.test.pipeline[1]['flip'] = True
# config.data.test.pipeline[1]['flip_direction'] = ['horizontal', 'vertical']
config.data.test.pipeline[1]['flip_direction'] = ['horizontal']

config.test_cfg['rcnn']['score_thr'] = 0.005
config.test_cfg['rcnn']['nms']['iou_thr'] = 0.4


# In[ ]:


model = init_detector(config, CHECKPOINT_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


# In[ ]:


from tqdm import tqdm
results = []
with torch.no_grad():
    for img_name in tqdm(os.listdir(TEST_IMG_DIR)):
        img_pth = os.path.join(TEST_IMG_DIR, img_name)
        
        result = inference_detector(model, img_pth)
            
        boxes = result[0][:, :4]
        scores = result[0][:, 4]

        if len(boxes) > 0:
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes, scores, labels = run_wbf([boxes,],[scores,])
        
        result = {
            'image_id': img_name[:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
size = 300
idx =-1
font = cv2.FONT_HERSHEY_SIMPLEX 
image = cv2.imread(img_pth, cv2.IMREAD_COLOR)
fontScale = 1
color = (255, 0, 0) 

thickness = 2
for b,s in zip(boxes,scores):
    b = [int(a) for a in b]
    image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1) 
    image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(20),b[1]), font,  
                   fontScale, color, thickness, cv2.LINE_AA)
plt.figure(figsize=[20,20])
plt.imshow(image[:,:,::-1])
plt.show()


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/DetectoRS/')

