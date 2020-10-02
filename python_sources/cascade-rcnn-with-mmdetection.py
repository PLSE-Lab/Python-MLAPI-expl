#!/usr/bin/env python
# coding: utf-8

# # https://www.kaggle.com/superkevingit/faster-rcnn-with-mmdetection-without-internet

# In[ ]:


get_ipython().system('pip install ../input/mmcvwhl/addict-2.2.1-py3-none-any.whl')
get_ipython().system('pip install ../input/mmdetection20-5-13/mmcv-0.5.1-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install ../input/mmdetection20-5-13/terminal-0.4.0-py3-none-any.whl')
get_ipython().system('pip install ../input/mmdetection20-5-13/terminaltables-3.1.0-py3-none-any.whl')


# Copy the MMDetection framework to the writeable directoy.

# In[ ]:


get_ipython().system('cp -r ../input/mmdetection20-5-13/mmdetection/mmdetection .')


# We should prepare the data and convert to COCO format.

# In[ ]:


get_ipython().system('mkdir -p mmdetection/data/Wheatdetection/annotations')
get_ipython().system('cp -r ../input/global-wheat-detection/test mmdetection/data/Wheatdetection/test')
get_ipython().system('cp -r ../input/global-wheat-detection/sample_submission.csv mmdetection/data/Wheatdetection/')
get_ipython().system('mkdir mmdetection/configs/wheatdetection')


# Some config files for faster-rcnn implement and test data.

# In[ ]:


get_ipython().system('cp ../input/mmdetfasterrcnn/config/config/faster_rcnn_r50_fpn_1x_coco_test.py mmdetection/configs/wheatdetection')
get_ipython().system('cp ../input/mmdetfasterrcnn/config/config/wheat_detection_test.py mmdetection/configs/_base_/datasets')
get_ipython().system('cp ../input/mmdetfasterrcnn/config/config/__init__.py mmdetection/mmdet/datasets')
get_ipython().system('cp ../input/mmdetfasterrcnn/config/config/wheat.py mmdetection/mmdet/datasets')


# In[ ]:


cd mmdetection


# compile the coco toolbox

# In[ ]:


get_ipython().system('cp -r ../../input/mmdetection20-5-13/cocoapi/cocoapi .')


# In[ ]:


cd cocoapi/PythonAPI


# In[ ]:


get_ipython().system('make')


# In[ ]:


get_ipython().system('make install')


# In[ ]:


get_ipython().system('python setup.py install')


# In[ ]:


import pycocotools


# In[ ]:


cd ../..


# compile the MMDetection Framework

# In[ ]:


get_ipython().system('pip install -v -e .')


# In[ ]:


cd ../


# Now, we can use MMDetection for the inference.

# In[ ]:


import sys
sys.path.append('mmdetection') # To find local version


# In[ ]:


from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset

import pandas as pd
import os
import json

from PIL import Image

import torch


# In[ ]:


from mmdet.apis import inference_detector, init_detector
WEIGHTS_FILE = '../input/zlb0525/epoch_18.pth'# f'{DIR_WEIGHTS}/epoch_50.pth'
config_file = '../input/zlb0525/cascade_rcnn_r50_fpn_1x_coco.py'
model = init_detector(config_file, WEIGHTS_FILE)


# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


# In[ ]:


#################################### cascade rcnn ############################################
DATA_ROOT_PATH = '../input/global-wheat-detection/test/'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()
from tqdm import tqdm
results = []
with torch.no_grad():
    for img_name in tqdm(os.listdir(DATA_ROOT_PATH)):
        img_pth = os.path.join(DATA_ROOT_PATH, img_name)
        result = inference_detector(model, img_pth)
        
        boxes = result[0][:, :4]
        scores = result[0][:, 4]
        if len(boxes) > 0:
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': img_name[:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()


# In[ ]:


get_ipython().system('rm -rf mmdetection/')

