#!/usr/bin/env python
# coding: utf-8

# Hi kagglers! Here I am going to show how to use a toolbox to test the object detection baseline in Global Wheat Detection problem.
# 
# [MMDetection](https://github.com/open-mmlab/mmdetection) is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by Multimedia Laboratory, CUHK.
# 
#  [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](http://https://github.com/TuSimple/simpledet) are codebases for computer vision tasks, but the training speed of MMDetection is faster and updated frequency.

# In the Global Wheat Detection Challenge we can not use Internet during inference, so it is not easy to use MMDetection as there are many dependences that are not installed in default kaggle environment.

# First, I followed this [blog](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/113195) to install python dependences.

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


# We define a function `gen_test_annotation`, use the hot-plug way to generate the tesing annotation to fit the final tesing environment.

# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def gen_test_annotation(test_data_path, annotation_path):
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            img_info = {}
            img_info['filename'] = img
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['width'] = img_size[0]
            img_info['height'] = img_size[1]
            test_anno_list.append(img_info)
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_list, f)


# In[ ]:


DIR_INPUT = '/kaggle/working/mmdetection/data/Wheatdetection'
DIR_TEST = f'{DIR_INPUT}/test'
DIR_ANNO = f'{DIR_INPUT}/annotations'

DIR_WEIGHTS = '/kaggle/input/mmdetfasterrcnn'
WEIGHTS_FILE = f'{DIR_WEIGHTS}/epoch_50.pth'

test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

# prepare test data annotations
gen_test_annotation(DIR_TEST, DIR_ANNO + '/detection_test.json')


# In[ ]:


config_file = '/kaggle/working/mmdetection/configs/wheatdetection/faster_rcnn_r50_fpn_1x_coco_test.py'
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True

distributed = False


# In[ ]:


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=distributed,
    shuffle=False)


# In[ ]:


#################################### faster rcnn ############################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, WEIGHTS_FILE, map_location='cpu')

model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader, False, None, 0.5)

results = []

for images_info, result in zip(dataset.data_infos, outputs):
    boxes = result[0][:, :4]
    scores = result[0][:, 4]

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    result = {
        'image_id': images_info['filename'][:-4],
        'PredictionString': format_prediction_string(boxes, scores)
    }

    results.append(result)


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

# save result
test_df.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('rm -rf mmdetection/')

