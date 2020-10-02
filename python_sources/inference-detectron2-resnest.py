#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


get_ipython().system('pip install -U /kaggle/input/orkatzfdata/torch-1.5.0+cu101-cp37-cp37m-linux_x86_64.whl /kaggle/input/orkatzfdata/torchvision-0.6.0+cu101-cp37-cp37m-linux_x86_64.whl')


# In[ ]:


get_ipython().system("pip install '/kaggle/input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl'")
get_ipython().system('pip install /kaggle/input/orkatzfdata/yacs-0.1.7-py3-none-any.whl')
get_ipython().system('mkdir fvcore')
get_ipython().system("cp -R '/kaggle/input/orkatzfdata/fvcore-0.1.dev200407/fvcore-0.1.dev200407/' ./fvcore")
get_ipython().system('pip install fvcore/fvcore-0.1.dev200407/.')
get_ipython().system('mkdir detectron2-ResNeSt')
get_ipython().system('cp -R /kaggle/input/orkatzfdata/detectron2-ResNeSt/* ./detectron2-ResNeSt/')
get_ipython().system('pip install detectron2-ResNeSt/.')


# In[ ]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# In[ ]:


from detectron2.config import get_cfg
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 


cfg.MODEL.WEIGHTS = os.path.join('/kaggle/input/model-pth2/', "model_fold2.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set the testing threshold for this model
cfg.DATASETS.TEST = ("m5_val", )
predictor1 = DefaultPredictor(cfg)


# In[ ]:


import pandas as pd
df_sub = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')


# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


# In[ ]:


import cv2
import glob
results = []
for image_id in df_sub['image_id']:
    im = cv2.imread('/kaggle/input/global-wheat-detection/test/{}.jpg'.format(image_id))
    boxes = []
    scores = []
    labels = []
    outputs = predictor1(im)
    out = outputs["instances"].to("cpu")
    scores = out.get_fields()['scores'].numpy()
    boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int)
    labels= out.get_fields()['scores'].numpy()
    boxes = boxes.astype(int)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}
    results.append(result)


# In[ ]:


from matplotlib import pyplot as plt
image = im.copy()
size = 300
font = cv2.FONT_HERSHEY_SIMPLEX 
    
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
for b,s in zip(boxes,scores):
    image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1) 
    image = cv2.putText(image, '{:.2}'.format(s), (b[0],b[1]), font,  
                   fontScale, color, thickness, cv2.LINE_AA)
plt.figure(figsize=[20,20])
plt.imshow(image[:,:,::-1])
plt.show()


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)


# In[ ]:




