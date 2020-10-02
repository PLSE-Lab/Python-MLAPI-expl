#!/usr/bin/env python
# coding: utf-8

# I just wrote a side note on what is happening in every line, this is not definitly my original work

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #for file reading
import subprocess


# In[ ]:


#my pip was not updated
get_ipython().system('pip install --upgrade pip')


# In[ ]:


get_ipython().system('pip install -U /kaggle/input/tasnim/orkatzfdata/torch-1.5.0+cu101-cp37-cp37m-linux_x86_64.whl /kaggle/input/tasnim/orkatzfdata/torchvision-0.6.0+cu101-cp37-cp37m-linux_x86_64.whl')


# In[ ]:


get_ipython().system('pip install /kaggle/input/pycocotoolso/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install /kaggle/input/tasnim/orkatzfdata/yacs-0.1.7-py3-none-any.whl ')
if not os.path.exists('fvcore'):
    os.makedirs('fvcore') 
get_ipython().system("cp -R '/kaggle/input/tasnim/orkatzfdata/fvcore-0.1.dev200407/fvcore-0.1.dev200407/' ./fvcore ")
get_ipython().system('pip install fvcore/fvcore-0.1.dev200407/.')


# In[ ]:


get_ipython().system('cp -R /kaggle/input/tasnim/orkatzfdata/detectron2-ResNeSt/* ./detectron2-ResNeSt/')
get_ipython().system('pip install detectron2-ResNeSt/.')

if not os.path.exists('detectron2-ResNeSt'):
    os.makedirs('detectron2-ResNeSt ') 


# In[ ]:


import detectron2 
from detectron2.utils.logger import setup_logger #function to setup logging for libgs
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo #don't know
from detectron2.engine import DefaultPredictor #prediction
from detectron2.config import get_cfg #don't know
from detectron2.utils.visualizer import Visualizer #visualizing the results
from detectron2.data import MetadataCatalog #don't know


# CFG is a configuration file format used for storing settings. CFG files are created by many programs to store information and settings that differ from the factory defaults. CFG files usually appear as text documents, and can be opened by word processors though it is not recommended.

# In[ ]:


from detectron2.config import get_cfg #reading cfg file
cfg = get_cfg() #get_cfg changed it's name

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml")) #coco-detection
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  #only wheat detection, so class number is 1


cfg.MODEL.WEIGHTS = os.path.join('/kaggle/input/tasnim/best-inrae-1/', "model_final.pth") #get the model weights 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set the testing threshold for this model
cfg.DATASETS.TEST = ("m5_val", ) #datasets test don't know
predictor1 = DefaultPredictor(cfg) #predicting with predictor


# In[ ]:


import pandas as pd
df_sub = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv') #reading my submission file as dataframe


# In[ ]:


def format_prediction_string(boxes, scores): #format_prediction_string (csv file col 2)
    pred_strings = [] #empty set taken
    for j in zip(scores, boxes): 
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3])) #scores, bbox index

    return " ".join(pred_strings)


# The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order

# In[ ]:


import cv2 #image processing
import glob 
def norm(s):
    return s/s.max()-0.01 #normalization
results = []
for image_id in df_sub['image_id']:
    im = cv2.imread('/kaggle/input/global-wheat-detection/test/{}.jpg'.format(image_id)) #reading image with image id
    boxes = [] #bbox
    scores = [] #score
    labels = [] #label
    outputs = predictor1(im) #default_predictor
    out = outputs["instances"].to("cpu") #this detection is done in cpu
    scores = out.get_fields()['scores'].numpy() #from output, they are fetching the scores field, converting to np array
    boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int) #prediction box as int type
    labels= out.get_fields()['scores'].numpy() #labling the scores
    boxes = boxes.astype(int) #boxes: did not understand the boxes thingi
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)} #fusion the result
    results.append(result)


# In[ ]:


from matplotlib import pyplot as plt #for plotting data
image = im.copy() #copying the image
size = 300 #size of the image shown
font = cv2.FONT_HERSHEY_SIMPLEX  #the text_font
  
# org 
org = (50, 50) #why you exist bro
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) #Blue color my boxes
  
# Line thickness of 2 px 
thickness = 2
for b,s in zip(boxes,scores):
    image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1) #image shown in rectangle
    image = cv2.putText(image, '{:.2}'.format(s), (b[0],b[1]), font,  
                   fontScale, color, thickness, cv2.LINE_AA) #label the boxes
plt.figure(figsize=[20,20]) #plot my figure
plt.imshow(image[:,:,::-1]) #don't know
plt.show()


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)

