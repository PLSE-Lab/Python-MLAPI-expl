#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt
import urllib
from tqdm.notebook import tqdm
import re
import math


# In[ ]:


pwd


# In[ ]:


cd /kaggle/input/


# In[ ]:


cp-r kerasretinanet /kaggle/working/


# In[ ]:


cd /kaggle/working/kerasretinanet


# In[ ]:


get_ipython().system('pip install /kaggle/input/wh1files/keras_resnet-0.1.0-py2.py3-none-any.whl')


# In[ ]:


data_dir = '/kaggle/input/global-wheat-detection/'
train_path = data_dir + '/train/'
test_path = data_dir + '/test/'


# In[ ]:





# In[ ]:


os.listdir(data_dir)


# In[ ]:


cd /kaggle/working/kerasretinanet


# In[ ]:


get_ipython().system('ls snapshots')


# In[ ]:


from keras_retinanet import models


# In[ ]:


model_path = '/kaggle/input/trainedmodel/keras-retinanet/snapshots/resnet50_csv_03.h5'








# In[ ]:


get_ipython().system('python setup.py build_ext --inplace')


# In[ ]:


pwd


# In[ ]:


model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)


# In[ ]:


model_path


# In[ ]:





# In[ ]:


li=os.listdir(test_path)
li[:5]


# In[ ]:


def predict(image):
    image = preprocess_image(image.copy())
    #image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

    #boxes /= scale

    return boxes, scores, labels


# In[ ]:


THRES_SCORE = 0.3

def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)


# In[ ]:


def show_detected_objects(image_name, boxes, scores, labels):
    img_path = test_path+'/'+image_name
  
    image = read_image_bgr(img_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_detections(draw, boxes, scores, labels)
    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# In[ ]:


samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
imgs = samsub['image_id'].values


# In[ ]:


import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


# In[ ]:


pred_string=[]
for img in imgs:
    preds=''
    img_name=img+'.jpg'
    img_path = test_path+'/'+img_name
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict(image)
    show_detected_objects(img_name, boxes, scores, labels)
    boxes=boxes[0]
    scores=scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx]>THRES_SCORE:
            box,score=boxes[idx],scores[idx]
            preds+="{:0.2f} {} {} {} {} ".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]))
    pred_string.append(preds)


# In[ ]:


sub={"image_id":imgs, "PredictionString":pred_string}
sub=pd.DataFrame(sub)
sub.head(10)


# In[ ]:


sub.to_csv('/kaggle/working/submission.csv',index=False)


# In[ ]:





# In[ ]:




