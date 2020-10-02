#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import time


# In[ ]:


net = cv2.dnn.readNetFromDarknet("../input/helmet-detection-yolov3/yolov3-helmet.cfg","../input/helmet-detection-yolov3/yolov3-helmet.weights")
classes = []
with open("../input/helmet-detection-yolov3/helmet.names","r") as f:
   classes = [line.strip() for line in f.readlines()]


# In[ ]:


print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(outputlayers)


# In[ ]:




