#!/usr/bin/env python
# coding: utf-8

# The reference is taken from following:<br>
# [Deep Learning based Object Detection using YOLOv3 with OpenCV ( Python / C++ )](https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/)<br>
# [YOLO object detection using Opencv with Python](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

# In[ ]:


import numpy as np 
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


# First need to configure yolov3 model with opencv.
# The *readNet* function from dnn module detects an original framwork of train model and calls automatically the function *readNetFromDarknet*.
# And *readNetFromDarknet* function returns the object that is ready to do forward, throw an exception in failure cases.
# 
# For *readNet* function order for passing the weights and cfg files doesn't matter.
# > So here *readNetFromDarknet* also can be used instead of *readNet*.
# But the reason for considering readNet is to make it generic. If thr trained model belongs to tensorflow *readNet* automatcally calls *readNetFromTensorflow*.
# 
# 

# In[ ]:


net = cv2.dnn.readNet("/kaggle/input/yolov3-weight/yolov3.weights", "/kaggle/input/yolov3-weight/yolov3.cfg")


# Let's have a look at different layers using *getLayerNames*. 

# In[ ]:


layer_names = net.getLayerNames()
print("layers names:")
print(layer_names)


# * conv - convolution layer<br>
#   Convolution layer applies a filter to an input to create a feature map
# * bn - batch normalization layer<br>
#  This normalize the input for the hidden layer and also helps to reduce the training time, to reduce the effect of covariate shift and also add regularization effect.
# * relu - relu activation layer
# * shortcut - skip connection or residual connection<br>
# This helps to improve the accuracy for a large neural network which tends to reduce the accuracy because of vanishing gradients as the network grows.
# * Permute - Permute layer<br>
# This is used to re-order the dimention of the input according to the given pattern.
# * identity - This layer maps the output of unconnected layer to next input layer. [yolo_84(unconnected layer) --> conv_84]
# * upsample - Convolution layer performs downsampling by filtering input genarate the output of a smaller shape compare to input. Upsample layer performs the reverse opration by repeating rows and columns of input.
# * concat - This merges s list of inputs.
# * yolo - This is an output layer which a list of bounding boxes along with the recognised classes.

# Let's idetify output layers using a function *getUnconnectedOutLayersNames*.

# In[ ]:


output_layers = net.getUnconnectedOutLayersNames()
print("output layers:")
print(output_layers)


# YoloV3 is trained to indetify 80 different types of objects.
# Let's fetch this detail from coco names.

# In[ ]:


classes = []
with open("/kaggle/input/coconames/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
colors = np.random.uniform(0, 255, size=(len(classes), 3)) #This will be used later to assign colors for the bounding box for the detected objects


# The network requires the image is blob format.<br>
# Blob - Binary Large Objects.<br>
# Blob represents the group of pixels having simmilar values and different from surrounding pixels.<br>
# The function blobFromImage convets the image in blob.<br>
# We can scale, resize , subtract the mean from each pixels, change the order of the channels from BGR to RGB using swapRB argument and also crop the image.<br>
# With the method setInput, the blob of an image is set as input for the network.<br>
# The forward method propragate the blob of an image through the network and return the predictions.

# In[ ]:


def get_objects_predictions(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor = 1/255, size = (416, 416), mean= (0, 0, 0), swapRB = True, crop=False)
    net.setInput(blob)
    predictions = net.forward(output_layers)
    return predictions,height, width


# The first 4 elements represent the center_x, center_y, width and height. The fifth element represents the confidence that the bounding box encloses an object.<br>
# The rest of the elements are the confidence associated with each class (i.e. object type). The box is assigned to the class corresponding to the highest score for the box.<br>
# The highest score for a box is also called its confidence.(here the confidence is set as 0.5). If the confidence of a box is less than the given threshold, the bounding box is dropped and not considered for further processing.

# In[ ]:


def get_box_dimentions(predictions,height, width, confThreshold = 0.5):
    class_ids = []
    confidences = []
    boxes = []
    for out in predictions:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)#Identifing the class type of the detected object by checking maximum confidence
            confidence = scores[class_id]
            if confidence > confThreshold:
                # Object detected
                center_x = int(detection[0] * width) #converting center_x with respect to original image size
                center_y = int(detection[1] * height)#converting center_y with respect to original image size
                w = int(detection[2] * width)#converting width with respect to original image size
                h = int(detection[3] * height)#converting height with respect to original image size
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes,confidences,class_ids


# The Non max suppression technique is used to ensure that the obeject is detected only once.<br>
# In this the bounding box with probability more nmsThresold is considered, other bounding boxes will be dropped out.

# In[ ]:


def non_max_suppression(boxes,confidences,confThreshold = 0.5, nmsThreshold = 0.4):
    return cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)


# Let's draw the bounding boxes.

# In[ ]:


def draw_bouding_boxes(img,boxes,confidences,class_ids,nms_indexes,colors):
    for i in range(len(boxes)):
        if i in nms_indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + ' :' + str(int(confidences[i]*100)) + '%'
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y - 15),cv2.FONT_HERSHEY_PLAIN ,2, color, 3)
    return img


# In[ ]:


def detect_objects(img_path):
    predictions,height, width = get_objects_predictions(img_path)
    boxes,confidences,class_ids = get_box_dimentions(predictions,height, width)
    nms_indexes = non_max_suppression(boxes,confidences)
    img = draw_bouding_boxes(img_path,boxes,confidences,class_ids,nms_indexes,colors)
    return img


# In[ ]:


files = ['/kaggle/input/open-images-2019-object-detection/test/' + i for i in os.listdir('/kaggle/input/open-images-2019-object-detection/test')]


# Let's have a look at some images with object detection.

# In[ ]:


plt.figure(figsize=(25,30))

for i in range(1,13):
    index = np.random.randint(len(files))
    plt.subplot(6, 2, i)
    plt.imshow(detect_objects(cv2.imread(files[index])), cmap='cool')
plt.show()

