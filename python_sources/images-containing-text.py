#!/usr/bin/env python
# coding: utf-8

# In the pursuit of a nice juicy "leakage" I give you a simple and straightforward text detection facility using plain OpenCV (EAST text detection caffe model).It manages to automatically locate text in both train and test images. The majority of the code was taken from the great Pyimagesearh [OpenCV Text Detection (EAST text detector)](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/). No character/word recognition yet. I believe there must be a correlation between text appearance and `new_whale` tag.
# 
# To reduce false positives we restrict our search to the the bottom area of each image (see `offsetY/H < 0.80` below)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import cv2
import os

from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from glob import glob
import multiprocessing

import os
 
# Any results you write to the current directory are saved as output.


# In[ ]:


whale = pd.read_csv('../input/humpback-whale-identification/train.csv')
whale.head()
print(len(whale))
len(np.unique(whale.Id))


# In[ ]:


unknown_whale = whale[whale.Id=='new_whale']
unknown_whale.head()


# In[ ]:


train_path = '../input/humpback-whale-identification/train/'
train_images = unknown_whale.Image.values#os.listdir(train_path)
test_path = '../input/humpback-whale-identification/test/'


# In[ ]:


whale_dict = dict(zip(whale.Image, whale.Id))


# In[ ]:


layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]


# In[ ]:


get_ipython().system('pip install imutils')


# The architecture of EAST text detector is depicted in the image below. Among the outputs is a set of text  boxes. We loop through this test for increasing `y` values and we can get an estimate of the number of lines.  
# ![image](https://www.pyimagesearch.com/wp-content/uploads/2018/08/opencv_text_detection_east.jpg)
# 

# In[ ]:


import time
from imutils.object_detection import non_max_suppression
net = cv2.dnn.readNet('../input/frozen-east-text-detection/frozen_east_text_detection.pb')


# In[ ]:


WW = 320
HH = 160
def get_images_with_text(path, with_class=True, WW=320, HH=160):

    image_files = os.listdir(path)
    FOUND  = []
    new_whale_count = 0
    for image_file in tqdm_notebook(image_files):

        # load the input image and grab the image dimensions
        image = cv2.imread(path + image_file)
        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (WW, HH)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]


        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=False, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction


        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        text_found = 0
        text_lines = 0
        # loop over the number of rows
        for y in range(0, numRows):

            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            found = False
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.8:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                if offsetY/H < 0.80:
                    continue


                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
                found = True

            if found == True:
                text_lines += 1


        boxes = non_max_suppression(np.array(rects), probs=confidences)
        #if len(boxes)>0:
        #    print (image_file, text_lines)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

        if len(boxes) > 0:
            if with_class==True:
                FOUND.append([image_file, whale_dict[image_file], text_lines])
                if whale_dict[image_file] == 'new_whale':
                    new_whale_count = new_whale_count + 1
            else:
                FOUND.append([image_file, text_lines])
    return FOUND


# In[ ]:


df_train = get_images_with_text(train_path)
df_test = get_images_with_text(test_path, with_class=False)


# In[ ]:


df_train= pd.DataFrame(df_train)
df_train.columns = ['image', 'class', 'line_count']
df_train.head()


# In[ ]:


df_test= pd.DataFrame(df_test)
df_test.columns = ['image', 'line_count']
df_test.head()


# In[ ]:


df_train.to_csv('train_text.csv')
df_test.to_csv('train_text.csv')


# Let us plot some training images along with class annotations and number of lines.

# In[ ]:


fig, axes = plt.subplots(5, 5)
 
fig.set_figwidth(20)
fig.set_figheight(20)

for i, row in df_train.iterrows():
    if i >= 25:
        break
    img = cv2.imread(train_path + row['image'])
    axes[int(i/5), i%5].imshow(img)
    axes[int(i/5), i%5].set_title(row['image']  + '-' + str(whale_dict[row['image']]) + ' (' + str(row['line_count']) + ')')
    axes[int(i/5), i%5].axis('off')

plt.show()


# Let us plot some tet images along with the  number of lines. It seems that text detector is producing more false positives.

# In[ ]:


fig, axes = plt.subplots(5, 5)
 
fig.set_figwidth(20)
fig.set_figheight(20)

for i, row in df_test.iterrows():
    if i >= 25:
        break
    img = cv2.imread(test_path + row['image'])
    axes[int(i/5), i%5].imshow(img)
    axes[int(i/5), i%5].set_title( row['image']  + '-' +  ' (' + str(row['line_count']) + ')')
    axes[int(i/5), i%5].axis('off')

plt.show()


# ## Some observations:
# ### 1. By looking at the training images there seems to be great correlation between images with many lines of text  and identified whales

# In[ ]:


df_train[df_train.line_count>3]


# ### 2. Text that starts with '#' followed by a four digit number is most probably new_whale (see  #3332, #0518 in training images above) 

# In[ ]:




