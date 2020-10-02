#!/usr/bin/env python
# coding: utf-8

# Show how to draw bbox and crop them as train datas.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import glob 
import matplotlib.pyplot as plt
import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.listdir('/kaggle/input/iwildcam-2020-fgvc7')


# # EDA

# In[ ]:


test=pd.read_csv('../input/iwildcam-2020-fgvc7/sample_submission.csv')
test.shape                  


# In[ ]:


train_jpeg = glob.glob('../input/iwildcam-2020-fgvc7/train/*')
test_jpeg = glob.glob('../input/iwildcam-2020-fgvc7/test/*')

print("number of train jpeg data:", len(train_jpeg))
print("number of test jpeg data:", len(test_jpeg))
'''
number of train jpeg data: 217959
number of test jpeg data: 62894

'''


# In[ ]:


fig = plt.figure(figsize=(25, 16))
for i,im_path in enumerate(train_jpeg[:16]):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    im = Image.open(im_path)
    im = im.resize((480,270))
    plt.imshow(im)


# # Draw bbox and Save

# ## One sample

# * This picture has over 3 bboxes. We will choose three bboxes which conf are greater than 0.9.

# {'detections': [{'category': '1', 'bbox': [0.0, 0.4669, 0.1853, 0.4238], 'conf': 1.0}, {'category': '1', 'bbox': [0.2406, 0.4672, 0.0309, 0.1105], 'conf': 0.998}, {'category': '1', 'bbox': [0.5058, 0.4577, 0.06, 0.1043], 'conf': 0.911}, {'category': '1', 'bbox': [0.9902, 0.4283, 0.0098, 0.0487], 'conf': 0.697}, {'category': '1', 'bbox': [0.9956, 0.4284, 0.0044, 0.049], 'conf': 0.505}, {'category': '1', 'bbox': [0.9974, 0.4293, 0.0026, 0.0481], 'conf': 0.505}, {'category': '1', 'bbox': [0.5078, 0.4574, 0.046, 0.0656], 'conf': 0.316}], 'id': '905a4416-21bc-11ea-a13a-137349068a90', 'max_detection_conf': 1.0}

# In[ ]:


im = Image.open("../input/iwildcam-2020-fgvc7/train/905a4416-21bc-11ea-a13a-137349068a90.jpg")

plt.imshow(im)
print(im.size)


# In[ ]:


with open('../input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json', encoding='utf-8') as fin:
    train_df=json.load(fin)


# In[ ]:


# for i,item in enumerate(train_df['images']):
#     if i>20:break
#     print('*'*50)
#     print(item)
        


# ## Draw one bbox

# In[ ]:


box=[0.0, 0.4669, 0.1853, 0.4238]


# In[ ]:


from PIL import Image, ImageDraw


x1, y1,w_box, h_box = box
ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box


draw = ImageDraw.Draw(im)
imageWidth=im.size[0]
imageHeight= im.size[1]
(left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
print(left, right, top, bottom)

draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=4, fill='Red')


# In[ ]:


# plt.imshow(crop_img)
im


# ## Crop and save the bbox image

# In[ ]:


#Image.crop(left, up, right, below)
crop_shape=(left,top , right, bottom)
crop_img = im.crop(crop_shape)
crop_img=crop_img.resize((299,299))
iImage = im.format
crop_img.save('dogs1.jpg'.format(iImage))
crop_img


# ## Draw the second bbox

# In[ ]:


box=[0.2406, 0.4672, 0.0309, 0.1105]


# In[ ]:


x1, y1,w_box, h_box = box
ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box


(left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
print(left, right, top, bottom)

draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=4, fill='Red')


# In[ ]:


im


# In[ ]:


#Image.crop(left, up, right, below)
crop_shape=(left,top , right, bottom)
crop_img = im.crop(crop_shape)
crop_img=crop_img.resize((299,299))
iImage = im.format
crop_img.save('dogs2.jpg'.format(iImage))
crop_img


# ## Draw the third bbox

# In[ ]:


box=[0.5058, 0.4577, 0.06, 0.1043]
x1, y1,w_box, h_box = box
ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box


(left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
print(left, right, top, bottom)

draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=4, fill='Red')


# In[ ]:


im


# In[ ]:


#Image.crop(left, up, right, below)
crop_shape=(left,top , right, bottom)
crop_img = im.crop(crop_shape)
crop_img=crop_img.resize((299,299))
iImage = im.format
crop_img.save('dogs3.jpg'.format(iImage))
crop_img


# In[ ]:


print('Done!')

