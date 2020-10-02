#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from os.path import join as op
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageStat
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


PATH = '../input/'
df_train = pd.read_csv(PATH + 'train.csv')


# # RGB or BGR

# We can know if a photo is in gray if the three color channels are identical.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef grey_cv(row, dataset):\n    filename = op(PATH,dataset,row['Image'])\n    img = cv2.imread(filename)\n    if (img[:,:,0] == img[:,:,1]).all():\n        return img.shape[0], img.shape[1], True\n    else:\n        return img.shape[0], img.shape[1], False\n    \ndf_train['h'], df_train['w'], df_train['gray'] = zip(*df_train.apply(lambda row: grey_cv(row, 'train'), axis=1))")


# Get mean intensity for each channel RGB

# In[ ]:


def get_rgb_men(row):
    img = cv2.imread(PATH + 'train/' + row['Image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.sum(img[:,:,0]), np.sum(img[:,:,1]), np.sum(img[:,:,2])

df_train['R'], df_train['G'], df_train['B'] = zip(*df_train.apply(lambda row: get_rgb_men(row), axis=1) )


# ## Most red images

# In[ ]:


df = df_train[(df_train['gray'] == False) & (df_train['B'] < df_train['R']) & (df_train['G'] < df_train['R'])]
plt.figure(figsize=(15,15))
for i,(_,row) in enumerate(df.iloc[:9].iterrows()):
    plt.subplot(3,3,i+1)
    img = cv2.imread(PATH + 'train/' + row['Image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 
    plt.axis('off')


# In[ ]:


num_photos = 6
fig, axr = plt.subplots(num_photos,2,figsize=(15,15))
for i,(_,row) in enumerate(df.iloc[:num_photos].iterrows()):
    img = cv2.imread(PATH + 'train/' + row['Image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axr[i,0].imshow(img)
    axr[i,0].axis('off')
    axr[i,1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 
    x, y = np.histogram(img[:,:,0], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='R', alpha=0.8, color='C0')
    x, y = np.histogram(img[:,:,1], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='G', alpha=0.8, color='C5')
    x, y = np.histogram(img[:,:,2], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='B', alpha=0.8, color='C1')
    axr[i,1].legend()
    axr[i,1].axis('off')


# Some whales have yellow spots and some images are redish, maybe due to sunset

# ## Most blue images

# In[ ]:


df = df_train[(df_train['gray'] == False) & (df_train['B'] > df_train['R']) & (df_train['B'] > df_train['G'])]
plt.figure(figsize=(15,15))
for i,(_,row) in enumerate(df.iloc[:9].iterrows()):
    plt.subplot(3,3,i+1)
    img = cv2.imread(PATH + 'train/' + row['Image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 
    plt.axis('off')


# In[ ]:


num_photos = 6
fig, axr = plt.subplots(num_photos,2,figsize=(15,15))
for i,(_,row) in enumerate(df.iloc[:num_photos].iterrows()):
    #plt.subplot(2,2,i+1)
    img = cv2.imread(PATH + 'train/' + row['Image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axr[i,0].imshow(img)
    axr[i,0].axis('off')
    axr[i,1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 
    x, y = np.histogram(img[:,:,0], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='R', alpha=0.8, color='C0')
    x, y = np.histogram(img[:,:,1], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='G', alpha=0.8, color='C5')
    x, y = np.histogram(img[:,:,2], bins=255, normed=True)
    axr[i,1].bar(y[:-1], x, label='B', alpha=0.8, color='C1')
    axr[i,1].legend()
    axr[i,1].axis('off')


# Some images are rotated and refill with white background, which explains the high peak of the histogramm.

# # Contours ?

# In[ ]:


from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch


# In[ ]:


img = cv2.imread(PATH + 'train/' + df_train.iloc[:1]['Image'].values[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#bilateral = cv2.bilateralFilter(gray, 5, 5,5)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
norm_hist = clahe.apply(gray)
blurred = cv2.GaussianBlur(norm_hist, (13, 13), 0)
#eq = cv2.equalizeHist(bilateral)
edged = cv2.Canny(blurred, 0, 150)


# In[ ]:


plt.figure(figsize=(14,10))
plt.imshow(edged)
plt.show()


# In[ ]:


_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
h, w = img.shape[:2]
thresh_area = 0.001
list_contours = list()
for c in contours:
    area = cv2.contourArea(c)

    if (area > thresh_area*h*w): 
        #rect_page = cv2.minAreaRect(c)
        #box_page = np.int0(cv2.boxPoints(rect_page))
        list_contours.append(c)


# In[ ]:


plt.figure(figsize=(10,10))
axr = plt.axes()
axr.imshow(img)
colors = ["lightblue", "lightgreen", "coral", "cyan", "magenta", "yellow", "black","darkblue", "darkgreen", "darkred", "gold", "chocolate", "blue", "green", "red"]
for i, cnt in enumerate(list_contours):
    polygonA = Polygon([(x[0][0],x[0][1]) for x in  cnt])
    patch = PolygonPatch(polygonA, fc='none', ec=colors[i%13], lw=3)
    axr.add_patch(patch)


# Still work to do in order to use the contour ..
