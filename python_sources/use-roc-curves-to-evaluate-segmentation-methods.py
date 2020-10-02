#!/usr/bin/env python
# coding: utf-8

# Here is the starting point for processing EM microscopy segmentation data from Kaggle

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from skimage.io import imread
import matplotlib.pyplot as plt


# # Load and Preview Images

# In[ ]:


em_image_vol = imread('../input/training.tif')[:21]
em_thresh_vol = imread('../input/training_groundtruth.tif')[:21]>0
print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
em_slice=em_image_vol[20]
em_slice_mask=em_thresh_vol[20]
fig, (ax1, ax2 ,ax3) = plt.subplots(1,3, figsize = (14,4))
ax1.imshow(em_slice,cmap='gray')
ax1.set_title('EM Image')
ax2.hist(em_image_vol.ravel())
ax2.set_title('EM Intensity Distribution')
ax3.imshow(em_slice_mask, cmap = 'bone')
ax3.set_title('Segmented')


# ## Just focus on the nerve
# Show the segmentation of the nerve and the intensity inside that region

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (10,10))
ax1.imshow(em_slice,cmap='gray')
ax1.set_title('EM Image')
counts, bins, _ = ax2.hist(em_image_vol.ravel())
ax2.set_title('EM Intensity Distribution')
ax3.imshow(em_slice*em_slice_mask,cmap='gray')
ax3.set_title('Segmented Nerve Image')
ax4.hist(em_image_vol[em_thresh_vol], bins = bins) # use the same bins again
ax4.set_title('EM Masked Intensity Distribution')


# # Simple Segmentation
# We see too peaks in this histogram, lets try segmenting between them

# In[ ]:


fig, (ax1, ax2 ,ax3) = plt.subplots(1,3, figsize = (14,4))
ax1.imshow(em_slice,cmap='gray')
ax1.set_title('EM Image')
ax2.imshow((em_slice>65) & (em_slice<150),cmap='bone')
ax2.set_title('Test Segmentation')
ax3.imshow(em_slice_mask, cmap = 'bone')
ax3.set_title('Segmented Nerve')


# # Scoring the Result
# We can use the ROC curve to score the result and see how accurately we classified the images

# In[ ]:


from sklearn.metrics import roc_curve, auc # roc curve tools
ground_truth_labels = em_thresh_vol.ravel() # we want to make them into vectors
score_value = 1-em_image_vol.ravel()/255.0 # we want to make them into vectors
fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
roc_auc = auc(fpr,tpr)


# In[ ]:


fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right")


# # Gaussian Filter
# Here we apply a gaussian filter and see how the performance changes

# In[ ]:


from skimage.filters import gaussian
gus_image = gaussian(em_image_vol/255,sigma = 4)*255
fig, (ax1, ax2 ,ax3) = plt.subplots(1,3, figsize = (14,4))
ax1.imshow(gus_image[20])
ax1.set_title('Gaussian Filtered Image')
ax2.imshow((gus_image[20]>65) & (gus_image[20]<150),cmap='bone')
ax2.set_title('Test Segmentation')
ax3.imshow(em_slice_mask, cmap = 'bone')
ax3.set_title('Segmented Nerve')


# In[ ]:


ground_truth_labels = em_thresh_vol.ravel() # we want to make them into vectors
score_value = 1-gus_image.ravel()/255.0 # we want to make them into vectors
fpr_gus, tpr_gus, _ = roc_curve(ground_truth_labels,score_value)
roc_auc_gus = auc(fpr_gus,tpr_gus)
fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='Raw ROC curve (area = %0.2f)' % roc_auc)
ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right")


# ## Scoring from 110
# Instead of using the inverse value as a score we can rewrite it as 
# $$
# 1-|110-x|
# $$
# So it focuses on the values closer to 110 rather than linear with the scale

# In[ ]:


ground_truth_labels = em_thresh_vol.ravel() # we want to make them into vectors
score_value = 1-np.abs(110-gus_image.ravel())/255.0 # we want to make them into vectors
fpr_gus2, tpr_gus2, _ = roc_curve(ground_truth_labels,score_value)
roc_auc_gus2 = auc(fpr_gus2,tpr_gus2)
fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='Raw ROC curve (area = %0.2f)' % roc_auc)
ax.plot(fpr_gus, tpr_gus, label='Gaussian ROC curve (area = %0.2f)' % roc_auc_gus)
ax.plot(fpr_gus2, tpr_gus2, label='Gaussian Alt ROC curve (area = %0.2f)' % roc_auc_gus)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right")


# # Tasks 
# 
#  1. How can you improve filtering in this analysis?
#   - Which filter elements might improve the area under the ROC?
#  3. Try making workflows to test out a few different filters
#  - Where might morphological operations fit in?
#  - How can you make them part of this workflow as well?
#  6. (Challenge) Try and use the optimize toolbox of scipy with the fmin
#     function (from scipy.optimize import fmin) to find the optimum
#     parameters for the highers area (hint: fmin finds the minimum value)

# In[ ]:




