#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from skimage.measure import label # for labeling regions
from skimage.measure import regionprops # for shape analysis
import numpy as np # for matrix operations and array support
from skimage.color import label2rgb # for making overlay plots
import matplotlib.patches as mpatches # for showing rectangles and annotations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread # for reading images
from skimage.feature import greycomatrix, greycoprops
grayco_prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']


# In[4]:


em_image_vol = imread('../input/training.tif')
em_thresh_vol = imread('../input/training_groundtruth.tif')
print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)


# # 2D Analysis of Textures

# In[5]:


em_idx = np.random.permutation(range(em_image_vol.shape[0]))[0]
em_slice = em_image_vol[em_idx]
em_thresh = em_thresh_vol[em_idx]
print("Slice Loaded, Dimensions", em_slice.shape)


# In[6]:


# show the slice and threshold
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (11, 5))
ax1.imshow(em_slice, cmap = 'gray')
ax1.axis('off')
ax1.set_title('Image')
ax2.imshow(em_thresh, cmap = 'gray')
ax2.axis('off')
ax2.set_title('Segmentation')
# here we mark the threshold on the original image

ax3.imshow(label2rgb(em_thresh,em_slice, bg_label=0))
ax3.axis('off')
ax3.set_title('Overlayed')


# # Tiling
# Here we divide the image up into unique tiles for further processing

# In[7]:


xx,yy = np.meshgrid(np.arange(em_slice.shape[1]),np.arange(em_slice.shape[0]))
region_labels=np.floor(xx/16)*64+np.floor(yy/16)
plt.matshow(region_labels,cmap='rainbow')


# # Calculating Texture
# Here we calculate the texture by using a tool called the gray level co-occurrence matrix which are part of the [features library in skimage][1]. We focus on two metrics in this, specifically dissimilarity and correlation which we calculate for each tile. We then want to see which of these parameters correlated best with belonging to a nerve fiber.
# 
# 
#   [1]: http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html

# In[24]:


# compute some GLCM properties each patch
from tqdm import tqdm
prop_imgs = {}
for c_prop in grayco_prop_list:
    prop_imgs[c_prop] = np.zeros_like(em_slice, dtype=np.float32)
score_img = np.zeros_like(em_slice, dtype=np.float32)
out_df_list = []
for patch_idx in tqdm(np.unique(region_labels)):
    xx_box, yy_box = np.where(region_labels==patch_idx)
    
    glcm = greycomatrix(em_slice[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()],
                        [5], [0], 256, symmetric=True, normed=True)
    
    mean_score = np.mean(em_thresh[region_labels == patch_idx])
    score_img[region_labels == patch_idx] = mean_score
    
    out_row = dict(
        intensity_mean=np.mean(em_slice[region_labels == patch_idx]),
        intensity_std=np.std(em_slice[region_labels == patch_idx]),
        score=mean_score)
    
    for c_prop in grayco_prop_list:
        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]
        prop_imgs[c_prop][region_labels == patch_idx] = out_row[c_prop]
        
    out_df_list += [out_row]


# In[25]:


# show the slice and threshold
fig, m_axs = plt.subplots(2, 4, figsize = (20, 10))
n_axs = m_axs.flatten()
ax1 = n_axs[0]
ax2 = n_axs[1]
ax1.imshow(em_slice, cmap = 'gray')
ax1.axis('off')
ax1.set_title('Image')
ax2.imshow(score_img,cmap='gray')
ax2.axis('off')
ax2.set_title('Regions')
for c_ax, c_prop in zip(n_axs[2:], grayco_prop_list):
    c_ax.imshow(prop_imgs[c_prop], cmap = 'gray')
    c_ax.axis('off')
    c_ax.set_title('{} Image'.format(c_prop))


# In[26]:


out_df=pd.DataFrame(out_df_list)
out_df['positive_score']=out_df['score'].map(lambda x: 'FG' if x>0 else 'BG')
out_df.sample(3)


# # Look for Correlations
# Here we look for relationships between the texture features and the original labeling

# In[27]:


import seaborn as sns
sns.pairplot(out_df,
             hue='positive_score',
             diag_kind="kde",
            diag_kws=dict(shade=True),
                        kind="reg")


# In[ ]:




