#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook uses the [CONX package](https://github.com/Calysto/conx) from [Douglas Blank](https://cs.brynmawr.edu/~dblank/) and the Calysto team to show how the basic U-Net architecture works on segmentation problems. We use segmenting lungs on Chest X-rays as a good use-case for U-net segmentation since there are not easily derivable rules for automatically segmenting them. The kernel/notebook is best run interactively (fork it and run it a in an interactive Kaggle Session or download the IPython notebook and run it using mybinder on 
# [![imagetool](https://img.shields.io/badge/launch-UNET_Demo-yellow.svg)](http://mybinder.org/v2/gh/Quantitative-Big-Imaging/conx/master?urlpath=%2Fapps%2Fseg_notebooks%2FUNetDemo.ipynb)
# 
# ## Note
# This is not 'real' U-Net since it does not have the proper cropping layers nor the correct size and depth (Ronneberger trained the original model using 512x512 images and having many more layers of max-pooling and upsampling).  The cropping layers are important as well since edges can skew the weights in the convolutions and cause the algorithm to converge slowly or with small enough windows incorrectly.

# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
cxr_paths = glob(os.path.join('..', 'input', 'Montgomery', 'MontgomerySet', '*', '*.png'))
cxr_images = [(c_path, 
               [os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','leftMask', os.path.basename(c_path)),
               os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','rightMask', os.path.basename(c_path))]
              ) for c_path in cxr_paths]
print('CXR Images', len(cxr_paths), cxr_paths[0])
print(cxr_images[0])


# # Loading Training Data
# Here we load the images from the [Tuberculosis CXR Dataset](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities)

# In[51]:


from skimage.io import imread as imread_raw
from skimage.transform import resize
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning, module='skimage') # skimage is really annoying
img_size = (96, 96)
def imread(in_path):
    i_img = imread_raw(in_path)
    return np.expand_dims(resize(i_img, img_size, order = 1), -1)

img_vol, seg_vol = [], []
for img_path, s_paths in tqdm(cxr_images):
    img_vol += [imread(img_path)]    
    seg_vol += [np.max(np.stack([imread(s_path) for s_path in s_paths],0),0)]
img_vol = np.stack(img_vol,0)
seg_vol = np.stack(seg_vol,0)


# In[33]:


np.random.seed(2018)
t_img, m_img = img_vol[0], seg_vol[0]
fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')


# In[34]:


import conx as cx


# In[35]:


net = cx.Network("MiniUNet")
base_depth = 16


# In[36]:


net.add(cx.ImageLayer("input", img_size, t_img.shape[-1])) 
net.add(cx.BatchNormalizationLayer("bnorm"))
c2 = lambda i, j, act = "relu": cx.Conv2DLayer("conv_{}".format(i, j), j, (3, 3), padding='same', activation=act)
net.add(c2(0, base_depth))
net.add(c2(1, base_depth))
net.add(cx.MaxPool2DLayer("pool1", pool_size=(2, 2), dropout=0.25))
net.add(c2(2, 2*base_depth))
net.add(c2(3, 2*base_depth))
net.add(cx.MaxPool2DLayer("pool2", pool_size=(2, 2), dropout=0.25))
net.add(c2(4, 4*base_depth))
net.add(c2(5, 4*base_depth))
net.add(cx.UpSampling2DLayer("up2", size = (2,2)))
net.add(cx.ConcatenateLayer("cat2"))
net.add(c2(6, 2*base_depth))
net.add(c2(7, 2*base_depth))
net.add(cx.UpSampling2DLayer("up1", size = (2,2)))
net.add(cx.ConcatenateLayer("cat1"))
net.add(c2(8, 2*base_depth))
net.add(cx.Conv2DLayer("output", 1, (1, 1), padding='same', activation='sigmoid'));


# # Connections
# We have to connect all of the layers together in a U-Net style. The tricky part is the skip connections that skip over the max pooling layers and go directly to the concatenate to combine the higher resolution information with the lower resolution feature space

# In[37]:


net.connect('input', 'bnorm')
net.connect('bnorm', 'conv_0')
net.connect('bnorm', 'cat1')
net.connect('conv_0', 'conv_1')


# In[38]:


net.connect('conv_1', 'pool1')
net.connect('pool1', 'conv_2')
net.connect('conv_2', 'conv_3')
net.connect('conv_3', 'pool2')
net.connect('pool2', 'conv_4')
net.connect('conv_4', 'conv_5')
net.connect('conv_5', 'up2')
net.connect('up2', 'cat2')
net.connect('conv_3', 'cat2')
net.connect('cat2', 'conv_6')
net.connect('conv_6', 'conv_7')
net.connect('conv_7', 'up1')
net.connect('up1', 'cat1')
net.connect('cat1', 'conv_8')
net.connect('conv_8', 'output')


# In[39]:


net.compile(error="binary_crossentropy", optimizer="adam")


# In[40]:


net.picture(t_img, dynamic = True, rotate = True, show_targets = True, show_errors=True, scale = 1.0)


# In[41]:


net.dataset.clear()
ip_pairs = [(x,y) for x,y in zip(img_vol, seg_vol)]
net.dataset.append(ip_pairs)
net.dataset.split(0.25)


# In[42]:


net.propagate_to_image("conv_5", t_img, scale = 1)


# In[43]:


net.train(epochs=20, record=True)


# In[44]:


net.propagate_to_image("conv_5", t_img)


# In[46]:


net.picture(t_img, dynamic = True, rotate = True, show_targets = True, show_errors=True, scale = 1)


# In[47]:


net.dashboard()


# In[48]:


net.movie(lambda net, epoch: net.propagate_to_image("conv_5", t_img, scale = 3), 
                'mid_conv.gif', mp4 = False)


# In[49]:


net.movie(lambda net, epoch: net.propagate_to_image("conv_8", t_img, scale = 3), 
                'hr_conv.gif', mp4 = False)


# In[50]:


net.movie(lambda net, epoch: net.propagate_to_image("output", t_img, scale = 3), 
                'output.gif', mp4 = False)


# In[ ]:




