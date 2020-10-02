#!/usr/bin/env python
# coding: utf-8

# I was very impressed with the beautiful painting titled *Traveling Salesman* by [Julian Lethbridge](http://www.ulae.com/artists/JulianLethbridge/), which uses two different texture for inside and outside regions created by a tour.
# So, I started making an image using a tour of this competition. I am not good at painting and other artworks. But we can make an art using *Neural Style Transfer*.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# # Original Image
# 
# According to the link in top page of this competition, the cities are made from the photo by [Norman Tsui](https://unsplash.com/photos/KBKHXjhVQVM).
# 
# <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@arainbowman?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Norman Tsui"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-1px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M20.8 18.1c0 2.7-2.2 4.8-4.8 4.8s-4.8-2.1-4.8-4.8c0-2.7 2.2-4.8 4.8-4.8 2.7.1 4.8 2.2 4.8 4.8zm11.2-7.4v14.9c0 2.3-1.9 4.3-4.3 4.3h-23.4c-2.4 0-4.3-1.9-4.3-4.3v-15c0-2.3 1.9-4.3 4.3-4.3h3.7l.8-2.3c.4-1.1 1.7-2 2.9-2h8.6c1.2 0 2.5.9 2.9 2l.8 2.4h3.7c2.4 0 4.3 1.9 4.3 4.3zm-8.6 7.5c0-4.1-3.3-7.5-7.5-7.5-4.1 0-7.5 3.4-7.5 7.5s3.3 7.5 7.5 7.5c4.2-.1 7.5-3.4 7.5-7.5z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Norman Tsui</span></a>
# 
# So, we will download the image to use it as base content image.

# In[ ]:


get_ipython().system('wget https://unsplash.com/photos/KBKHXjhVQVM/download?force=true -O reindeer.jpg')


# In[ ]:


img = Image.open('reindeer.jpg')
img = img.resize((int(img.size[0] * 0.6), int(img.size[1] * 0.6)), Image.BICUBIC)
img.save('reindeer.jpg')
img


# # Get a good tour
# 
# We can get a good tour using  **Lin-Kernighan** algorithm implemented in Concorde.
# 
# To reduce complications of installing Concorde and QSopt, we will install pyconcorde and build it using it.

# In[ ]:


get_ipython().system('git clone https://github.com/jvkersch/pyconcorde')
get_ipython().system('pip install -e ./pyconcorde')


# Write a problem file

# In[ ]:


cities = pd.read_csv('../input/cities.csv')
xy_int = (cities[['X', 'Y']] * 1000).astype(np.int64)
with open('xy_int.csv', 'w') as fp:
    print(len(xy_int), file=fp)
    print(xy_int.to_csv(index=False, header=False, sep=' '), file=fp)


# Execute Lin-Kernighan algorithm

# In[ ]:


get_ipython().system('./pyconcorde/build/concorde/LINKERN/linkern -s 1 -o lk.sol -N 2 xy_int.csv > /dev/null')


# Plot the obtained tour

# In[ ]:


order = []
with open('lk.sol', 'r') as fp:
    lines = fp.readlines()
order = [int(v.split(' ')[0]) for v in lines[1:]] + [0]


# In[ ]:


plt.figure(figsize=(15, 10))
xy = cities.loc[order, ['X', 'Y']].values
plt.plot(xy[:, 0], xy[:, 1], lw=1., ms=10, c='black')
plt.axis('equal')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())


# # Stylize The Original Image
# 
# Neural Style Transfer is a method for creating art with deep learning. Originally, it is proposed by [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), and it is extended by many works.  We will use a fast variant of these methods usually called Fast Style Transfer proposed by [Johnson et al.](https://arxiv.org/abs/1603.08155)
# 
# We will use [yusuketomoto/chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) which is an implementation by chainer.

# In[ ]:


# download a fork of the repository to use recent chainer
get_ipython().system('git clone -b use-new-api https://github.com/zaburo-ch/chainer-fast-neuralstyle')


# In[ ]:


get_ipython().run_line_magic('run', 'chainer-fast-neuralstyle/generate.py reindeer.jpg -m chainer-fast-neuralstyle/models/composition.model -o composition.png')


# In[ ]:


get_ipython().run_line_magic('run', 'chainer-fast-neuralstyle/generate.py reindeer.jpg -m chainer-fast-neuralstyle/models/seurat.model -o seurat.png')


# # Merge two stylized figures by tour mask
# 
# A good tour of 2D TSP has no crossing edges, because we can get cheaper tour by changing it into uncrossing edges. Therefore, the tour divides the region into inside and outside of it.
# We will merge the images made in above using one for inside and the other for outside.

# Made a mask using the tour

# In[ ]:


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
xy = cities.loc[order, ['X', 'Y']].values
poly = plt.Polygon(xy, fc='black')
ax.add_patch(poly)
plt.axis('equal')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('mask.png', bbox_inches='tight', pad_inches=0, dpi=150)


# In[ ]:


img1 = Image.open(f'composition.png')
mask = Image.open('mask.png').convert('L')
mask = np.asarray(mask.resize(img1.size, Image.BICUBIC)) >= 256 // 2
img1 = np.asarray(img1).copy()
img2 = Image.open(f'seurat.png')
img1[mask] = np.asarray(img2)[mask]


# # Result

# In[ ]:


Image.fromarray(img1)


# # Clean up
# Remove directories for the limit of path depth in kaggle kernel.

# In[ ]:


get_ipython().system('rm -rf chainer-fast-neuralstyle')
get_ipython().system('rm -rf pyconcorde')


# Have your best Christmas ever!
