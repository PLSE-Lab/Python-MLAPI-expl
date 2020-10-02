#!/usr/bin/env python
# coding: utf-8

# # Overview
# We take images and turn them into features by performing a few simple analyses. We focus initially on colors and then on shapes

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'viridis' # grayscale looks better


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread as imread
from skimage.util import montage as montage2d
from skimage.color import label2rgb
from PIL import Image
base_dir = Path('..') / 'input'


# # Overview
# Load and process data

# In[ ]:


image_overview_df = pd.read_json(base_dir/ 'image_subset.json')
print(image_overview_df.shape[0], 'image, recipe pairs loaded')
image_overview_df.sample(3)


# ## Keep just a few columns

# In[ ]:


simple_images_df = image_overview_df[['id','image_path','title','top_ingredients']]
simple_images_df.head(3)


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            simple_images_df.head(9).iterrows()):
    c_ax.imshow(imread(base_dir / 'subset' / c_row['image_path']))
    c_ax.set_title('\n'.join(c_row['top_ingredients'][:4]))
    c_ax.axis('off')


# # Create Color Features
# We start with simple color features by grouping the image into 

# In[ ]:


test_row = simple_images_df.iloc[1]
print(test_row)


# ### Reduce the number of colors
# Currently we have $ \underbrace{2^8}_{\textrm{8-bit}}$ and $\underbrace{3 \textrm{channel}}_{\textrm{Red, Green, Blue}}$. This means we have $2^{8^3} \rightarrow 16,581,375$ different colors. 
# We can convert the image to 8-bit format to reduce the number of colors by a factor of 65536

# In[ ]:


test_image = Image.open(base_dir / 'subset' / test_row['image_path']) # normal image
# convert to 8bit color (animated GIF) and then back
web_image = test_image.convert('P', palette='WEB', dither=None)
few_color_image = web_image.convert('RGB')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_image)
ax2.imshow(few_color_image)


# In[ ]:


print('Unique colors before', len(set([tuple(rgb) for rgb in np.array(test_image).reshape((-1, 3))])))
print('Unique colors after', len(set([tuple(rgb) for rgb in np.array(few_color_image).reshape((-1, 3))])))


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
for c_channel, c_name in enumerate(['red', 'green', 'blue']):
    ax1.hist(np.array(test_image)[:, :, c_channel].ravel(), 
             color=c_name[0], 
             label=c_name, 
             bins=np.arange(256), 
             alpha=0.5)
    ax2.hist(np.array(few_color_image)[:, :, c_channel].ravel(), 
             color=c_name[0], 
             label=c_name, 
             bins=np.arange(256), 
             alpha=0.5)


# # How do the colors look?

# In[ ]:


idx_to_color = np.array(web_image.getpalette()).reshape((-1, 3))/255.0


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.imshow(few_color_image)
counts, bins = np.histogram(web_image, bins=np.arange(256))
for i in range(counts.shape[0]):
    ax2.bar(bins[i], counts[i], color=idx_to_color[i])
ax2.set_yscale('log')
ax2.set_xlabel('Color Id')
ax2.set_ylabel('Pixel Count')


# # Calculate for Many Images

# In[ ]:


def color_count_feature(in_path):
    raw_image = Image.open(base_dir / 'subset' / in_path) 
    web_image = raw_image.convert('P', palette='WEB', dither=None)
    counts, bins = np.histogram(np.array(web_image).ravel(), bins=np.arange(256))
    return counts*1.0/np.prod(web_image.size) # normalize output


# ## Calculate a few test images

# In[ ]:


get_ipython().run_cell_magic('time', '', "image_subset_df = simple_images_df.sample(100).copy()\nimage_subset_df['color_features'] = image_subset_df['image_path'].map(color_count_feature)\nimage_subset_df.sample(3)")


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
combined_features = np.stack(image_subset_df['color_features'].values, 0)
ax1.imshow(combined_features)
ax1.set_title('Raw Color Counts')
ax1.set_xlabel('Color')
ax1.set_ylabel('Frequency')
color_wise_average = np.tile(np.mean(combined_features, 0, keepdims=True), (combined_features.shape[0], 1))
ax2.imshow(combined_features/color_wise_average, vmin=0.05, vmax=20)
ax2.set_title('Normalized Color Counts')
ax2.set_xlabel('Color')
ax2.set_ylabel('Frequency')


# ## PCA Components
# We can use a tool called principle component analysis to show the images in features

# In[ ]:


from sklearn.decomposition import PCA
xy_pca = PCA(n_components=2)
xy_coords = xy_pca.fit_transform(combined_features)
image_subset_df['x'] = xy_coords[:, 0]
image_subset_df['y'] = xy_coords[:, 1]


# In[ ]:


fig, ax1 = plt.subplots(1,1, figsize=(15, 15))
for _, c_row in image_subset_df.iterrows():
    ax1.plot(c_row['x'], c_row['y'], '*')
    ax1.text(s=c_row['title'][:15], x=c_row['x'], y=c_row['y'])


# In[ ]:


def show_xy_images(in_df, image_zoom=1):
    fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
    artists = []
    for _, c_row in in_df.iterrows():
        c_img = Image.open(base_dir / 'subset' / c_row['image_path']).resize((64, 64))
        img = OffsetImage(c_img, zoom=image_zoom)
        ab = AnnotationBbox(img, (c_row['x'], c_row['y']), xycoords='data', frameon=False)
        artists.append(ax1.add_artist(ab))
    ax1.update_datalim(in_df[['x', 'y']])
    ax1.autoscale()
    ax1.axis('off')
show_xy_images(image_subset_df)


# ## TSNE Representation
# Rather than using simple PCA we can come up with a fancier representation called [TSNE](https://lvdmaaten.github.io/tsne/)

# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_iter=250, verbose=True)
xy_coords = tsne.fit_transform(combined_features)
image_subset_df['x'] = xy_coords[:, 0]
image_subset_df['y'] = xy_coords[:, 1]


# In[ ]:


show_xy_images(image_subset_df)


# ## Calculate for all images

# In[ ]:


get_ipython().run_cell_magic('time', '', "simple_images_df['color_features'] = simple_images_df['image_path'].map(color_count_feature).map(lambda x: x.tolist())\nsimple_images_df.sample(3)")


# In[ ]:


simple_images_df.to_json('color_features.json')


# In[ ]:




