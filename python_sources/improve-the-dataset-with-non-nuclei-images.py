#!/usr/bin/env python
# coding: utf-8

# # Dataset of non nuclei images
# 
# We prepared a small dataset of images that contain no nuclei and can be added to training in order to help your model deal with some artifacts.

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import math
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def plot_list(images=[], labels=[], n_rows=1):
    n_img = len(images)
    n_lab = len(labels)
    n_cols = math.ceil((n_lab+n_img)/n_rows)
    plt.figure(figsize=(16,10))
    for i, image in enumerate(images):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(image)
    for j, label in enumerate(labels):
        plt.subplot(n_rows,n_cols,n_img+j+1)
        plt.imshow(label, cmap='nipy_spectral')
    plt.show()


# In[20]:


non_nuclei_images = joblib.load('../input/non-nuclei-images/non_nuclei_images.pkl')


# In[21]:


plot_list(non_nuclei_images, n_rows=4)


# # Full pipeline
# If you would like to see how we plugged this data in just go to our [open solution](https://github.com/neptune-ml/open-solution-data-science-bowl-2018)
# 
# ![full open solution pipeline](https://gist.githubusercontent.com/jakubczakon/10e5eb3d5024cc30cdb056d5acd3d92f/raw/e85c1da3acfe96123d0ff16f8145913ee65e938c/full_pipeline.png)
# 
# In the `main.py` of the `dev-patching` branch we have a function that generates metadata that looks like this:
# 
# ```python
# @action.command()
# def prepare_metadata():
#     logger.info('creating metadata')
#     meta = generate_metadata(data_dir=params.data_dir,
#                              masks_overlayed_dir=params.masks_overlayed_dir,
#                              contours_overlayed_dir=params.contours_overlayed_dir,
#                              contours_touching_overlayed_dir=params.contours_touching_overlayed_dir,
#                              centers_overlayed_dir=params.centers_overlayed_dir)
#     logger.info('calculating clusters')
#     meta_train = meta[meta['is_train'] == 1]
#     meta_test = meta[meta['is_train'] == 0]
#     vgg_features_clusters = get_vgg_clusters(meta_train)
#     meta_train['vgg_features_clusters'] = vgg_features_clusters
#     meta_test['vgg_features_clusters'] = 'NaN'
#     meta = pd.concat([meta_train, meta_test], axis=0)
# 
#     logger.info('adding artifacts metadata')
#     meta_artifacts = build_artifacts_metadata(artifacts_dir=params.artifacts_dir)
#     meta = pd.concat([meta, meta_artifacts], axis=0)
# 
#     meta.to_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'), index=None)
# ```
# 
# Feel free to use it in your solution.
# Good luck!

# In[ ]:




