#!/usr/bin/env python
# coding: utf-8

# *This Kernel can be used to evaluate the generated descriptors for Landmark Retrieval 2020*

# # Landmark Image Retrieval
# The goal of the Landmark Recognition challenge is to recognize a landmark presented in a query image, while the goal of Landmark Retrieval 2019 is to find all images showing that landmark.
# ![](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_374127%2Fimages%2Fpictures%2Frec_demo.jpg)

# # Deep Local and Global Image Features
# In this kernel I will use pretrained DELF model for landmark retrieval. DELF project presents code for deep local and global image feature methods, which are particularly useful for the computer vision tasks of instance-level recognition and retrieval. These were introduced in the DELF, Detect-to-Retrieve, DELG.  <br>
# **Acknowledgment:** In the following link, you can find the project source code, installation guidlines and pretrained models by **@andre faraujo**: <br>
# https://github.com/tensorflow/models/tree/master/research/delf
# 
# **Please upvote if you find this kernel useful**
# 
# ![](https://www.i-programmer.info/images/stories/News/2018/march/A/delfpipeline.JPG)

# # Acknowledgement:
# Source code published on [Colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf_hub_delf_module.ipynb) by Tensorflow developers. In this kernel, I am elaborating more on the sub modules
# <br>Installing needed Packages

# In[ ]:


get_ipython().system('pip install -q scikit-image')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from absl import logging

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen


# # The data
# We will use a set of images on landmarks in different viewports. Then, we specify the URLs of two images we would like to process with DELF in order to match and compare them.

# In[ ]:


# The tutorial is based on the DElf Hub tutorial from Tensorflow
# from: https://commons.wikimedia.org/wiki/File:Bridge_of_Sighs,_Oxford.jpg
# by: N.H. Fischer
IMAGE_1_URL = 'https://upload.wikimedia.org/wikipedia/commons/2/28/Bridge_of_Sighs%2C_Oxford.jpg'
# from https://commons.wikimedia.org/wiki/File:The_Bridge_of_Sighs_and_Sheldonian_Theatre,_Oxford.jpg
# by: Matthew Hoser
IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/c/c3/The_Bridge_of_Sighs_and_Sheldonian_Theatre%2C_Oxford.jpg'


# In[ ]:


def download_and_resize(name, url, new_width=256, new_height=256):
  path = tf.keras.utils.get_file(url.split('/')[-1], url)
  image = Image.open(path)
  image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
  return image


# In[ ]:


image1 = download_and_resize('image_1.jpg', IMAGE_1_URL)
image2 = download_and_resize('image_2.jpg', IMAGE_2_URL)

plt.subplot(1,2,1)
plt.imshow(image1)
plt.subplot(1,2,2)
plt.imshow(image2)


# # The pre-trained DELF(Deep Local Feature) module:
# This module is available on TensorFlow Hub can be used for image retrieval as a drop-in replacement for other keypoint detectors and descriptors. It describes each noteworthy point in a given image with multidimensional vectors known as feature descriptor.

# In[ ]:


# You can replace this line to use your custome model
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']


# # Apply the DELF module to the data
# The DELF module takes an image as input and represnt the image in a reduced dimentional space asfeature descriptor. The following cell contains the core of this colab's logic. The DELF Image retrieval system can be decomposed into four main blocks:
# 
# * Dense localized feature extraction,
# * Keypoint selection,
# * Dimensionality reduction,
# * Indexing and retrieval.
# 
# The model will return the descriptors and feature locations. This can be used now to compare images using similarity-based methods (nearest-neighbor matches using a KD tree)

# In[ ]:


def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))


# In[ ]:


result1 = run_delf(image1)
result2 = run_delf(image2)


# # Match Images at Runtime
# At runtime, the query image was first resized and cropped to 256x256 resolution followed by the DELF module computing its descriptors and locations. Then we query the KD-tree to find K nearest neighbors for each descriptor of the query image. Next, aggregate all the matches per database image. Finally, we perform geometric verification using RANSAC and employ the number of inliers as the score for retrieved images.

# # RANSAC for geometric verification
# 
# RANSAC for geometric verification can be used to estimate geometric transformations. We want to make sure all matches are consistent with a global geometric transformation; however, there are many incorrect matches. Take the following graph for example, without the geometric verification there are many inconsistent matches while after applying RANSAC, we can estimate the geometric transformation and the set of consistent matches simultaneously.

# In[ ]:


def match_images(image1, image2, result1, result2):
  distance_threshold = 0.8

  # Read features.
  num_features_1 = result1['locations'].shape[0]
  print("Loaded image 1's %d features" % num_features_1)
  
  num_features_2 = result2['locations'].shape[0]
  print("Loaded image 2's %d features" % num_features_2)

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(result1['descriptors'])
  _, indices = d1_tree.query(
      result2['descriptors'],
      distance_upper_bound=distance_threshold)
  
  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      result2['locations'][i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      result1['locations'][indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
    # Perform geometric verification using RANSAC.
  _, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000)

  print('Found %d inliers' % sum(inliers))

  # Visualize correspondences.
  _, ax = plt.subplots(figsize=(20, 20))
  inlier_idxs = np.nonzero(inliers)[0]
  plot_matches(
      ax,
      image1,
      image2,
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b')
  ax.axis('off')
  ax.set_title('DELF correspondences')


# In[ ]:


match_images(image1, image2, result1, result2)


# # **Please upvote if you find this kernel useful**
