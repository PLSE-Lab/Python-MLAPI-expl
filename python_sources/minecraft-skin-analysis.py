#!/usr/bin/env python
# coding: utf-8

# Minecraft Skin Analysis
# ===
# 
# I am performing simple Minecraft Skin analysis for machine-learning motivated skin program. I am planning to perform skin clustering, skin suggestions, and finally, skin generation using random mix and match and using machine-learning.
# 
# In this part of notebook, I am using K-Means algorithm to perform clustering using RBG histogram image description. I will write out my analysis throughout the notebook.

# ### Import modules and utility functions

# In[ ]:


import pandas as pd
import numpy as np
import os
import random
from matplotlib import image as im
from matplotlib import pyplot as plt
random.seed(9876543210)


# In[ ]:


DIR = "../input/minecraft-skins/skins"

def get_max_bound():
    return len(os.listdir(DIR))

def get_image(index):
    try:
        return im.imread(os.path.join(DIR, os.listdir(DIR)[index]))
    except Exception as e:
        return im.imread(os.path.join(DIR, os.listdir(DIR)[index]), 0)

def get_file(index):
    return os.listdir(DIR)[index]

def get_or_none(index):
    try:
        img = get_image(index)
        assert np.equal(img.shape, (64, 64, 4)).all
        return img
    except:
        return None


# In[ ]:


plt.imshow(get_image(505))
plt.show()


# ### Stream functions
# 
# This is a stream class inspired by Java Stream class. This provides better code readability.

# In[ ]:


# simple stream object from: https://gist.github.com/ye-yu/26d6806ceb1f7b0712763a1ce6bdac29
# simple stream object
class Stream:
  def __init__(self, iterable):
    self.iterable = iterable
    self.operations = []
  
  def filter(self, fn):
    return self.of(filter(fn, self.iterable))._stamp(self, "filter " + fn.__name__)

  def map(self, fn):
    return self.of(map(fn, self.iterable))._stamp(self, "map " + fn.__name__)
  
  def map_key(self, fn):
    '''
    fn must return tuple length more than 1
    '''
    return self.of(self.__map_key(fn, self.collect()))._stamp(self, "map " + fn.__name__)

  def reduce(self, fn):
    iterable = self.iterable
    result = None
    try:
      result = next(iterable)
      while(True):
        result = fn(result, next(iterable))
    except:
      return result
    
  def collect(self):
    return list(self.iterable)
  
  def enumerate(self):
    return Stream(enumerate(self.iterable))
  
  def _stamp(self, streamobj, name):
    self.operations = streamobj.operations
    self.operations += [name]
    return self

  def __str__(self):
    return self.__repr__() + "\n  -> " + "\n  -> ".join(self.operations)
  
  def __repr__(self):
    return "[Stream object]"
  
  @staticmethod
  def of(iterable):
    return Stream(iter(iterable))
  
  @staticmethod
  def __map_key(fn, arr):
    ref = dict()
    for i in arr:
      k = fn(i)
      if (k[0] not in ref.keys()):
        ref[k[0]] = []
      ref[k[0]] += k[1:]

    for k in ref.keys():
      yield [k] + ref[k]


# # Simple image analysis
# 
# ### Check invalid images
# 
# I found out that some of the images cannot be read by the `matplotilb` module. We can also see the utilisation of Stream in action. After analysis, there are less than 1% faulty skin files from the whole dataset, and I reckon we can safely discard these.

# In[ ]:


invalid_imgs = Stream.of(range(get_max_bound())).filter(lambda x: get_or_none(x) is None).collect()


# In[ ]:


invalid_imgs


# # Feature engineering
# 
# ### Image utility function
# 
# These classes help to extract components of the Minecraft Skin. In the `MinecraftSkin` class, there is a function called `get_deterministic_random` of which the output is unique for each skin. This can be useful for random skin generation using seed.

# In[ ]:


# utility function from: https://gist.github.com/ye-yu/1f8e5eaa4a6d162d3ee11354c83fe910
import numpy as np

class ImageCapture:
  def __init__(self, x, y, width, height):
    assert type(x) is int, f"x must be int. Got {x}"
    assert type(y) is int, f"y must be int. Got {y}"
    assert type(width) is int, f"width must be int. Got {width}"
    assert type(height) is int, f"height must be int. Got {height}"

    self.x = x
    self.y = y
    self.width = width
    self.height = height

    
  def capture(self, arr):
    if (not isinstance(arr, np.ndarray)):
      arr = np.array(arr)
    return arr[
      self.y:self.y + self.height,
      self.x:self.x + self.width,
      :
    ]

class Translator(ImageCapture):
  def __init__(self, sourceX=None, sourceY=None, destX=None, destY=None, width=None, height=None, imageCapture=None):
    if imageCapture is not None:
      super().__init__(imageCapture.x, imageCapture.Y, imageCapture.width, imageCapture.height)
    else:
      super().__init__(sourceX, sourceY, width, height)
    
    assert type(destX) is int, f"destX must be int. Got {destX}"
    assert type(destY) is int, f"destY must be int. Got {destY}"

    self.destX = destX
    self.destY = destY

  def translate(self, source, dest):
    source = self.capture(source)
    dest[self.destY:self.destY + self.height, self.destX:self.destX + self.width, :] = source
    return dest
  
  def to_img_capture(self):
    return ImageCapture(self.x, self.y, self.width, self.height)

class MinecraftSkin:
  def __init__(self, arr):
    if np.equal(arr.shape, (32, 64, 4)).all():
      self.img32 = arr
      self.img64 = np.vstack((self.img32, np.zeros((32, 64, 4), dtype=np.uint8)))
      for translator in MinecraftSkin.__get_translators():
        self.img64 = translator.translate(self.img32, self.img64)
    elif np.equal(arr.shape, (64, 64, 4)).all():
      self.img64 = arr
      self.img32 = self.img64[:32, :, :]
    else:
      raise Exception("Invalid shape. Got", arr.shape)
      
    components = {}
    
    chunk = 8
    tops = [ImageCapture(i * chunk, 0, chunk, chunk) for i in range(8)]
    bots = [ImageCapture(i * chunk, chunk, chunk, chunk) for i in range(8)]
    
    components['head_top'] = tops[1]
    components['head_bottom'] = tops[2]
    components['head_left'] = bots[0]
    components['head_front'] = bots[1]
    components['head_behind'] = bots[2]
    components['head_right'] = bots[3]
    
    components['helm_top'] = tops[5] 
    components['helm_bottom'] = tops[6]
    components['helm_left'] = bots[4]
    components['helm_front'] = bots[4]
    components['helm_behind'] = bots[4]
    components['helm_right'] = bots[4]
    
    mids = [[ImageCapture(0, (i + 1) * 2 * chunk, 2 * chunk, 2 * chunk),
              ImageCapture(2 * chunk, (i + 1) * 2 * chunk, 3 * chunk, 2 * chunk),
              ImageCapture(5 * chunk, (i + 1) * 2 * chunk, 2 * chunk, 2 * chunk)]
              for i in range(2)]
    
    components['leg_right'] = mids[0][0]
    components['leg_right_acc'] = mids[1][0]
    components['body'] = mids[0][1]
    components['body_acc'] = mids[1][1]
    components['arm_right'] = mids[0][2]
    components['arm_right_acc'] = mids[0][2]
    
    last = [ImageCapture(2 * i*chunk, 6 * chunk, 2 * chunk, 2 * chunk) for i in range(4)]
    
    components['leg_left_acc'] = last[0]
    components['leg_left'] = last[1]
    components['arm_left'] = last[2]
    components['arm_left_acc'] = last[3]
    
    self.components = components
  
  def get_keys(self):
    return self.components.keys()
  
  def get_component(self, comp):
    return self.components[comp].capture(self.img64)
  
  def get_folded_component(self, comp):
    if 'leg' in comp or 'arm' in comp:
      return MinecraftSkin.fold_limb(self.get_component(comp))
    return self.get_component(comp)
  
  def to_img64(self):
    return self.img64
  
  def to_img32(self):
    return self.img32
  
  def get_deterministic_random(self, params=1):
    import random
    random.seed(0)
    for i in self.to_img64().flatten():
      random.seed(random.random() + i)
    return [random.random() for i in range(params)]

  @staticmethod
  def fold_limb(img):
    chunk = 4
    assert np.equal(img.shape, (4 * chunk, 4 * chunk, 4)).all(), f"Shape must be {(16, 16, 4)}, got {img.shape}"
    left  = Translator(        0, chunk,     chunk, 4 * chunk, chunk, 3 * chunk)
    right = Translator(3 * chunk, chunk, 2 * chunk, 4 * chunk, chunk, 3 * chunk)
    img = np.vstack((img, np.zeros((3 * chunk, img.shape[1], img.shape[2]), dtype=np.uint8)))
    img =  left.translate(img, img)
    img = right.translate(img, img)
    return img[:, chunk:3 * chunk, :]
      
  @staticmethod
  def unfold_limb(img):
    assert np.equal(img.shape, (28, 8, 4)).all(), f"Shape must be {(28, 8, 4)}, got {img.shape}"
    chunk = 4
    left  = Translator(    0, 4 * chunk,         0,     chunk,     chunk, 3 * chunk)
    right = Translator(chunk, 4 * chunk, 3 * chunk,     chunk,     chunk, 3 * chunk)
    body  = Translator(    0,         0,     chunk,         0, 2 * chunk, 4 * chunk)
    out = np.zeros((16, 16, 4), dtype=np.uint8)
    out = left.translate(img, out)
    out = body.translate(img, out)
    out = right.translate(img, out)
    return out

  @staticmethod
  def __get_translators():
    return [
      Translator(4, 16, 20, 48, 4, 4),
      Translator(8, 16, 24, 48, 4, 4),
      Translator(0, 20, 24, 52, 4, 12),
      Translator(4, 20, 20, 52, 4, 12),
      Translator(8, 20, 16, 52, 4, 12),
      Translator(12, 20, 28, 52, 4, 12),
      Translator(44, 16, 36, 48, 4, 4),
      Translator(48, 16, 40, 48, 4, 4),
      Translator(40, 20, 40, 52, 4, 12),
      Translator(44, 20, 36, 52, 4, 12),
      Translator(48, 20, 32, 52, 4, 12),
      Translator(52, 20, 44, 52, 4, 12),
    ]


# This is one of the example of the output of the `get_deterministic_random`. The parameter for this function is for the number of outputs of this function. 

# In[ ]:


ms = MinecraftSkin(get_image(9))
ms.get_deterministic_random(3)


# The first one is a function for getting RBG histogram of the image. This histogram will be fed into the clustering algorithm. I am planning to experiment with different histogram.

# In[ ]:


def get_rgb_histogram(img, channel=0, nbins=np.iinfo(np.uint8).max + 1, clip_low = 1):
    img = img[:, :, channel].flatten()
    img = img[img >= clip_low]
    ratio = (np.iinfo(np.uint8).max + 1) / nbins
    bins = [i * ratio for i in range(nbins)] + [np.iinfo(np.uint8).max + 1]
    return np.histogram(img, bins=bins, density=True)


# In[ ]:


def get_merged_histogram(img, histogram_fn, **other_params):
    chnl1 = histogram_fn(img, channel=0,**other_params)
    chnl2 = histogram_fn(img, channel=1,**other_params)
    chnl3 = histogram_fn(img, channel=2,**other_params)
    return np.hstack((chnl1[0], chnl2[0], chnl3[0]))


# To get random images, we use probability to filter out unneeded images and make use of the stream function. Here, 50% of the dataset will be used for the training of the clustering. Upon collecting the stream, there will be an error log appearing in the output. This is due to error in the histogram function, so we use `np.max` to check if the max can be computed. If the value cannot be computed, `np.isnan` can be used to filter out the faulty output.
# 
# I assigned a lambda expression so that we can easily reuse the lambda function. In the lambda functoion, I have set the number of bins to 180 bins.

# In[ ]:


mhist_fn = lambda x: get_merged_histogram(x, get_rgb_histogram, nbins=180)

keep = 0.5
imgs_hist = Stream.of(range(get_max_bound())).filter(lambda x: random.random() < keep).map(get_or_none).filter(lambda x: x is not None).map(mhist_fn).filter(lambda x: not np.isnan(np.max(x))).collect()


# Here, I perform checking to make sure not many of histogram are the faulty outputs. The sum is still about 50% of the whole dataset, so it is safe to assume that faulty outputs are minimally produced.

# In[ ]:


np.array(imgs_hist).shape


# # Clustering algorithm
# 
# I am trying with 5 clusters for the K-Means algorithm and running 30 reps for different centroid seeds.

# In[ ]:


from sklearn.cluster import KMeans
nclusters = 5
ninit = 30
kmeans = KMeans(n_clusters=nclusters, n_init=ninit, algorithm='elkan', n_jobs=-2).fit(imgs_hist)


# Now, we prepare the validation set to check inspect the output quality using only 2% of the dataset.

# In[ ]:


tst_prob = 0.02
img_test = Stream.of(range(get_max_bound())).filter(lambda x: random.random() < tst_prob).map(get_or_none).filter(lambda x: x is not None).filter(lambda x: x.shape[2] == 4).collect()


# In[ ]:


predictions = Stream.of(img_test).map(lambda x: (x, mhist_fn(x))).filter(lambda x: not np.isnan(np.max(x[1]))).map(lambda x: (x[0], kmeans.predict([x[1]])[0])).map_key(lambda x: (x[1], x[0])).collect()


# After performing `map_key`, the key of the each grouping is placed at the first of the array. So, we can extract this out and derive a dictionary object.

# In[ ]:


p_sorted = dict(
  zip(
    [i[0] for i in predictions],
    [i[1:] for i in predictions]
))


# This is the utility function to stack images in rows and columns.

# In[ ]:


def stack_images(iterable, ncols=8):
    dim = iterable[0].shape
    blank = np.zeros(dim, dtype=np.uint8)
    a = Stream.of(iterable).enumerate().map(lambda x: (x[0] % ncols, x[1])).map_key(lambda x: (x[0], x[1])).collect()
    a = [i[1:] for i in a]
    height = max([len(i) for i in a])
    for i, v in enumerate(a):
        if (len(v) == height): continue
        a[i].append(blank)
    return np.hstack([np.vstack(i) for i in a])


# Here, we can use the `MinecraftSkin` class to view only the front head texture of the skin.

# In[ ]:


for i, group in enumerate(p_sorted.keys()):
    print(f"Group {i + 1} - Cluster Name: {group}")
    stacked = stack_images(Stream.of(p_sorted[group])
                           .map(lambda x: MinecraftSkin(x).get_component('head_front'))
                           .collect())
    ratio = stacked.shape[0] / stacked.shape[1]
    width = 5
    plt.figure(figsize = (width, width * (ratio)))
    plt.imshow(stacked)
    plt.show()


# From the output, it is hard to tell on what aspect are they clustered together. Afterall, the clustering is taking account of the whole skin file rather than just the heads. So, lets take a look at the skin body of the texture that took larger space of the skin file.

# In[ ]:


for i, group in enumerate(p_sorted.keys()):
    print(f"Group {i + 1} - Cluster Name: {group}")
    stacked = stack_images(Stream.of(p_sorted[group])
                           .map(lambda x: MinecraftSkin(x).get_component('body'))
                           .collect())
    ratio = stacked.shape[0] / stacked.shape[1]
    width = 5
    plt.figure(figsize = (width, width * (ratio)))
    plt.imshow(stacked)
    plt.show()


# In the cluster result, we can observe that there is one cluster has the strongest similarities among their members. All skins in this group are primarily dark. The second best cluster is the one that has the most members of which the skins are all not too bright nor too dark. The third most cluster consists of a number of colourful, high-contrast skins while the rest of the cluster are inconclusive due to small number of samples.

# ## Conclusion
# 
# The training dataset for the clustering is rather sparse because some parts of the skins are not used for skin rendering. This is excluded from the histogram calculation by in the `clip_min` parameters. By normalising the histogram, the values that we removed can be equalised among others.
# 
# Even after the features engineering, the clustering quality are not well-defined. `clip_min` also removes the transparent pixels of the actual skin. Perhaps, I should make use of the `MinecraftSkin` class and only crop out the actual skin to compute the histogram without removing the black pixel.
# 
# In the clustering group, there are two performing clusters while the rest are sub-optimal. Perhaps, the best number of clusters is three where the first group are for primarily dark skins, the second group are for normal skins, and the third are for bright skins. Upon completion of the clustering, we can label each cluster for skin suggestion using KNN algorithm.
