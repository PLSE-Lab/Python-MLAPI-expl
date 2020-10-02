#!/usr/bin/env python
# coding: utf-8

# Minecraft Skin Analysis: Skin Suggestions using KNN
# ---
# 
# This is a follow-up to my [minecraft skin clustering notebook](https://www.kaggle.com/yedata/minecraft-skin-analysis) for developing a skin suggestions algorithm using KNN. 
# 
# There is a slight improvement on the feature engineering where I merge histograms for both RGB and HSV channels and use lower number bins for better generalisation. Clustering gives us the idea whether or not the features extracted are suitable for generalisation or not. Here, we will revisit clustering for a brief moment to check the clustering quality from the new histogram feature.

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import os
import random
import pickle
from matplotlib import image as im
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
random.seed(123456789)


# # Utility Scripts
# 
# MinecraftSkin + Stream class
# 
# I still have yet to figure out why my stream utility script is not appearing in this notebook, so I am going to put it here manually.

# In[ ]:


from mc_skin_util import MinecraftSkin, Translator # own utility script

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
  
  def next(self):
    return next(self.iterable)

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


# In[ ]:


DIR = "../input/minecraft-skins/skins/"

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
        assert np.equal(img.shape, (64, 64, 4)).all()
        return img
    except:
        return None

def get_rgb_histogram(img, channel=0, nbins=np.iinfo(np.uint8).max + 1, clip_low = 0):
    img = img[:, :, channel].flatten()
    img = img[img >= clip_low]
    ratio = (np.iinfo(np.uint8).max + 1) / nbins
    bins = [i * ratio for i in range(nbins)] + [np.iinfo(np.uint8).max + 1]
    return np.histogram(img, bins=bins, density=True)

def get_hsv_histogram(img_rgb, channel=0, nbins=np.iinfo(np.uint8).max + 1):
    img_bgr = cv2.cvtColor(img_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return get_rgb_histogram(img_hsv, channel=channel, nbins=nbins, clip_low=0) # alway unclip low

def get_merged_histogram(img, histogram_fn, **other_params):
    chnl1 = histogram_fn(img, channel=0,**other_params)
    chnl2 = histogram_fn(img, channel=1,**other_params)
    chnl3 = histogram_fn(img, channel=2,**other_params)
    return np.hstack((chnl1[0], chnl2[0], chnl3[0]))

def get_minecraft_histogram(img, hist_fn, **kwargs):
    mc = MinecraftSkin(img)
    main_features = [i for i in MinecraftSkin(get_image(0)).get_keys() if 'helm' not in i and 'acc' not in i]
    comps = [mc.get_component(i) for i in main_features]
    return np.hstack([hist_fn(i, **kwargs) for i in comps])

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


# # Clustering: Revisited
# 
# To recap, there are two performing clusters while the others are not well-defined in the last clustering notebook. So, I have concluded that I can only make three clusters for this dataset.

# ### Histogram Features

# In[ ]:


rgb_bins = 20 # rgb bins
hsv_bins = 10 # hsv bins

rgbhist_fn = lambda x: get_merged_histogram(x, get_rgb_histogram, nbins=rgb_bins, )
hsvhist_fn = lambda x: get_merged_histogram(x, get_rgb_histogram, nbins=hsv_bins)

mhist_fn = lambda x: np.hstack([get_minecraft_histogram(x, rgbhist_fn), get_minecraft_histogram(x, hsvhist_fn)])


# ### Training Set 
# 
# Using only 50% from the dataset

# In[ ]:


keep = 0.5
imgs_hist = Stream.of(range(get_max_bound())).filter(lambda x: random.random() < keep).map(get_or_none).filter(lambda x: x is not None).map(mhist_fn).filter(lambda x: not np.isnan(np.max(x))).collect()


# ### K-Means Clustering
# 
# There will be 3 clusters and 30 trials for finding the best clustering model.

# In[ ]:


from sklearn.cluster import KMeans
nclusters = 3
ninit = 30
kmeans = KMeans(n_clusters=nclusters, n_init=ninit, algorithm='elkan', n_jobs=-2).fit(imgs_hist)


# ### Validation Set
# 
# Using only 2% of the dataset

# In[ ]:


tst_prob = 0.02
img_test = Stream.of(range(get_max_bound())).filter(lambda x: random.random() < tst_prob).map(get_or_none).filter(lambda x: x is not None).filter(lambda x: x.shape[2] == 4).collect()


# In[ ]:


predictions = Stream.of(img_test).map(lambda x: (x, mhist_fn(x))).filter(lambda x: not np.isnan(np.max(x[1]))).map_key(lambda x: (kmeans.predict([x[1]])[0], x[0])).collect()

p_sorted = dict(
  zip(
    [i[0] for i in predictions],
    [i[1:] for i in predictions]
))


# ### Cluster Sample (Body Skin)

# In[ ]:


for i, group in enumerate(p_sorted.keys()):
    print(f"Group {i + 1} - Cluster Name: {group}")
    imgs = Stream.of(p_sorted[group]).map(lambda x: MinecraftSkin(x).get_component('head_front')).collect()
    stacked = stack_images(imgs)
    ratio = stacked.shape[0] / stacked.shape[1]
    width = 5
    plt.figure(figsize = (width, width * (ratio)))
    plt.imshow(stacked)
    plt.show()


# The cluster with the second most member shows mostly darker shade of skin color as compared to the cluster with the most member, but both on them has well-defined gradient in their head skin colour. In the last cluster, the head skins are mostly plain, which gives them a high contrast feature. Therefore, we can make three groupings of:
# 
# - High contrast skin
# - Dark skins
# - Light skins

# # Labelling and Analysing Clusters
# 
# Here, I re-run the predictions on the whole dataset and then write the output into a `DataFrame` so that we can analyse the clusters.

# In[ ]:


labels = Stream.of(range(get_max_bound())).map   (lambda x: (x, get_or_none(x))).filter(lambda x: x[1] is not None).map   (lambda x: (x[0], mhist_fn(x[1]))).filter(lambda x: not np.isnan(np.max(x[1]))).map   (lambda x: (x[0], kmeans.predict([x[1]])[0])).collect()


# From the DataFrame shape, we can see that about 25% out of 7830 images in the dataset is excluded, which leaves us 5581 of images. This is majorly due to the image filter for those that does not have an alpha channel as well as those that cannot be parsed into histograms. However, I think 5000 is still a large number and we can still get somem good insights out of this.

# In[ ]:


clusters = pd.DataFrame(data=labels, columns=['imagename', 'cluster'])
clusters.shape


# We can also group the dataframe based on their cluster name and then count the number of instances in each group. From the output, we can see that cluster with the least number of skins is for the high-contrast skins. The rest of the clusters have about the same number of instances.

# In[ ]:


clusters.groupby('cluster').count()


# Here, I save the histogram parameters so that I can reconstruct the histogram function to be used for KNN algorithm input.

# In[ ]:


pickle.dump([rgb_bins, hsv_bins], open('hist_bin_params.pickle', 'wb'))


# In[ ]:


def get_hist_fn(rgb_bins, hsv_bins):
    rgbhist_fn = lambda x: get_merged_histogram(x, get_rgb_histogram, nbins=rgb_bins)
    hsvhist_fn = lambda x: get_merged_histogram(x, get_rgb_histogram, nbins=hsv_bins)

    return lambda x: np.hstack([get_minecraft_histogram(x, rgbhist_fn), get_minecraft_histogram(x, hsvhist_fn)])  


# # K-Nearest Skin Suggestions
# 
# In the skin suggestion algorithm, the suggested skin will be among the given dataset only, so the provided output will be local-optima, but because the skin option is huge, I think the user will not ever run out of option. The KNN algorithm will give out the list of skins for the whole dataset ordered by their closest histogram distance, so I will make use of my own `Stream#next` implementation to iterate through the list.

# In[ ]:


rbins, hbins = pickle.load(open('hist_bin_params.pickle', 'rb'))
mhist_fn = get_hist_fn(rbins, hbins)


# Here, I expand the csv from the index reference to the image array and then to their histogram feature. The reason I don't save the histogram features into csv is because I want to dynamically regenerate the feature for different histogram parameter (number of bins, etc). This makes it easily tweakable for interface that I am planning to implement on skin suggestion application.

# In[ ]:


expanded = Stream.of(range(get_max_bound())).map   (lambda x: (x, get_or_none(x))).filter(lambda x: x[1] is not None).map   (lambda x: (x[0], mhist_fn(x[1]))).filter(lambda x: not np.isnan(np.max(x[1]))).map(lambda x: np.hstack(x)).collect()


# In[ ]:


features = ['imgname'] + [f'f{i}' for i, v in enumerate(np.array(expanded).T[:-1])]
img_hists = pd.DataFrame(expanded, columns=features)
img_hists.imgname = img_hists.imgname.astype(np.int)
img_hists = img_hists.set_index('imgname')


# Here, I initialise the model that can run on multithreading using all available cores.

# In[ ]:


from sklearn.neighbors import NearestNeighbors as KNN

nearest_model = KNN(n_jobs=-1)
nearest_model.fit(img_hists)
n_samples = img_hists.shape[0]


# In[ ]:


def get_nearest_skins(hist, model, n):
    return model.kneighbors([hist], n)[1][0]
knn_fn = lambda x: (
    Stream.of(get_nearest_skins(mhist_fn(x), nearest_model, n_samples))
    .map(lambda x: img_hists.index[x])
)


# This is an utility function that can show the front view of the skin with the skin accessories. 

# In[ ]:


def compute_skin(ms):
    if not isinstance(ms, MinecraftSkin):
        ms = MinecraftSkin(ms)
    head = ms.get_component('head_front')
    body = ms.get_component('body')
    larm = ms.get_component('arm_left')
    rarm = ms.get_component('arm_right')
    lleg = ms.get_component('leg_left')
    rleg = ms.get_component('leg_right')

    helm = ms.get_component('helm_front')
    body_acc = ms.get_component('body_acc')
    larm_acc = ms.get_component('arm_left_acc')
    rarm_acc = ms.get_component('arm_right_acc')
    lleg_acc = ms.get_component('leg_left_acc')
    rleg_acc = ms.get_component('leg_right_acc')

    head = np.array(Image.alpha_composite(Image.fromarray(head), Image.fromarray(helm)))
    body = np.array(Image.alpha_composite(Image.fromarray(body), Image.fromarray(body_acc)))
    larm = np.array(Image.alpha_composite(Image.fromarray(larm), Image.fromarray(larm_acc)))
    rarm = np.array(Image.alpha_composite(Image.fromarray(rarm), Image.fromarray(rarm_acc)))
    lleg = np.array(Image.alpha_composite(Image.fromarray(lleg), Image.fromarray(lleg_acc)))
    rleg = np.array(Image.alpha_composite(Image.fromarray(rleg), Image.fromarray(rleg_acc)))

    chunk = 8
    semi = 4

    spacing = 0
    margin = 2

    #                       arm              body              arm
    total_width = margin + semi + spacing + chunk + spacing + semi + margin

    #                        head                 body                  leg
    total_height = margin + chunk + spacing + 3 * semi + spacing + 3 * semi + margin

    # translators
    head_trans = Translator(0, 0, int((total_width - head.shape[1]) / 2), margin, head.shape[1], head.shape[0])
    body_trans = Translator(semi, semi, int((total_width - chunk) / 2), margin + head.shape[0] + spacing, chunk, 3 * semi)
    larm_trans = Translator(semi, semi, margin, margin + head.shape[0] + spacing + 1, semi, 3 * semi)
    rarm_trans = Translator(semi, semi, total_width - margin - semi, margin + head.shape[0] + spacing + 1, semi, 3 * semi)
    lleg_trans = Translator(semi, semi, int(total_width / 2) - semi, total_height - margin - 3 * semi, semi, 3 * semi)
    rleg_trans = Translator(semi, semi, int(total_width / 2), total_height - margin - 3 * semi, semi, 3 * semi)

    canvas = np.zeros((total_height, total_width, 4), dtype=np.uint8)

    canvas = head_trans.translate(head, canvas)
    canvas = body_trans.translate(body, canvas)
    canvas = larm_trans.translate(larm, canvas)
    canvas = rarm_trans.translate(rarm, canvas)
    canvas = lleg_trans.translate(lleg, canvas)
    canvas = rleg_trans.translate(rleg, canvas)
    return canvas


# Now, we can select any of the valid images and grab the nearest 16 images from the dataset. Of course, from the previous, the closest skin will always appear the same with the supplied skin because we are using the skin from the same dataset. This may not happen when supplying skin outside of the data set.

# In[ ]:


test_img = get_image(200)

suggested = knn_fn(test_img)

stacked = [compute_skin(get_image(suggested.next())) for _ in range(4 * 4)]
stacked = stack_images(stacked, 8)

plt.imshow(compute_skin(test_img))
plt.show()

ratio = stacked.shape[0] / stacked.shape[1]
plt.figure(figsize=(8, 8 * ratio))
plt.imshow(stacked, interpolation='none')
plt.show()


# Most of the suggested skins appear the similar with the supplied skin. Black feature is the strongest similarities among all. However, there are a lot of duplication, which has made the suggestions becomes dirty.

# In[ ]:


test_img = get_image(800)

suggested = knn_fn(test_img)

stacked = [compute_skin(get_image(suggested.next())) for _ in range(4 * 4)]
stacked = stack_images(stacked, 8)

plt.imshow(compute_skin(test_img))
plt.show()

ratio = stacked.shape[0] / stacked.shape[1]
plt.figure(figsize=(10, 10 * ratio))
plt.imshow(stacked, interpolation='none')
plt.show()


# In this following skin generation, the suggested skins are mostly bright. Skin duplication is still noticable, but we can see that the suggestions are very similar to the supplied image.

# # Deduplicate Skin
# 
# One advantage of skin images is that their texture feature are fixed and the same. For example, the head of the skin is always at the same position for all skin. To identify similar skins, we can make a distance function to directly compare the skin images pixel by pixel and use a very low threshold. The following distance function calculates the percentage of absolute difference between two front view of a skin. 

# In[ ]:


def diff(target_img, test_img):
    diff_1 = compute_skin(target_img).astype(np.int)
    diff_2 = compute_skin(test_img).astype(np.int)
    diff = np.abs(diff_1 - diff_2)
    return diff.sum() / (diff.shape[0] * 255 + 1)


# We use a dictionary to put the duplications as the dictionary key and associate it with the similar image index. I am using `tqdm` to see the progress of the image analysis.

# In[ ]:


similar = {}
threshold = 0.01
errors = []
hist_df = img_hists
with tqdm(total=hist_df.shape[0]) as progress:
    progress.set_description("Processing...")
    for i in hist_df.iterrows():
        progress.update(1)
        index, value = i
        if (index in similar.keys()):
            progress.set_description(f'Image {index} is already similar with image {similar[index]}.')
            continue
        target_img = get_image(index)
        candidates = nearest_model.kneighbors([mhist_fn(target_img)], n_samples)[1][0]
        s = Stream.of(candidates)

        sims = 0
        try:
            while(True):
                test = hist_df.index[s.next()]
                if (test == index): continue
                test_img = get_image(test)
                if (diff(target_img, test_img) >= threshold):
                    break
                similar[test] = index
                sims += 1
                progress.set_description(f'Image {index} has {sims} other similar images.')
        except Exception as e:
            errors += [index]
            print(f'Error! At Image {index}', e)


# We can inspect the content in the `similar` dictionary to check if we have successfully register the duplicated entries.

# In[ ]:


target = random.choice(list(similar.keys()))
test = similar[target]

target_img = get_image(target)
test_img = get_image(test)

plt.imshow(compute_skin(target_img))
plt.show()
plt.imshow(compute_skin(test_img))
plt.show()
assert target != test


# Unfortunately, after filtering all duplications, we are only left with about 25% of the original size of dataset. 

# In[ ]:


img_hists[img_hists.index.isin(similar.keys())].shape


# # Conclusion
# 
# I have re-run the whole notebook with different value for rgb and hsv bins. Overall, I think HSV should be just enough to help adding more context to the image. When there are too many parameter, the generalisation becomes less effecient to group similar skins. Skin suggestion application can be made possible because the suggested skins are very similar with the supplied skin in terms of the tone of the skin color, but there are many duplications. But this can be easily removed because the duplicated skins usually appear next to each other, so we can introduce distance threshold for two skins to be considered as a duplication. Finally, this is where I found out that the dataset has too many duplicated skins and only about 25% of them is unique.
