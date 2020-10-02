#!/usr/bin/env python
# coding: utf-8

# In my [last kernel](https://www.kaggle.com/bdubreu/investigating-outlier-pixels-dicom-gotchas), I was looking at outliers scans in terms of their pixels values. We saw that +30000 pixel value was a perfectly reasonable value, because sometimes patients have a golden tooth that has a very high density.
# 
# While doing that, we found a crappy scan. Can we find others ? Let's install some stuff and get started

# In[ ]:


get_ipython().system('pip install torch torchvision feather-format pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai_dev             > /dev/null')

from fastai2.basics           import *
from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)


# In[ ]:


path_inp = Path('../input')
path = path_inp/'rsna-intracranial-hemorrhage-detection'
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'


# In[ ]:


# This is the crappy file we found: 
dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').show(figsize=(6,6))

# the image is really noisy. That's a 600K+ images dataset we're looking at, so there might be more of those. Can we find them ?


# In[ ]:


# This pic has two interesting features:
# - a std of 11269!
# - a lower quartile of -2000
pd.Series((dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array.flatten())).describe()


# ### We'll start by sorting images with their stds

# In[ ]:


# To understand what's going on here, please refer to https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai
path_df = path_inp/'creating-a-metadata-dataframe'

df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth')

comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
assert not len(comb[comb['any'].isna()])


# In[ ]:


# So, a std of 11269 was indeed a very weird value, considering the 99th percentile is 1340 
# (that means 99% of picture have a std of 1340 or lower)
comb['img_std'].quantile([0.5, 0.7, 0.9, 0.99, 0.999, 0.9999])


# In[ ]:


# Indeed, this is actually the only image with that kind of standard deviation
# The second largest standard dev (in terms of pixel values) is 1513
comb['img_std'].sort_values(ascending=False)[:5]


# In[ ]:


# other images with a large std (> 1500) show no signs of being corrupt
f_name = comb[ comb['img_std'] > 1500 ].sample()['fname'].values[0]
print(f_name)
dcmread(f_name).show(figsize=(6,6))
pd.Series(dcmread(f_name).pixel_array.flatten()).describe()


# So looking for image with huge stds in terms of pixel values won't help us find other corrupted images. But it might be only because there aren't any to find. I think the std approach was correct because of the following:

# In[ ]:


# I used matplotlib to get a sense of which line started to get crappy
import matplotlib.pyplot as plt
plt.imshow(dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array)
# Around 300-350


# In[ ]:


# Doing some bisection search, I found the limit:
# line 332 (indexing starts at 0) has 864 std
dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array[331].std()


# In[ ]:


# but then at line 333, std in terms of pixel values starts to skyrocket:
dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array[333].std()


# #### So I don't think we'll find another picture like the above in the dataset. I've also looked at the test set, and luckily for us there aren't any images with high std. However, I found interesting stuff while looking at i...

# # Train vs. Test distribution of pixel_values

# In[ ]:


df_trn['img_std'].drop(640545).describe()


# In[ ]:


df_tst['img_std'].describe()


# The mean stardard deviation of the train and tests set are very different. It might start getting confusing here, so let me rephrase: on average, the standard deviation in the train set is 791.9.
# In the train set, a typical image will see its pixel move away from their mean a lot less than a typical image in the test set. This ought to be investigated.
# 
# Also, what is an image with a std of 0 ? Are all pixels black ?

# # Images with low stds: corrupt and/or peculiar

# In[ ]:


# we find back the image noted in this kernel : https://www.kaggle.com/tonyyy/corrupted-pixeldata-in-id-6431af929-dcm
df_trn[df_trn['img_std'] == 0]['fname']
# in fact, we can't plot any of those pics because they are all corrupt


# In[ ]:


# in fact, we can't plot any of those pics because they are all corrupt
# although only the ID_6431af929.dcm image will raise a value error
# the four other raise index Errors, and I don't clearly know why
# select the three following lines and then CTRL+/ to uncomment them all at once (amazing, isn't it?)

# fname = df_trn[df_trn['img_std'] == 0].sample()['fname'].values[0]
# print(fname)
# dcmread(fname).show(figsize=(6,6))


# How does the other stds look like ? 

# In[ ]:


# Images with std between 0 and 1 are really spherical. I don't know what we should do with those ?! 
# You can press shift-enter several times here to see a bunch of examples
fname = df_trn[(df_trn['img_std'] > 0) & (df_trn['img_std'] < 1)].sample()['fname'].values[0]
print(fname)
dcmread(fname).show(figsize=(6,6))


# In[ ]:


# These is (only) one similar picture in the test set
fname = df_tst[(df_tst['img_std'] > 0) & (df_tst['img_std'] < 1)].sample()['fname'].values[0]
print(fname)
dcmread(fname).show(figsize=(6,6))


# # So what does it mean, for a scan, to have high vs. low std ?
# You'll see that they hardly look similar. In fact, std less than 300 hardly look like brains to me... Whereas 600+ std images look more "normal" ?

# ### Low std images

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(20,10))
for i, img in enumerate(comb[ comb['img_std'] < 300 ].sample(8)['fname'].values):
    dcmread(img).show(ax=axes[i%2, i//2])


# ### Medium std images

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(20,10))
for i, img in enumerate(comb[ comb['img_std'] > 600 ].sample(8)['fname'].values):
    dcmread(img).show(ax=axes[i%2, i//2])


# ### High std images

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(20,10))
for i, img in enumerate(comb[ comb['img_std'] > 1200 ].sample(8)['fname'].values):
    dcmread(img).show(ax=axes[i%2, i//2])


# #### Perhaps unsurprisingly, the more the std, the more detail in the picture. But I would be wary to pass the first bunch of pictures to a model... since the stds seem to be higher in the test set, maybe we can discard the low-detailed pictures ?

# 1. ### Low std images (test set)

# In[ ]:


# Problem: we have low detail images in the test set as well
fig, axes = plt.subplots(2, 4, figsize=(20,10))
for i, img in enumerate(df_tst[ df_tst['img_std'] < 300 ].sample(8)['fname'].values):
    dcmread(img).show(ax=axes[i%2, i//2])


# # temporary conclusions
# 1. we shouldn't find other noisy pictures like the one at the top of this notebook
# 2. pixel values vary much less in test data than in train data
# 3. This is probably a problem, because pictures with very low std are actually very different both in SHAPE and TEXTURE than their high std counterparts. Pics with std between 0 and 1 are basically weird spheres (english grammar here: should I say different than or different to ?)
# 4. Are these pictures susceptible to screw up the normalizing of the data ? I don't know... What do you think ?

# In[ ]:




