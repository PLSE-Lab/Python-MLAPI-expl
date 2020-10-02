#!/usr/bin/env python
# coding: utf-8

# # Basic data exploration:
# 
# 1. distribution of images per whale
# 1. viewing some images (same whale, different whale, 'new_whale')
# 1. distribution of image resolution between train & test
# 1. duplicate image analysis by perceptual hash

# In[ ]:


# used ideas from:
# https://www.kaggle.com/mmrosenb/whales-an-exploration 
# https://www.kaggle.com/stehai/duplicate-images


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 9]

import collections
from PIL import Image

DIR = "../input"

train = pd.read_csv(os.path.join(DIR, "train.csv"))
test = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))


# In[ ]:





# In[ ]:





# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# ## Distribution of images per whale is highly skewed.
# 
# 1. 2000+ whales have just one image
# 2. Single whale with most images have 73 of them
# 3. Images dsitribution:
#   1. almost 30% comes from whales with 4 or less images
#   1. almost 40% comes from 'new_whale' group
#   1. the rest 30% comes from whales with 5-73 images
# 

# In[ ]:


train['Id'].value_counts()[:4]


# In[ ]:


counted = train.groupby("Id").count().rename(columns={"Image":"image_count"})
counted.loc[counted["image_count"] > 80,'image_count'] = 80
plt.figure()
sns.countplot(data=counted, x="image_count")
plt.show()
sns.distplot(counted["image_count"], norm_hist=True, kde=False, hist_kws={'cumulative': True})


# In[ ]:


image_count_for_whale = train.groupby("Id", as_index=False).count().rename(columns={"Image":"image_count"})
whale_count_for_image_count = image_count_for_whale.groupby("image_count", as_index=False).count().rename(columns={"Id":"whale_count"})
whale_count_for_image_count['image_total_count'] = whale_count_for_image_count['image_count'] * whale_count_for_image_count['whale_count']
whale_count_for_image_count['image_total_count_cum'] = whale_count_for_image_count["image_total_count"].cumsum() / len(train)
sns.barplot(x='image_count',y='image_total_count_cum',data=whale_count_for_image_count)


# In[ ]:


whale_count_for_image_count[:5]


# In[ ]:


whale_count_for_image_count[-3:]


# # Let's see some images
# 
# #### Some images of 'new_whale'

# In[ ]:


fig = plt.figure(figsize = (20, 15))
for idx, img_name in enumerate(train[train['Id'] == 'new_whale']['Image'][:12]):
    y = fig.add_subplot(3, 4, idx+1)
    img = cv2.imread(os.path.join(DIR,"train",img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y.imshow(img)

plt.show()


#     #### Now some pictures of whales that have just 1 image: quite a large variance in colors

# In[ ]:


single_whales = train['Id'].value_counts().index[-12:]
fig = plt.figure(figsize = (20, 15))

for widx, whale in enumerate(single_whales):
    for idx, img_name in enumerate(train[train['Id'] == whale]['Image'][:1]):
        axes = widx + idx + 1
        y = fig.add_subplot(3, 4, axes)
        img = cv2.imread(os.path.join(DIR,"train",img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y.imshow(img)

plt.show()


# #### Below: each row shows pictures of one whale. I think it's quite easy to at least see similiar appearence there

# In[ ]:


topN=5
top_whales = train['Id'].value_counts().index[1:1+topN]
fig = plt.figure(figsize = (20, 5*topN))

for widx, whale in enumerate(top_whales):
    for idx, img_name in enumerate(train[train['Id'] == whale]['Image'][:4]):
        axes = widx*4 + idx+1
        y = fig.add_subplot(topN, 4, axes)
        img = cv2.imread(os.path.join(DIR,"train",img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y.imshow(img)

plt.show()


# # Resolutions
# 
# #### over 7000 unique resolutions but 39 most popular cover ~45% images (both in train and in test)

# In[ ]:


imageSizes_train = collections.Counter([Image.open(f'{DIR}/train/{filename}').size
                        for filename in os.listdir(f"{DIR}/train")])
imageSizes_test = collections.Counter([Image.open(f'{DIR}/test/{filename}').size
                        for filename in os.listdir(f"{DIR}/test")])


# In[ ]:


def isdf(imageSizes):
    imageSizeFrame = pd.DataFrame(list(imageSizes.most_common()),columns = ["imageDim","count"])
    imageSizeFrame['fraction'] = imageSizeFrame['count'] / sum(imageSizes.values())
    imageSizeFrame['count_cum'] = imageSizeFrame['count'].cumsum()
    imageSizeFrame['count_cum_fraction'] = imageSizeFrame['count_cum'] / sum(imageSizes.values())
    return imageSizeFrame

train_isdf = isdf(imageSizes_train)
train_isdf['set'] = 'train'
test_isdf = isdf(imageSizes_test)
test_isdf['set'] = 'test'


# In[ ]:


isizes = train_isdf.merge(test_isdf, how="outer", on="imageDim")
isizes['total_count'] = isizes['count_x'] + isizes['count_y']
dims_order = isizes.sort_values('total_count', ascending=False)[['imageDim']]
len(dims_order)


# In[ ]:


isizes = pd.concat([train_isdf, test_isdf])


# In[ ]:


isizes.shape


# In[ ]:


isizes.head()


# In[ ]:


popularSizes = isizes[isizes['fraction'] > 0.002]
popularSizes.shape


# In[ ]:


popularSizes.groupby('set').max()['count_cum_fraction']


# In[ ]:


sns.barplot(x='imageDim',y='fraction',data = popularSizes, hue="set")
_ = plt.xticks(rotation=45)


# # Duplicates
# 
# 1. 1 duplicate in train set
# 1. 3 duplicates between train and test
# 1. totally different than in playground dataset: 
#   1. [playground duplicates](https://www.kaggle.com/stehai/duplicate-images)
#   1. [solution that used duplicate information](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563)
# 

# In[ ]:


import imagehash

def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, img_hash

def get_img_duplicates_info(df, dataset):
    
    m = df.Image.apply(lambda x: getImageMetaData(os.path.join(DIR, dataset, x)))
    df["Hash"] = [str(i[2]) for i in m]
    df["Shape"] = [i[0] for i in m]
    df["Mode"] = [str(i[1]) for i in m]
    df["Length"] = df["Shape"].apply(lambda x: x[0]*x[1])
    df["Ratio"] = df["Shape"].apply(lambda x: x[0]/x[1])
    df["New_Whale"] = df.Id == "new_whale"
    
    
    img_counts = df.Id.value_counts().to_dict()
    df["Id_Count"] = df.Id.apply(lambda x: img_counts[x])
    return df


# In[ ]:


train_dups = get_img_duplicates_info(train, "train")


# In[ ]:


train_dups.head()


# In[ ]:


t = train_dups.Hash.value_counts()
t = t.loc[t>1]


# In[ ]:


"Duplicate hashes: {}".format(len(t))


# In[ ]:


t


# In[ ]:


t.index[0]


# In[ ]:


train_dups[train_dups['Hash'] == t.index[0]].head()


# #### The only duplicate found in train dataset comes from the same whale.
# 

# In[ ]:


fig = plt.figure(figsize = (20, 10))
for idx, img_name in enumerate(train_dups[train_dups['Hash'] == t.index[0]]['Image'][:2]):
    y = fig.add_subplot(3, 4, idx+1)
    img = cv2.imread(os.path.join(DIR,"train",img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y.imshow(img)

plt.show()


# In[ ]:


test_dups = get_img_duplicates_info(test, "test")


# In[ ]:


test_d = test_dups.Hash.value_counts()
test_d = test_d.loc[test_d>1]
"Duplicate hashes in test: {}".format(len(test_d))


# In[ ]:


common_hashes = test_dups.merge(train_dups, how="inner", on="Hash", suffixes=("_test","_train"))
common_hashes.head()


# In[ ]:


"Duplicate hashes between train and test: {}".format(len(common_hashes))


# ### below each row shows images with the same pHash, left column from train, right from test

# In[ ]:


fig = plt.figure(figsize = (10, 10))

for idx, images in enumerate(common_hashes[['Image_train','Image_test']].values):
    y = fig.add_subplot(len(common_hashes),2, idx*2+1)
    img = cv2.imread(os.path.join(DIR,"train",images[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y.imshow(img)

    y = fig.add_subplot(len(common_hashes),2, idx*2+2)
    img = cv2.imread(os.path.join(DIR,"test",images[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y.imshow(img)

    
plt.show()


# In[ ]:


# train duplicates - to remove:
train_to_remove = train_dups[train_dups['Hash'] == t.index[0]].drop_duplicates('Hash')[['Image']]
train_to_remove.to_csv("train_remove.csv",index=False)
train_to_remove.head()


# In[ ]:


# easy answers in test:
easy_peasy = common_hashes[['Image_test','Id_train']]
easy_peasy.to_csv("test_easy.csv", index=False)
easy_peasy.head()

