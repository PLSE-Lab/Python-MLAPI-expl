#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 200
pd.options.display.max_rows = 500

import os
print(os.listdir("../input"))


# In[2]:


import glob
import json
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import gc

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")
labels_df = pd.read_csv("../input/labels.csv")


# In[4]:


print("[train]")
print(len(train_df))
print(Counter(map(lambda x: x.split(".")[-1], os.listdir("../input/train/"))))

print("[test]")
print(len(test_df))
print(Counter(map(lambda x: x.split(".")[-1], os.listdir("../input/test/"))))


# ## check labels

# In[5]:


len(labels_df)


# In[6]:


labels_df["attribute_type"] = labels_df.attribute_name.apply(lambda x: x.split("::")[0])
print(labels_df["attribute_type"].value_counts())
labels_df.attribute_type.value_counts().plot.bar()


# In[7]:


labels_df.query("attribute_type == 'culture'").index 


# In[8]:


labels_df.query("attribute_type == 'tag'").index 


# In[9]:


labels_df[:398].head(10)


# In[10]:


labels_df[398:].head(10)


# ## check attibure_id's frequency in train

# In[11]:


train_attr_ohot = np.zeros((len(train_df), len(labels_df)), dtype=int)

for idx, attr_arr in enumerate(train_df.attribute_ids.str.split(" ").apply(lambda l: list(map(int, l))).values):
    train_attr_ohot[idx, attr_arr] = 1


# In[12]:


names_arr = labels_df.attribute_name.values
train_df["attribute_names"] = [", ".join(names_arr[arr == 1]) for arr in train_attr_ohot]


# In[13]:


train_df["attr_num"] = train_attr_ohot.sum(axis=1)
train_df["culture_attr_num"] = train_attr_ohot[:, :398].sum(axis=1)
train_df["tag_attr_num"] = train_attr_ohot[:, 398:].sum(axis=1)


# In[14]:


train_df.head()


# In[16]:





# ## number of images by attribute

# In[37]:


# count how many items per attribute and tag
counter = dict()

for k,v in train_df.iterrows():
    for an in v.attribute_names.split(','):
        an = an.strip()
        if an in counter.keys():
            counter[an]+= 1
        else:
            counter[an] = 1


# In[67]:


s = pd.Series(counter, name='item_count')
df_attr_cnt = s.reset_index()
df_attr_cnt = df_attr_cnt.rename(columns={'index':'attribute'})
df_attr_cnt.head()


# In[70]:


df_attr_cnt = df_attr_cnt.sort_values('item_count', ascending=False)
df_attr_cnt.head()


# In[80]:


df_attr_cnt.tail()


# In[79]:


df_attr_cnt.plot(kind='bar',x='attribute', y='item_count', figsize=(200,5))


# ### number of attributes each art has

# In[15]:


train_df.attr_num.value_counts().sort_index()


# ### number of _"culture"_ attributes each art has

# In[ ]:


train_df.culture_attr_num.value_counts().sort_index()


# ### number of _"tag"_ attributes each art has

# In[ ]:


train_df.tag_attr_num.value_counts().sort_index()


# ### plots

# In[ ]:


fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(3,1,1)
sns.countplot(train_df.attr_num,)
ax1.set_title("number of attributes each art has")
ax2 = fig.add_subplot(3,1,2,)
sns.countplot(train_df.culture_attr_num, ax=ax2)
ax2.set_title("number of 'culture' attributes each art has")
ax3 = fig.add_subplot(3,1,3,)
ax3.set_title("number of 'tag' attributes each art has")
sns.countplot(train_df.tag_attr_num, ax=ax3)


# In[73]:


pd.pivot_table(
    train_df, index='culture_attr_num', columns='tag_attr_num', values='attr_num',
    aggfunc=len)


# In[74]:


train_df.culture_attr_num.value_counts(normalize=True).sort_index()


# In[75]:


train_df.tag_attr_num.value_counts(normalize=True).sort_index()


# There is difference between the distribution of number of culture attributes and one of tag attributes.  
# 
# The number of _culture_ attributes 99% of arts have is 0 or 1 or 2, moreover, **80% is 1**.  
# On the other hands, the number of _tag_ attributes shows a **gentler** slope from 1 to 5. Very few of arts have no tag attribute.
# 
# I think these observations may be useful for deciding thresholds ofclassifiers. 
# 
# 
# Next, I show the arts which have many culuture or tag attributes. 

# In[76]:


train_df.sort_values(by="culture_attr_num", ascending=False).head(15)


# In[77]:


train_df.sort_values(by="tag_attr_num", ascending=False).head(15)


# It is difficult for me to find somthing from these tables. Let's show images of arts in these tables.

# In[ ]:


from PIL import Image


# ### arts with many _culture_ attributes

# In[ ]:


fig = plt.figure(figsize=(5 * 5, 5 * 6))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df.sort_values(by="culture_attr_num", ascending=False)[["id", "attribute_names"]].values[:15]):
    ax = fig.add_subplot(5, 3, i // 3 * 3 + i % 3 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# ### arts with many _tag_ attirbutes

# In[ ]:


fig = plt.figure(figsize=(5 * 6, 5 * 5))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df.sort_values(by="tag_attr_num", ascending=False)[["id", "attribute_names"]].values[:15]):
    ax = fig.add_subplot(5, 3, i // 3 * 3 + i % 3 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# Since I have poor knowledge of art, cannot validate _culture_ attributes.
# 
# How about _tag_ attributes? They are relatively interpretable, but may be splitable into some types:
# 
# * objects painted (or carved) on arts: animals, humans, places, ...
# * type of arts: 'cups', 'coat of arms', 'textiles'. 'dishes', 'vines'...
# * special situations: 'nativity', 'last jugement', 'crucifixion',...
# * actions: 'hourse riding', 'reading', 'suffering', ...
# * ...
# 
# Therefore, I think it may be useful for classification to consider type of _tag_ attribute.
# <br>
# <br>
# <br>
# With respect to number, pictures tend to have more _tag_ attributes because of painted objects on them.  
# I have one assumption that number of tag attributes depends on type of arts. Then, check several examples.

# ### arts with 1 _tag_ attribute

# In[ ]:


fig = plt.figure(figsize=(5 * 8, 5 * 7))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 1][["id", "attribute_names"]].values[:49]):
    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# ### arts with 2 _tag_ attributes

# In[ ]:


fig = plt.figure(figsize=(5 * 8, 5 * 7))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 2][["id", "attribute_names"]].values[:49]):
    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# ### arts with 3 _tag_ attributes

# In[ ]:


fig = plt.figure(figsize=(5 * 8, 5 * 7))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 3][["id", "attribute_names"]].values[:49]):
    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# ### arts with 4 _tag_ attributes

# In[ ]:


fig = plt.figure(figsize=(5 * 8, 5 * 7))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 4][["id", "attribute_names"]].values[:49]):
    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# ### arts with 5 _tag_ attributes

# In[ ]:


fig = plt.figure(figsize=(5 * 8, 5 * 7))
fig.subplots_adjust(wspace=0.6, hspace=0.6)
for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 5][["id", "attribute_names"]].values[:49]):
    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)
    im = Image.open("../input/train/{}.png".format(art_id))
    ax.imshow(im)
    im.close()
    attr_split = attr_names.split(", ")
    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))
    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))
    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))


# It seems that more _tag_ attributes arts have, more complex they are.
# 
# Most of arts with one _tag_ attributes are single objects such as ornaments.  
# In contrast, most of arts with five ones are pictures or objects with complex design.
# 
# Maybe, we can predict number of _tag_ attributes by **_complexity_** of arts ?
