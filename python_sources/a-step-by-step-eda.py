#!/usr/bin/env python
# coding: utf-8

# ## In this step by step EDA, you will find: 
# 
# * **0. Data dummification before further preprocessing**
#     A way to dummify data in order to get them useable 
#     
# * **I. Number and type of labels**
#     A brief look at the number of labels by image and their type
#     
# * **II. What are the most frequent labels?** 
#     A brief look at the labels repartition
#     
# * **III. Label coexistence**  
#     A brief study on labels correlation
#     
# * **IV. Random image displayer**
#     A simple function randomly displaying an image of a given label together with its shape and all its tags  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import re
from collections import Counter

# Ploting
import matplotlib.pyplot as plt
import seaborn as sns 

plt.rcParams['figure.figsize'] = (30,30)
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

def append_ext(fn):
    return fn+".png"

def remove_ext(fn):
    return fn[:-4]


# In[ ]:


labels_list = pd.read_csv('../input/labels.csv')
labels = pd.read_csv("../input/train.csv")
test_submission = pd.read_csv('../input/sample_submission.csv')
labels['attribute_ids'] = labels['attribute_ids'].str.split(" ")
labels['id'] = labels['id'].apply(append_ext)
test_submission['id'] =test_submission['id'].apply(append_ext)
labels_list.head()
labels = labels


# ### 0. Data dummification before further preprocessing

# In[ ]:


start = datetime.datetime.now()

labels_dummified = pd.DataFrame(columns=labels_list['attribute_id'])
d_list = []
for index, row in labels.iterrows():
    for value in row['attribute_ids']:
        d_list.append({'name':row['id'], 
                       'value':value})
labels_dummified = labels_dummified.append(d_list, ignore_index=True)
labels_dummified = labels_dummified.groupby('name')['value'].value_counts()
labels_dummified = labels_dummified.unstack(level=-1).fillna(0)
labels_dummified = labels_dummified[[str(y) for y in sorted([int(x) for x in labels_dummified.columns])]]
labels_dummified.columns = labels_list['attribute_name']
end = datetime.datetime.now()
print("Elapsed time:",end-start)
labels_dummified.head()


# ### I. Number and type of labels
# 
# How many labels do the images have?  

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
ax.hist(labels_dummified.sum(axis=1),bins=10)
plt.show()


# All the object have at least one label, and usually two and more. Having more than 7 labels is pretty rare. 

# What are the types of labels and how frequent are they?

# In[ ]:


n_labels = labels_dummified.sum()
Counter([re.match('([a-z]+)::\w+',x)[1] for x in n_labels.index])


# There are only 2 type of labels: 
# - the culture of the object
# - and the tag, ie its content
# 
# 2/3 of the labels are related to tags, and 398 different cultures are present. Now it's interesting to see how often images do have a tag and/or a culture label. 

# How frequently images have culture & tag labels? 

# In[ ]:


culture_columns = [x for x in labels_dummified.columns if x.startswith('culture')]
tag_columns = [x for x in labels_dummified.columns if x.startswith('tag')]

n_culture_labels = labels_dummified[culture_columns].sum(axis=1)
n_tag_labels = labels_dummified[tag_columns].sum(axis=1)


fig, ax = plt.subplots(figsize=(20,10))
ax.hist(n_culture_labels,bins=4,alpha=0.7)
ax.axvline(n_culture_labels.mean())

ax.hist(n_tag_labels,bins=9,alpha=0.7)
ax.axvline(n_tag_labels.mean(),color='orange')

print("Number of images with 0 culture label:",(n_culture_labels == 0 ).sum())

print("Number of images with 0 tag label:",(n_tag_labels == 0 ).sum())

plt.show()


# Tag and culture labels have very different behaviour: 
# - There is almost always at least one tag labels. They are also largely non-exclusive, as the average number of tag labels per image is more than 2. 
# - On the contrary, the culture label are often exclusive, with about 90k images having exactly one culture. It's also pretty common to have an object without identified culture in the dataset: more than 10k image have not culture label. 
# 
# Those differences in behaviour will probably allows specific strategies to take them into account. 

# ### II. What are the most frequent labels?

# In[ ]:


labels_count = labels_dummified.sum().reset_index().sort_values(ascending=False,by=0)
labels_count


# In[ ]:


labels_count.loc[labels_count['attribute_name'].str.startswith('culture')]


# In[ ]:


labels_count.loc[labels_count['attribute_name'].str.startswith('tag')]


# The most frequent labels are either very common contents (men, women, flowers etc.)/type of artwork (portrait, inscription etc.) or cultures famous in art history and cultural production (French, Italian, British, American, Japanese, Chinese, Egyptian etc.). 
# 
# Interestingly, almost all of the less frequent labels are culture ones (subculture, less famous culture, combination of cultures etc.). It may be interesting to see if some culture should not basically be merged, especially when the description seems so accurate that I am not sure that we will find them in the dataset. 

# ### III. Label coexistence 

# First, let's have a look at the coexistence between the 100 most frequent labels

# In[ ]:


most_frequent_labels = labels_dummified[labels_dummified.sum().sort_values(ascending=False)[:100].index]
labels_corr = most_frequent_labels.corr()

fig_dims = (30, 16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(labels_corr,ax=ax)
plt.plot()


# To have a better look at the correlation, let's now take the 200 most frequent tags, and let's see what are the 40 most important correlation between them. 

# In[ ]:


most_frequent_200_labels = labels_dummified[labels_dummified.sum().sort_values(ascending=False)[:200].index] # We take the most frequent labels
labels_corr_200 = most_frequent_200_labels.corr() # We then look at their correlations
largest_corr_200 = pd.DataFrame(np.sort(abs(labels_corr_200).values)[:,-2:-1], columns=['2nd-largest'],index=labels_corr_200.index) # And order by the second largest (absolute) correlation 
largest_corr_40 = largest_corr_200.sort_values(by='2nd-largest',ascending=False).iloc[:40] # And then just take the 40 first.

fig_dims = (30, 16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(most_frequent_200_labels.loc[:,largest_corr_40.index].corr(),ax=ax, vmax=0.7)
plt.plot()


# Looking at the most frequent labels, we can see some main topics:
# - Plants ( tag::flowers, tag::leaves) 
# - American (holywoodian/star?) modern culture (culture::american, tag::actress, tag::portrait, tag::women)
# - Ancient Greece (culture::greek, culture::attic, culture::south_italia)
# - Ancient Egypt (culture::egyptian, tag::scarabs,tag::hyeroglyph)
# - Ancient Mesopotamia (tag::cuneiform, tag::tablets, culture::babylonian) 
# - Water scene (tag::bodies of water, tag::boat)
# - British culture (culture::british, tag::london) 
# - Christian religion (tag:: virgin mary, tag::christ, tag::christian imagery)
# - Landscape (tag:: landscape, tag::tree, tag::houses, tag::mountain)
# - French culture (culture::french, culture::paris) 
# - etc.

# In[ ]:


labels_count.loc[labels_count['attribute_name'].str.startswith('culture')]


# Let's now look at the correlation between the culture labels only

# In[ ]:


culture_labels_dummified = labels_dummified[labels_count.loc[labels_count['attribute_name'].str.startswith('culture')]['attribute_name']]
most_frequent_400_culture_labels = culture_labels_dummified[culture_labels_dummified.sum().sort_values(ascending=False)[:400].index] # We take the 400 most frequent labels
culture_labels_corr = most_frequent_400_culture_labels.corr() #
culture_corr = pd.DataFrame(np.sort(abs(culture_labels_corr).values)[:,-2:-1], columns=['2nd-largest'],index=culture_labels_corr.index)
culture_corr = culture_corr.sort_values(by='2nd-largest',ascending=False).iloc[:40]

fig_dims = (30, 16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(culture_labels_dummified.loc[:,culture_corr.index].corr(),ax=ax, vmax=0.7)
plt.plot()


# Some culture labels are very similar, almost synonyms (london original/after british, augsburg original/after german). Notice that those strong correlations are also probably caused by a relative scarcity of the labels, as we see below.

# In[ ]:


pd.merge(culture_corr,labels_count,left_index=True,right_on='attribute_name') # To see the number of images for those tags


# Let's now look at the correlation between the tag labels only

# In[ ]:


tag_labels_dummified = labels_dummified[labels_count.loc[labels_count['attribute_name'].str.startswith('tag')]['attribute_name']]
most_frequent_400_tag_labels = tag_labels_dummified[tag_labels_dummified.sum().sort_values(ascending=False)[:400].index] # We take the 300 most frequent labels
tag_labels_corr = most_frequent_400_tag_labels.corr() #
tag_corr = pd.DataFrame(np.sort(abs(tag_labels_corr).values)[:,-2:-1], columns=['2nd-largest'],index=tag_labels_corr.index)
tag_corr = tag_corr.sort_values(by='2nd-largest',ascending=False).iloc[:40]

fig_dims = (30, 16)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(tag_labels_dummified.loc[:,tag_corr.index].corr(),ax=ax, vmax=0.7)
plt.plot()

There is less strong correlations for tags than for culture. However, the number of images here are more important - and thuse more useful. 
# In[ ]:


pd.merge(tag_corr,labels_count,left_index=True,right_on='attribute_name') # To see the number ofimages for those tags


# There is somehow less strong tags coexistence than for culture labels. However, some thema are clearly delineated: adam & eve, sport, bouddhism etc. 

# ### IV. Random image displayer

# In[ ]:


def print_next_image(attribute_name="culture::abruzzi",dataset='train'):
    ''' This function generate images having "attribute_name" as label'''
    file = next(att_gen[attribute_name])
    img=plt.imread('../input/'+dataset+'/'+next(att_gen[attribute_name]))
    plt.imshow(img,aspect='auto')
    print("File:",file)
    print("Image shape:",img.shape)
    if dataset == 'train':
        idx_to_name = labels_list.set_index('attribute_id').to_dict()['attribute_name']
        labs = labels.loc[labels['id'] == file,'attribute_ids'].iloc[0]
        print('labels:', labs)
        print('labels names:', [idx_to_name[int(x)] for x in labs])
    plt.show()


# In[ ]:


att_gen= {}
for att in labels_list['attribute_name'][:10]:
    att_gen[att] = (x for x in labels_dummified.loc[labels_dummified[att] > 0].index)


# In[ ]:


print_next_image('culture::akkadian')

