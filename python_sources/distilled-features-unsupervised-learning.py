#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# The following kernal demostraits how to derive  distilled features, then further condense your feature space via unsuperived learning. I have a lot to add to the code, let me know what you think! I just started the kernel Febuary 4th, If you have any questions or constructive feedback please email me directly:
# 
# email: **ajg1444@rit.edu**
# 
# 
# 
# -- Alex G.
# 
# ***Distilling Features:***
# 
# The first step is to distill the *name, category_name, item_description*. To uderstand what distill means let me walk you through how to distill the *category_name*. 
# 
# We begin by breaking each row element of the dataframe apart such that each element in the new list is an
# * Individual lower case word
# * Has no numbers or special characters

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import re
import time as tmi
from collections import Counter
from functools import reduce
import matplotlib.pyplot as plt
import pprint as pretty
from IPython.display import display


# In[ ]:



train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
train.head(2)


# In[ ]:


# clean list data:
category_names = list(map(lambda x: x.lower(), list(map(str, train['category_name'].tolist()))))
train_id = train['train_id'].tolist()

# clean up category strings:
for i in range(len(category_names)):
    category_names[i] = str(category_names[i].split('/')).split()
    category_names[i] = map(lambda x: re.sub('[^a-z]+', '',x), category_names[i])
    category_names[i] = list(Counter(list(filter(None, map(str.strip, category_names[i])))).keys())


# In[ ]:


pd.set_option('max_colwidth', 60)
temp_frame = pd.DataFrame({"category_names":[category_names[0],category_names[1]]})
temp_frame.head()


# Now since every word is lowercase and special characters have been removed we may now create a dictionary which allows us to look up all records (or train_ids) which has the key word in it's distilled catigory list. 
# 
# For example
# 1. if we wish to search for 'men' the search will return, [0, .... ]
# 2. if we wish to search for 'electronics' the search will return [1, .... ]
# 

# In[ ]:



cat_hashmap ={}
for i in range(len(train_id)):
    for cat in category_names[i]:
        if not cat in cat_hashmap:
            cat_hashmap.update({cat:[train_id[i]]})
        else:
            cat_hashmap[cat].append(train_id[i])


# Great! we have compiled the dictionary now let us put it to the test. Before we do let us set our data frame index to train_id. After let us pull all train_id which have the keyword 'tops' in the distilled category names.

# In[ ]:


train = train.set_index('train_id')


# In[ ]:


view_1 = "tops"
view_1 = cat_hashmap[view_1]
pd.DataFrame({"train_id_with_word_tops":view_1,"category_name": train["category_name"].loc[view_1].values}).head(5)


# Before we further distill the words let us first analize how we may apply our distilled feature by using set theory. Let us analize the price difference between women and men's sports wear. To do this we must analize when 
# 
# (men and sports intersect) and (woman and sports intersect)  
# 
# Note the results have been filtered such that all results with a standard diviation of 3 have been excluded from the finding

# In[ ]:


# set train id as the primary key
name = "price"       # value which you are  analizing
view_1 = "men"       # subset_1
view_2 = "women"     # subset_2
intersect = "sports" # intersection of both subset_1 and subset_2

view_1  = cat_hashmap[view_1]
view_2  = cat_hashmap[view_2]
intersect = cat_hashmap[intersect]

# obtain intersection of view 1 and 2 & the compare list
view_1 = list(set(view_1) & set(intersect))
view_2 = list(set(view_2) & set(intersect))

# exctract data
plot_1 = train[name].loc[view_1].values
plot_2 = train[name].loc[view_2].values


# In[ ]:


# exctract data
Z_Thresh = 3

plot_1 = plot_1[abs(np.mean(plot_1)-plot_1)/np.std(plot_1) < Z_Thresh]
plot_2 = plot_2[abs(np.mean(plot_2)-plot_2)/np.std(plot_2) < Z_Thresh]


# In[ ]:


# pull training data
Bins  = 20
views = list([view_1,view_2])


fig=plt.figure(figsize=(18, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.hist(plot_1, alpha=0.5, normed=True, bins=Bins,label='men')
plt.hist(plot_2, normed=True,bins=Bins, alpha=0.5,label='women')
plt.ylabel('Probability',fontsize=15);
plt.xlabel('price',fontsize=15);
plt.legend(loc='upper right',fontsize=15)
plt.show();


# Seems there is no difference between the pricing of men's and women's sports attare here. Not the best example but I hope you get the picture. 
# 
# To further distill our category names we may throw low freqency names out of our universe. To do this let us create a data frame with two columns category_name & frequency. Roughly 581 words of our 1,008 have a freqency of occured less than 581 times out of our 1,482,535 records thats less than 0.03 percent!

# In[ ]:


Data = dict(Counter(map(lambda x: re.sub('[^a-z]+', '',x),(str(category_names)).split())))
category_df = pd.DataFrame({"category_name":list(Data.keys()),"frequency":list(Data.values())})
category_df = category_df.sort_values("frequency",ascending=False)


# In[ ]:


fig=plt.figure(figsize=(18, 5), dpi= 80, facecolor='w', edgecolor='k')
plot_1 = category_df["frequency"].values
plot_1 = plot_1[plot_1  < 500]
plt.hist(plot_1, alpha=0.5, normed=False, bins=10,label='men')
plt.ylabel('Frequency',fontsize=15);
plt.xlabel('Category Name Frequency',fontsize=15);
plt.show();


# The next step is to use unsuperviced learning to first transform the problem into a multiclass problem, then use the results aid in building your model.
