#!/usr/bin/env python
# coding: utf-8

# BDML Group 1 Casptone

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv(
    '../input/inclusive-images-challenge/tuning_labels.csv', header=None, names=['img_id', 'labels'])
description = pd.read_csv('../input/inclusive-images-challenge/class-descriptions.csv')

label_lookup = {label: descr for label, descr in zip(
    description.label_code.values, description.description.values)}


# In[ ]:


len(label_lookup)


# In[ ]:


import csv

label_count = {}
with open('../input/inclusive-images-challenge/tuning_labels.csv') as f:
    csv_rdr = csv.reader(f)
    for row in csv_rdr:
        inst_labels = row[1].split()
        label_count = {**label_count, **{l: label_count.get(l, 0) + 1 for l in inst_labels}}


# In[ ]:


inst_labels


# In[ ]:


top_label_name, top_label_count = zip(*[
    l for l in sorted(label_count.items(), reverse=True, key=lambda l: l[1])[:10]])
label_count_df = pd.DataFrame(data={'label': list(map(label_lookup.get, top_label_name)), 'count': top_label_count})
sns.set(rc={'figure.figsize':(12, 18)})
ax = sns.barplot(data=label_count_df, y='label', x='count')


# In[ ]:


print (label_count_df)


# In[ ]:


df = pd.read_csv('../input/inclusive-images-challenge/tuning_labels.csv', header=None, 
                 names=['img_id', 'labels'])


# In[ ]:


#import shutil, sys 
#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


# In[ ]:


description = pd.read_csv('../input/inclusive-images-challenge/class-descriptions.csv')


# In[ ]:


d={}
for i,j in zip(description.label_code.values, description.description.values):
    d[i]=j


# In[ ]:


count = pd.DataFrame(df['labels'].str.split().apply(lambda x:len(x)))
sns.countplot(data=count,x='labels')
plt.title('number of labels')


# In[ ]:


# An image has 9 or fewer labels. 


# In[ ]:




