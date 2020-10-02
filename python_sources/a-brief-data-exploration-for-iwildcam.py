#!/usr/bin/env python
# coding: utf-8

# # A Brief Data Exploration
# 
# Summary:
# 1. There are 23 classes but about 9 of them are not in the training set. 
# 2. Highly unbalanced data

# ## The data we obtain

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


sample_submission_csv = pd.read_csv("../input/sample_submission.csv")
sample_submission_csv.head()


# In[ ]:


test_csv = pd.read_csv("../input/test.csv")
test_csv.head()


# In[ ]:


train_csv = pd.read_csv("../input/train.csv")
train_csv.head()


# In[ ]:


print("There are {} trian images and {} test images".format(len(train_csv.file_name.values), len(test_csv.file_name.values)))


# # Classes

# In[ ]:


dictionary = {0:'empty',
              1:'deer',
              2:'moose',
              3:'squirrel',
              4:'rodent',
              5:'small_mammal',
              6:'elk',
              7:'pronghorn_antelope',
              8:'rabbit',
              9:'bighorn_sheep',
              10:'fox',
              11:'coyote',
              12:'black_bear',
              13:'raccoon',
              14:'skunk',
              15:'wolf',
              16:'bobcat',
              17:'cat',
              18:'dog',
              19:'opossum',
              20:'bison',
              21:'mountrain_goat',
              22:'mountain_lion',
             }
dictionary_reverse = dict((i,j) for j,i in dictionary.items())

train_ids = set(train_csv.category_id.values)
print("There are the following classes: {} in train".format(train_ids))


# In[ ]:


train_ids = set(dictionary[item] for item in train_ids)
print("There are the following classes: {} in train".format(train_ids))


# In[ ]:


print("There are some classes missing: {}".format(set(dictionary.values())-train_ids))


# In[ ]:


id_table = pd.read_csv("../input/train.csv")

appearance_dict = dict()
for i in range(23):
    appearance = len([val for val in id_table.category_id.values if val == i])
    appearance_dict[i] = appearance
    
print(appearance_dict)


# In[ ]:


import matplotlib.pyplot as plt

def plot_bar_x(index, values, x_label, y_label, title, y_lim=None):
    plt.bar(index, values)
    plt.xlabel(x_label, fontsize=5)
    plt.ylabel(y_label, fontsize=5)
    plt.xticks(range(len(index)), index, fontsize=5, rotation=90)
    plt.title(title)
    
    if y_lim != None:
        plt.ylim(top=y_lim)
    
    plt.show()


# In[ ]:


plot_bar_x(dictionary.values(), appearance_dict.values(), "Names", "Frequency", "Data Distribution Betwen Classes")


# In[ ]:


plot_bar_x(dictionary.values(), appearance_dict.values(), "Names", "Frequency", "Data Distribution Betwen Classes Without Class 0", y_lim=15000)


# # Heights

# In[ ]:


print("There are only one widths: {}".format(set(train_csv.width.values)))


# In[ ]:


print("But there are three heights: {}".format(set(train_csv.height.values)))

height_dict = dict()
for i in set(train_csv.height.values):
    height = len([val for val in id_table.height.values if val == i])
    height_dict[i] = height
    
print("Here is the distribution: {}".format(height_dict))


# In[ ]:


plot_bar_x(['768','747','748'], height_dict.values(), "Hights", "Frequency", "Data Distribution of Heights")


# ## The Structure of the Public LB
# F1 Score in each class: [0.089, ...]
# 
