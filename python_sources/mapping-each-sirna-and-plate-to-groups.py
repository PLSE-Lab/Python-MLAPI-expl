#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from tqdm import tqdm_notebook as bar

# Any results you write to the current directory are saved as output.
train_meta = pd.read_csv('../input/train.csv')
test_meta = pd.read_csv('../input/test.csv')


# # Mapping each siRNA and plate to 'groups'.
# 
# Thanks to [@zaharchikishev's kernel](https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak/notebook), we've met another milestone for this competition.
# 
# In short, in this competition, siRNAs are treated on the plate in a group-wise manner, 
# 
# which means that we can split the whole problem predicting one from 1108 siRNAs into 4 subproblems, each of them predicting one from 1108 / 4 = 277 siRNAs!
# 
# In other words, there are 'four groups of siRNAs' (say Group 1, 2, 3 and 4), and within an experiment, each of the 'four groups of siRNAs' will have one-to-one matching to 'four plates of experiments'.
# 
# In theory, there will be 24 types of such one-to-one matchings, but it seems that only four of 24 possible one-to-one mappings were used in this competition (3 shown in training set, one more in test set, according to [@zaharchikishev's kernel](https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak/notebook)).
# 
# So, in this kernel, I will simply focus on producing sirna-to-groups and plate-to-groups mapping table.

# In[ ]:


def get_sirna_set(experiment, plate):
    return set(train_meta[(train_meta.experiment == experiment)  & (train_meta.plate == plate)].sirna.values)


# In[ ]:


# Create template siRNA groups, each of which contains 277 siRNAs.
template_sirna_groups = [get_sirna_set('HEPG2-01', plate) for plate in range(1, 5)]

for exp in train_meta.experiment.unique():
    for plate in range(1, 5):
        sirna_set = get_sirna_set(exp, plate)
        
        most_similar_group, max_similarity = None, -1
        for i, template in enumerate(template_sirna_groups):
            similarity = len(sirna_set & template)
            if similarity > max_similarity:
                most_similar_group, max_similarity = i, similarity
            
        if len(template_sirna_groups[most_similar_group]) != 277:
            # Try to expand our template siRNA groups.
            template_sirna_groups[most_similar_group] |= sirna_set
            
for template in template_sirna_groups:
    print(len(template))


# In[ ]:


def get_sirna_group(experiment, plate, template_sirna_groups=template_sirna_groups):
    sirna_set = get_sirna_set(experiment, plate)
    
    for i, group in enumerate(template_sirna_groups, 1):
        if sirna_set.issubset(group):
            return i


# In[ ]:


sirna_group = {
    'sirna': [],
    'group': [],
}

for i, template in enumerate(template_sirna_groups, 1):
    for sirna in template:
        sirna_group['sirna'].append(sirna)
        sirna_group['group'].append(i)

sirna_group = pd.DataFrame(sirna_group).sort_values('sirna').reset_index()[['sirna', 'group']]


# In[ ]:


sirna_group.shape


# In[ ]:


sirna_group.head(3)


# In[ ]:


sirna_group.to_csv('sirna_groups.csv', index=False)


# In[ ]:


experiment_plate_group = {
    'experiment': [],
    'plate': [],
    'group': [],
}

for experiment in train_meta.experiment.unique():
    for plate in range(1, 5):
        group = get_sirna_group(experiment, plate)
        
        experiment_plate_group['experiment'].append(experiment)
        experiment_plate_group['plate'].append(plate)
        experiment_plate_group['group'].append(group)

experiment_plate_group = pd.DataFrame(experiment_plate_group)


# In[ ]:


experiment_plate_group.to_csv('plate_groups.csv', index=False)


# Let's see how many siRNA Group-to-plate mappings were used.

# In[ ]:


mapping_types = []

for experiment in train_meta.experiment.unique():
    groups = experiment_plate_group[experiment_plate_group.experiment == experiment].group.values
    mapping_types.append(''.join(map(str, groups)))
    print(experiment, groups)
    
print('\nThere are %d unique sirna group-to-plate mapping types in total.' % len(set(mapping_types)))


# Hope these tables help!
