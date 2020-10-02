#!/usr/bin/env python
# coding: utf-8

# # Control Wells Present on All Plates
# 
# An interesting factor in this competition is the presence of control experiments in each plate. Experimental controls provide a way of calibrating the outcome of an experiment against expected variance between experiments. The goal of this competition is to create a model that is robust to a high level of variance between plates. I think making intelligent use of controls will be an important factor in creating a high accuracy model.
# 
# There's one little wrinkle though. Not every control is present in every plate. The code in this notebook finds the control experiments that are common across every plate in both the training and testing set. The code also gathers image channel stats for the common control wells for each plate, which could be a useful feature.

# In[ ]:


import pandas as pd
import os
from pathlib import Path
import numpy as np


# In[ ]:


path = Path('../input')


# In[ ]:


def _load_dataset(base_path, dataset, include_controls=True):
    df =  pd.read_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = pd.read_csv(
            os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res


# In[ ]:


def combine_metadata(base_path=path,
                     include_controls=True):
    df = pd.concat(
        [
            _load_dataset(
                base_path, dataset, include_controls=include_controls)
            for dataset in ['test', 'train']
        ],
        sort=True)
    return df


# In[ ]:


md = combine_metadata()


# In[ ]:


neg_ctrls = []
pos_ctrls = []
for experiment in md.experiment.unique():
    for plate in range(1, 5):
        negs = set(md[(md.experiment == experiment) & 
                      (md.plate == plate) & (md.well_type == 'negative_control')].sirna)
        pos = set(md[(md.experiment == experiment) & 
                      (md.plate == plate) & (md.well_type == 'positive_control')].sirna)
        neg_ctrls.append(negs)
        pos_ctrls.append(pos)


# In[ ]:


positive_controls = set.intersection(*pos_ctrls)
negative_controls = set.intersection(*neg_ctrls)


# Positive control sirnas present in every plate. Note that the same sirna control might be present in a different well between two plates. The important thing to keep track of is the sirna itself.

# In[ ]:


positive_controls


# The same negative control (no sirna) is present in all plates

# In[ ]:


negative_controls


# Overall there are 18 controls (positive + negative) present in all plates

# In[ ]:


controls = list(negative_controls) + list(positive_controls)


# In[ ]:





# In[ ]:


stats = pd.read_csv(path/'pixel_stats.csv')


# In[ ]:


md.reset_index(inplace=True)
md['code_site'] = md.apply(lambda row: row['id_code'] + '_' + str(row['site']), axis=1)
stats['code_site'] = stats.apply(lambda row: row['id_code'] + '_' + str(row['site']), axis=1)

merged = pd.merge(md, stats[['mean', 'std', 'median', 'min', 'max', 'code_site']], on='code_site')


# In[ ]:





# Control well stats for each plate are saved as a numpy array

# In[ ]:


control_stats = []
control_ids = []

for experiment in merged.experiment.unique():
    for plate in range(1, 5):
        plate_data = []
        # looping through controls for each plate is slow, but ensures all controls are stored in the same order
        for rna in controls:
            data = merged[(merged.experiment == experiment) &
                          (merged.plate == plate) &
                          (merged.sirna == rna)][['mean', 'std', 'median', 'min', 'max']].values
            
            plate_data.append(data)
            
        data = np.concatenate(plate_data)
        control_stats.append(data)
        control_ids.append(experiment + '_' + str(plate))


# The data is stored in a 216x5 matrix for each plate. (18 controls/plate * 2 sites/control * 6 channels/site image = 216)

# In[ ]:


control_stats[0].shape


# In[ ]:


control_ids[0]


# In[ ]:




