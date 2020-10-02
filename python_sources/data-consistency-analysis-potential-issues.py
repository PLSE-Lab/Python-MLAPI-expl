#!/usr/bin/env python
# coding: utf-8

# # Data consistency analysis
# ## Motivation
# This kernel aims to explore the dataset and to find out if there are any issues with data consistency.
# 
# ## Assumptions
# Given that the data is not well described, we're going to have to make assumptions about the data format:
# 1. Each unit_id corresponds to a row in the results file; therefore, it probably corresponds to a unit of work with a measureable output: an instance of one of the tasks executed by a worker.
# 
# If this assumption is incorrect, the resulting analysis will also be incorrect. This kernel will be updated as more information about the dataset becomes available.
# 
# # Goals
# We are going to check two things in this kernel:
# 1. Are the unit ID values in each task's results file consistent with the mapping between unit_id and task_id fields in each of the activity files?
# 1. Do entries for each unit ID exist in all of the activity files or are some missing?
# 
# # Conclusions
# Unfortunately, the results show that there are  not a consistent mapping between unit_id and task_id fields for the folowing unit_ids: 862530894, 866629802, 861948358.
# 
# In addition, mouse and/or keyboard data is missing for over 24% of the unit_id values.
# 
# Either the assumption about the significance of the unit_id field is incorrect or there are problems with data consistency. In either case the dataset needs to be better described before it is useful.[](http://)

# First we import the required packages, load the data and combine it into a single pandas Dataframe.

# In[ ]:


import os
import pandas as pd
import numpy as np

# Load and combine the keyboard, mouse and tab tables
raw_path = '../input/'
kb = pd.DataFrame.from_csv(os.path.join(raw_path, 'activity_keyboard.csv'))
ms = pd.DataFrame.from_csv(os.path.join(raw_path, 'activity_mouse.csv'))
tb = pd.DataFrame.from_csv(os.path.join(raw_path, 'activity_tab.csv'))

data = pd.concat([kb, ms, tb], join='outer')
kb.reset_index(inplace=True)
ms.reset_index(inplace=True)
tb.reset_index(inplace=True)
data.reset_index(inplace=True)

data.head()


# # Check for consistent mapping between unit_id and task_id values
# ## Search the activity files for unit IDs which are associated with multiple task IDs

# In[ ]:


# Check that each unit ID only corresponds to a single task ID
duplicates = pd.DataFrame()
task_id_lookup = {}
for unit_id in data['unit_id'].unique():
    task_id = data.loc[data['unit_id'] == unit_id, 'task_id'].unique()
    if len(task_id) == 1:
        task_id = task_id[0]
    else:
        duplicates = duplicates.append({'unit_id': unit_id,
                                        'duplicated_across_task_ids': task_id},
                                       ignore_index=True)
if not duplicates.empty:
    duplicates.set_index('unit_id', inplace=True)
    duplicate_ids = set([int(i) for i in duplicates.index.unique().values])
    print('Unit IDs ', duplicate_ids, 'are associated with multiple task IDs and '
                                      'should be removed from the activity dataset.')
duplicates.head()


# ## Find unit ID values in the activity files which are missing from the corresponding task_id results file.

# In[ ]:


# Remove records where the unit_id is not found in the results
# file corresponding to the task_id
missing = pd.DataFrame()
for task_id in data['task_id'].unique():
    results = pd.DataFrame.from_csv(os.path.join(raw_path, 'results_' + str(task_id) + '.csv'))
    results.reset_index(inplace=True)
    unit_id = data.loc[data[(data['task_id'] == task_id) &
                            ~data['unit_id'].isin(results['X_unit_id'].unique())].index,
                       'unit_id']
    if not unit_id.empty:
        missing = missing.append({'task_id': task_id, 
                                  'missing_unit_ids': unit_id.unique()}, 
                                 ignore_index=True)
    # data.drop(data[(data['task_id'] == task_id) & 
    #                ~data['unit_id'].isin(results['X_unit_id'].unique())].index,
    #           inplace=True)
if not missing.empty:
    missing.set_index('task_id', inplace=True)
    missing_ids = set([i for j in missing['missing_unit_ids'] for i in j])
    print('Unit IDs', missing_ids, 'are missing from their corresponding task ID results files and should be removed from the activity dataset.')
missing.head()


# # Identify unit IDs which are not present in all of the activity files

# In[ ]:


# Initialise empty lists to record unit IDs with missing data
key_missing = []
mouse_missing = []
tab_missing = []

# Identify unit_id values which are missing keyboard or mouse data
for unit_id in data['unit_id'].unique():
    if kb[(kb['unit_id'] == unit_id)].empty:
        key_missing.append(unit_id)    
    if ms[(ms['unit_id'] == unit_id)].empty:
        mouse_missing.append(unit_id)        
    if tb[(tb['unit_id'] == unit_id)].empty:
        tab_missing.append(unit_id)
    
print('The dataset contains', len(data['unit_id'].unique()),
      'unit IDs across all activity files.')
print(len(key_missing), 'are missing keyboard data')
print(len(mouse_missing), 'are missing mouse data')
print(len(tab_missing), 'are missing tab data')
print(len(set(key_missing).intersection(set(mouse_missing))),
      'are missing both keyboard data and mouse data')
print('This leaves only ',
      len(data['unit_id'].unique()) - len(set(key_missing).union(set(mouse_missing))),
      'assignments with all data included')

