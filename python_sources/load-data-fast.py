#!/usr/bin/env python
# coding: utf-8

# # Load data faster
# 
# While doing EDA we load data mulitple time or if we are using doing EDA on our machine its faster to pickle the data which can improve data load time drastically 

# In[ ]:


import os
import os
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data_path = '/kaggle/input/data-science-bowl-2019/'
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world','event_data']


# ### Load data If the pickle file does not exists it will fetch it from csv annd will create an .pkl file. 

# The below code will take time as its first time loading. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntry:\n    train = pd.read_pickle(f'train.pkl')\n    test = pd.read_pickle(f'test.pkl')\n    specs = pd.read_pickle(f'specs.pkl')\n    train_labels = pd.read_pickle(f'train_labels.pkl')\n    sample_submission = pd.read_pickle(f'sample_submission.pkl')\n    print(f'Reading from pkl files')\nexcept (OSError, IOError) as e:\n    train = pd.read_csv(f'{data_path}train.csv')\n    test = pd.read_csv(f'{data_path}test.csv')\n    specs = pd.read_csv(f'{data_path}specs.csv')\n    train_labels = pd.read_csv(f'{data_path}train_labels.csv')\n    sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')\n    \n    # uncomment below when using.   \n#     train.to_pickle(f'train.pkl')\n#     test.to_pickle(f'test.pkl')\n#     specs.to_pickle(f'specs.pkl')\n#     train_labels.to_pickle(f'train_labels.pkl')\n#     sample_submission.to_pickle(f'sample_submission.pkl')\n    print(f'Reading from CSV files')")


# ### Test time to save to .pkl file

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain.to_pickle(f'train.pkl')\ntest.to_pickle(f'test.pkl')\nspecs.to_pickle(f'specs.pkl')\ntrain_labels.to_pickle(f'train_labels.pkl')\nsample_submission.to_pickle(f'sample_submission.pkl')\nprint(f'Reading from CSV files')")


# Below code is for demo purpose just to check time perfomance. 

# ### Test time to load from .pkl file

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_pickle(f'train.pkl')\ntest = pd.read_pickle(f'test.pkl')\nspecs = pd.read_pickle(f'specs.pkl')\ntrain_labels = pd.read_pickle(f'train_labels.pkl')\nsample_submission = pd.read_pickle(f'sample_submission.pkl')")


# ### Parquet File system

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntry:\n    train = pd.read_parquet(f'train.parquet')\n    test = pd.read_parquet(f'test.parquet')\n    specs = pd.read_parquet(f'specs.parquet')\n    train_labels = pd.read_parquet(f'train_labels.parquet')\n    sample_submission = pd.read_parquet(f'sample_submission.parquet')\n    print(f'Reading from pkl files')\nexcept (OSError, IOError) as e:\n    train = pd.read_csv(f'{data_path}train.csv')\n    test = pd.read_csv(f'{data_path}test.csv')\n    specs = pd.read_csv(f'{data_path}specs.csv')\n    train_labels = pd.read_csv(f'{data_path}train_labels.csv')\n    sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')\n    \n    # uncomment below when using.   \n#     train.to_parquet(f'train.parquet')\n#     test.to_parquet(f'test.parquet')\n#     specs.to_parquet(f'specs.parquet')\n#     train_labels.to_parquet(f'train_labels.parquet')\n#     sample_submission.to_parquet(f'sample_submission.parquet')\n    print(f'Reading from CSV files')")


# ### Saving time to parquet

# In[ ]:



get_ipython().run_cell_magic('time', '', "\ntrain.to_parquet(f'train.parquet')\ntest.to_parquet(f'test.parquet')\nspecs.to_parquet(f'specs.parquet')\ntrain_labels.to_parquet(f'train_labels.parquet')\nsample_submission.to_parquet(f'sample_submission.parquet')\nprint(f'Reading from parquet files')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_parquet(f'train.parquet')\ntest = pd.read_parquet(f'test.parquet')\nspecs = pd.read_parquet(f'specs.parquet')\ntrain_labels = pd.read_parquet(f'train_labels.parquet')\nsample_submission = pd.read_parquet(f'sample_submission.parquet')\nprint(f'Reading from pkl files')")


# In[ ]:


# train = train[keep_cols]
# test = test[keep_cols]


# In[ ]:


get_ipython().system('ls -ltrh')


# Pickle is 2.8 GB for train.csv and parquet took 546 MB space. **

# ### If you like this please upvote

# *...inprogress* 
# 

# Reference: 
# - https://www.kaggle.com/tdobson/30x-speedup-on-io
