#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm
import gensim


# ### Reformat dataset to train Word2Vec model

# In[ ]:


with open('vk_music_lists.csv', 'w') as f:
    for chunk in pd.read_csv('../input/vk_dataset_anon.csv', 
                                       iterator=True, 
                                       chunksize=1000000,
                                       header=None,
                                       lineterminator='\n',
                                       error_bad_lines=False,
                                       # remove this to use whole dataset
                                       nrows=5000000,
                                       ##################################
                                      ):
        chunk.columns = ['user', 'song', 'band']
        chunk['band'] = chunk['band'].astype(str)
        chunk = chunk.groupby('user')['band'].apply(','.join).reset_index()

        for row in chunk.iterrows():
            f.write(row[1]['band'] + '\n')


# ### Train basic model

# In[ ]:


class TextToW2V(object):
    def __init__(self, file_path):
        self.file_path = file_path


    def __iter__(self):
        for line in open(self.file_path, 'r'):
            yield line.lower().split(',')[::-1]  # reverse order (make old -> new)

music_collections = TextToW2V('vk_music_lists.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'estimator = gensim.models.Word2Vec(music_collections,\n                                   window=15,\n                                   min_count=30,\n                                   sg=1,\n                                   workers=4,\n                                   iter=10,\n                                   ns_exponent=0.8,\n                                  )')


# ### Sanity check

# In[ ]:


user_music = ['led zeppelin', 'Nirvana', 'Pink Floyd']
user_music = [m.lower().strip() for m in user_music]
predicted = estimator.predict_output_word(user_music)

[a[0] for a in predicted if a[0] not in user_music]


# In[ ]:




