#!/usr/bin/env python
# coding: utf-8

# There's been some discussion around dealing with PlayerTrackData.csv and it's strain on system memory. As some have pointed out, one solution is to use Kaggle kernels. Personally I don't like to develop with a Jupyter notebook, especially a remote one. I'd much rather work on my local machine and use something like VS Code with the python extension and all the other goodies. 
# 
# Today I finally got into the player track data. My desktop with 24<s>MB</s>GB RAM could barely handle the file straight up. So I used Pandas chunk feature along with my MemReducer utility script. There are many versions of memory reducers out there, but this one is more effective than most. It's actually pretty simple - you can find it [here](https://www.kaggle.com/jpmiller/skmem).
# 
# Also, for this data I prefer to have the Players, Games, and Plays separated vs. the hyphenated thing (makes it easier for grouping and filtering plus gives us memory-efficient integers instead of silly strings). Here's what I'm doing:

# In[ ]:


import sys
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
from tqdm.notebook import tqdm

import skmem #utility script


# In[ ]:


csize = 4_000_000 #set this to fit your situation
chunker = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv',
                      chunksize=csize)
track_list = []
mr = skmem.MemReducer()
for chunk in tqdm(chunker, total = int(80_000_000/csize)):
    chunk['PlayKey'] = chunk.PlayKey.fillna('0-0-0')
    id_array = chunk.PlayKey.str.split('-', expand=True).to_numpy()
    chunk['PlayerKey'] = id_array[:,0]
    chunk['GameID'] = id_array[:,1]
    chunk['PlayKey'] = id_array[:,2]
    chunk['event'] = chunk.event.fillna('none')
    floaters = chunk.select_dtypes('float').columns.tolist()
    chunk = mr.fit_transform(chunk, float_cols=floaters) #float downcast is optional
    track_list.append(chunk)


# In[ ]:


tracks = pd.concat(track_list)
tracks['event'] = tracks.event.astype('category') #retype after concat
col_order = [9,10,0,1,2,3,4,5,7,6,8]
tracks = tracks[[tracks.columns[idx] for idx in col_order]]
display(tracks.dtypes, tracks.head())


# In[ ]:


tracks.to_parquet('InjuryRecord.parq')


# You can see from the output that the memory footprint is reduced by a factor of 10. Saving as parquet is also nice. It preserves data types and is very efficient. Just make sure you have fastparquet and python-snappy installed in your environment.
# 
# And/or you can download the parquet version here and get going - good luck with the challenge!
