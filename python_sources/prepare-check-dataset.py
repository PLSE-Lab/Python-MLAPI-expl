#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from uuid import uuid4


# In[ ]:


get_ipython().system("mkdir 'birdcall-check'")
get_ipython().system("mkdir 'birdcall-check/test_audio'")


# In[ ]:


df_train = pd.read_csv('../input/birdsong-recognition/train.csv')


# In[ ]:


def copy_audio(ebird_code, filename, audio_id):
    get_ipython().system("cp '../input/birdsong-recognition/train_audio/{ebird_code}/{filename}' 'birdcall-check/test_audio/{audio_id}.mp3'")

data_check = []
for _, row in df_train[:5].iterrows():
    site = 'site_1'
    audio_id = str(uuid4()).replace('-', '')
    copy_audio(row['ebird_code'], row['filename'], audio_id)
    for i in range(row['duration']//5):
        seconds = int((i + 1)*5)
        row_id = f'{site}_{audio_id}_{seconds}'
        data_check.append({
            'site': site,
            'row_id': row_id,
            'seconds': seconds,
            'audio_id': audio_id
        })
        
for _, row in df_train[5:10].iterrows():
    site = 'site_2'
    audio_id = str(uuid4()).replace('-', '')
    copy_audio(row['ebird_code'], row['filename'], audio_id)
    for i in range(row['duration']//5):
        seconds = int((i + 1)*5)
        row_id = f'{site}_{audio_id}_{seconds}'
        data_check.append({
            'site': site,
            'row_id': row_id,
            'seconds': seconds,
            'audio_id': audio_id
        })
        
for _, row in df_train[10:15].iterrows():
    site = 'site_3'
    audio_id = str(uuid4()).replace('-', '')
    copy_audio(row['ebird_code'], row['filename'], audio_id)
    seconds = None
    row_id = f'{site}_{audio_id}'
    data_check.append({
        'site': site,
        'row_id': row_id,
        'seconds': seconds,
        'audio_id': audio_id
    })


data_check = pd.DataFrame(data_check)
data_check.to_csv('birdcall-check/test.csv', index=False)
data_check.head()

