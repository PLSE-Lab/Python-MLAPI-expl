#!/usr/bin/env python
# coding: utf-8

# ![](https://resources.tidal.com/images/76e0cb4c/bde6/4fca/983b/d5e4891b63dc/640x640.jpg)
# - [1. Basic Setup](#1)
#   -[1.1 Loading Libraries and Dataframes](#1-1)

# <a name='1'></a>
# # 1. Basic Setup

# <a name='1-1'></a>
# ## 1.1  Loading Libraries and Dataframes

# In[ ]:


import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import IPython.display as ipydisplay 


# In[ ]:


PATH_AUDIO = '../input/birdsong-recognition/train_audio/'
train_df = pd.read_csv('../input/birdsong-recognition/train.csv')
test_df = pd.read_csv('../input/birdsong-recognition/test.csv')
train_df.head()


# It seems that most of the data is recorded from North America (USA, CA, MEX)

# In[ ]:


plt.figure(figsize=(10, 5))
train_df.country.value_counts()[:10].plot.bar()
plt.title('Top 10 Countries where data is recorded')
plt.grid('True')
plt.xlabel('Courntries')
plt.ylabel('No of audio samples taken')
plt.show()


# In[ ]:


ebird_name_map = train_df[['ebird_code','species', 'primary_label']].drop_duplicates()
no_of_categories = ebird_name_map.shape[0]
print('No of bird categories: ', no_of_categories)


# In[ ]:


train_df.columns


# # Uncomment the following code cell to hear the sounds. 

# In[ ]:


# for ebird_code in listdir(PATH_AUDIO)[:10]:
#     sample_audio_path = f'{PATH_AUDIO}{ebird_code}/{listdir(PATH_AUDIO + ebird_code)[0]}'
#     species = ebird_name_map[ebird_name_map.ebird_code == ebird_code].species.values[0]
#     ipydisplay.display(ipydisplay.HTML(f"<h3>{ebird_code} ({species})</h3>"))    
#     ipydisplay.display(ipydisplay.Audio(sample_audio_path))


# In[ ]:


# train_df.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings", ascending=False).plot.barh(figsize = (5,100))
output = train_df.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings", ascending=False).reset_index(drop=True, inplace=False).copy()
plt.figure(figsize = (7, 60))
plt.xticks(rotation = 90)
plt.barh(output.species, output.recordings)
plt.grid(True)
plt.show()


# In[ ]:


print('Categories with 100 recordings: ', output[output.recordings ==100].species.count())
print('Categories with less than 100 recordings:', output[output.recordings <100].species.count())


# It seems that almost **(130/264) = 49.2%** of the categories of the birds have less than the maximum number of samples present(100) in the training folder. 
# * However in the extended datasets are available in [dataset part 1](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m) and [dataset part 2](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z) (thanks to [Vopani](https://www.kaggle.com/rohanrao)) which contains almost 20k samples each categories. 
# * **Therefore it will be a good idea to take samples from that original external dataset to fill out the categories which has less than 100 samples** 

# In[ ]:





# # I will keep updating this notebook as I walk through more things.
