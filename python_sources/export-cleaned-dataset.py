#!/usr/bin/env python
# coding: utf-8

# # Clean and export CORD data
# Much of the JSON loading code was taken from [this notebook](https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool).

# In[ ]:


import numpy as np 
import pandas as pd
import re
import glob
import os
import sys
import json


# In[ ]:


root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# In[ ]:


# load metadata
metadata = pd.read_csv(metadata_path)
metadata.head()


# In[ ]:


metadata.sha.isnull().sum()


# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.title = content['metadata']['title']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text) # could change this if we want to delineate sections
            
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


# In[ ]:


dict_ = {'paper_id': [], 'title':[], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['title'].append(content.title)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'title', 'abstract', 'body_text'])
df_covid.head()


# In[ ]:


# any true nulls in the data?
df_covid.isnull().sum(axis=0)


# In[ ]:


# quite a few empty titles and abstracts
(df_covid == '').sum(axis=0)


# In[ ]:


# no duplicate paper IDs
df_covid['paper_id'].is_unique


# In[ ]:


sum(df_covid[['title','abstract','body_text']].duplicated())


# In[ ]:


sum(df_covid['body_text'].duplicated())


# In[ ]:


# a few dupes - get rid of those
df_covid.drop_duplicates('body_text', inplace=True)


# In[ ]:


df_covid.shape


# In[ ]:


full_df = df_covid    .merge(metadata.rename(columns={'sha':'paper_id'}).drop(['abstract','title'], axis=1), 
           on='paper_id', how='left')

full_df.head()


# In[ ]:


full_df.drop_duplicates('body_text', inplace=True)
print(f"{full_df.shape[0]} rows in the output dataframe")

full_df.to_csv('covid_corpus_33k.csv', index=False)

