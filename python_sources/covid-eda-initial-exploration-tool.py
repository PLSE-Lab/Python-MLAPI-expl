#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import json


# In[ ]:


get_ipython().system('ls /kaggle/input/CORD-19-research-challenge')


# ## Loading Metadata

# In[ ]:


root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()


# ## Fetch All of JSON File Path

# In[ ]:


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# ## Checking JSON Schema Structure

# In[ ]:


with open(all_json[0]) as file:
    first_entry = json.load(file)
    print(json.dumps(first_entry, indent=4))


# It seems that the JSON schema given is incorrect if we look into dumped first entry. The `abstract`, `body_text`, `bib_entries`, `ref_entries`, and `back_matter` are not the child of `metadata` key.

# ## Helper: File Reader Class

# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


# ## Load the Data into DataFrame

# In[ ]:


dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
df_covid.head()


# ## Adding the Word Count Column

# In[ ]:


df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))
df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
df_covid.head()


# In[ ]:


df_covid.describe(include='all')


# Based on the `unique` count on `abstract` and `body_text` it seems there some duplication that may caused by some group of authors publish their paper in the different journal (need deeper exploration).

# ## Clean Duplicates

# In[ ]:


df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
df_covid.describe(include='all')


# ## Boxplot: Word Count

# In[ ]:


df_covid[['abstract_word_count', 'body_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))
plt.show()

