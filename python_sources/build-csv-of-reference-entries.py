#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os,json


# ## Load the metadata into a dataframe and create a list of SHAs with full text

# In[ ]:


DATA_DIR = '/kaggle/input/CORD-19-research-challenge'

df = pd.read_csv(f'{DATA_DIR}/metadata.csv')
shas = df.sha.tolist()


# ## Walk through the Data Directory, Load JSON, and Build List of Body Text

# In[ ]:


ref_entries = []
i = 0
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        sha, ext = os.path.splitext(file)
        if ext=='.json' and sha in shas:
            with open(f'{root}/{file}', 'r') as f:
                try:
                    article_json  = json.loads(f.read())
                except:
                    continue
            i+=1

            if i%250==0:
                print(f'Processed {i} articles. Currently processing {file}')

            
            for k, v in article_json['ref_entries'].items():
                ref_entries.append([sha, k, v['text'], v['latex'], v['type']])


# ## Create a DataFrame from the List and Save to CSV

# In[ ]:


print('Building dataframe')
df_ref_entries = pd.DataFrame(ref_entries, columns=['sha', 'name', 'text', 'latex', 'type'])

print('Saving dataframe')
df_ref_entries.to_csv(f'ref_entries.csv', index=False)

