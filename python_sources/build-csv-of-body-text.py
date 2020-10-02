#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os,json


# ## Load the Metadata into a DataFrame and Create a List of SHAs with Full Text

# In[ ]:


DATA_DIR = '/kaggle/input/CORD-19-research-challenge'

df = pd.read_csv(f'{DATA_DIR}/metadata.csv')
shas = df[df.has_full_text==True].sha.tolist()


# ## Walk Through the Data Directory, Load JSON, and Build A List of Body Text

# In[ ]:


full_text = []
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
            
            raw_text = []
            
            for item in article_json['body_text']:
                raw_text.append(item['text'])
            
            full_text.append((sha, ' '.join(raw_text)))


# ## Create a DataFrame from the List and Save to CSV
# This dataframe can be merged on sha with the original metadata as well

# In[ ]:


print('Building dataframe')
df_fulltext = pd.DataFrame(full_text, columns=['sha', 'body_text'])

print('Saving dataframe')
df_fulltext.to_csv(f'body_text.csv', index=False)


# In[ ]:




