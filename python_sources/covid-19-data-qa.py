#!/usr/bin/env python
# coding: utf-8

# As QA I want to care about data quality. Data quality is important for climbing a excellent result.
# Usually data testing should start from defining expected results with product owners and end-users. In this case I have no access to owners and i assume some expected results by myself. Anyway, I start QA from some exploratory testing session.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json


# **CSV file data**

# In[ ]:


df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")


# [](http://)**CSV-Test-1**: Sha key should be not null if has_full_test is True

# In[ ]:


isempty = df[(df['has_full_text'] == True) & (df['sha'] == 'NaN')].empty
print('Is the DataFrame empty :', isempty)


#  I'm only interested in sources with sha is not NaN

# * [](http://)**CSV-Test-2**: Sha key should be uniq

# In[ ]:


df_verified = df.dropna(subset=['sha'])
uniq_count = df_verified['sha'].drop_duplicates().count()
total_count = df_verified['sha'].count()
print('Uniq sha: ', uniq_count)
print('Total sha: ', total_count)
print(total_count - uniq_count)


# 12 sha values are not uniq

# In[ ]:


df_verified[df_verified.duplicated(subset=['sha'],keep=False)].sort_values('sha')


# **CSV-Test-3:** article should be existed

# In[ ]:


import os.path
def full_path(sha,full_text_file):
    data_path_pattern = "/kaggle/input/CORD-19-research-challenge/{full_text_file}/{full_text_file}/{sha}.json"
    path = data_path_pattern.format(
            full_text_file=full_text_file,
            sha=sha
            )
    return path


# In[ ]:


def is_file_exists(shas, full_text_file):
        result = all(os.path.exists(full_path(sha,full_text_file)) for sha in shas.split("; "))    
        return result

df_uniq = df_verified.drop_duplicates(subset = ["sha"])
df_uniq['is_file_exists'] = df_uniq.apply(lambda x: is_file_exists(x['sha'], x['full_text_file']), axis=1)
    


# In[ ]:


len(df_uniq[df_uniq['is_file_exists'] == False])


# In[ ]:


# Example of wp's data

data_path_pattern = "/kaggle/input/CORD-19-research-challenge/{full_text_file}/{full_text_file}/{sha}.json"

path = data_path_pattern.format(
    full_text_file="custom_license",
    sha="be7e8df88e63d2579e8d61e2c3d716d57d347676"
)

with open(path, "r") as f:
    data = json.load(f)


# *sha key*
# 1. Should be not null if full body is true (+)
# 2. Should be uniq (-) 12 sha keys are not uniq
# 3. Article with sha and has_full_text==true should be existed (+)

# ***Completeness***
# 0. Abstract is presented (-) many articles have no abstract
# 1. full body (?) There are articles with non clear body. len(body_text) = 1
# 2. number of words in full body (30k symbols)
# 3. link t external resources: (10)

# In[ ]:


def object_size(name, sha, full_text_file):
    path = full_path(sha,full_text_file)
    with open(path, "r") as f:
        data = json.load(f)
        return len(data[name])
    
def abstract_size(sha, full_text_file):
    return object_size('abstract', sha, full_text_file)

    
def body_size(sha, full_text_file):
    return object_size('body_text', sha, full_text_file)

def authors_size(sha, full_text_file):
    path = full_path(sha,full_text_file)
    with open(path, "r") as f:
        data = json.load(f)
        return len(data['metadata']['authors'])
    


# In[ ]:


from tqdm.notebook import trange, tqdm

df_uniq = df_verified.drop_duplicates(subset = ["sha"])[['sha','full_text_file']]
rows = []

for row in tqdm(df_uniq.iterrows()):
    _,(shas,full_text_file) = row
    for sha in shas.split("; "):
        new_row = {'sha': sha, 'full_text_file': full_text_file}
        rows.append(new_row)
        
df_test = pd.DataFrame(rows)


# In[ ]:


len(df_test)


# In[ ]:


df_test['is_file_exists'] = df_test.apply(lambda x: is_file_exists(x['sha'], x['full_text_file']), axis=1)


# In[ ]:


df_test['abstarct_size'] = df_test.apply(lambda x: abstract_size(x['sha'], x['full_text_file']), axis=1)


# In[ ]:


df_test['authors_size'] = df_test.apply(lambda x: authors_size(x['sha'], x['full_text_file']), axis=1)


# In[ ]:


df_test['body_size'] = df_test.apply(lambda x: body_size(x['sha'], x['full_text_file']), axis=1)


# In[ ]:


df_test['bib_entries_size'] = df_test.apply(lambda x: object_size('bib_entries', x['sha'], x['full_text_file']), axis=1)


# In[ ]:


df_test['authors_size'].describe()


# ***Found all articles with body test len = 1***
# print body
# 

# In[ ]:


from tqdm.notebook import trange, tqdm
df_uniq = df_verified.drop_duplicates(subset = ["sha"]).head(100)[['sha','full_text_file']]
for row in tqdm(df_uniq.iterrows()):
    _,(shas,full_text_file) = row
    for sha in shas.split("; "):
        path = data_path_pattern.format(
        full_text_file=full_text_file,
        sha=sha
        )
        with open(path, "r") as f:
            data = json.load(f)
            if (len(data['body_text']) == 1):
                print(data['body_text'])


# * I think I could assume that let(data['body_text']) == 1 is not good data too.
# 

# ***Uniqueness***
# 1. Data should contain uniq articals 
# 2. Sha key should be uniq : Look at test CSV-Test-2 

# > **Uniq-Test-1: Article should be uniq**
# 
# Idea: Data set should contain only uniq articles. 
# I can check articles in Metadata

# In[ ]:


duplicate_title_df = df_verified[df_verified.duplicated(subset=['title'],keep=False)].sort_values('title')
duplicate_title_df


# In[ ]:


duplicate_title_df['title'].value_counts()


# 1. I found that  titles have title "Index' and 'Author index'

# As bonus let's define journal distribution
# 
# 

# In[ ]:


df_verified['journal'].value_counts()


# > PLOS One is a peer-reviewed open access scientific journal published by the Public Library of Science since 2006. The journal covers primary research from any discipline within science and medicine. Wikipedia

# 

# ***Credibility***
# It is the most difficult for testing. How can o define Credibility for article? It means that should i find all "fake" articles? I'd like to found all articles by next creteriases:
# 1. Article should has a bib_entries
# 2. Article should has Authors

# **Cred-Test-1****: article should have at least 3 bib_entries

# In[ ]:


#Article should has a bib_entries > 3 items

from tqdm.notebook import trange, tqdm
df_uniq = df_verified.drop_duplicates(subset = ["sha"]).head(10)[['sha','full_text_file']]
for row in tqdm(df_uniq.iterrows()):
    _,(shas,full_text_file) = row
    for sha in shas.split("; "):
        path = data_path_pattern.format(
        full_text_file=full_text_file,
        sha=sha
        )
        with open(path, "r") as f:
            data = json.load(f)
            if (len(data['bib_entries']) < 3):
                print(data['bib_entries'])
                print(sha)


# **Cred-Test-2****: article should have at least 3 authors

# In[ ]:


#Article should has a **Cred-Test-2****: article should have at least 3 authors > 3 items

from tqdm.notebook import trange, tqdm
i = 0
df_uniq = df_verified.drop_duplicates(subset = ["sha"])[['sha','full_text_file']]
for row in tqdm(df_uniq.iterrows()):
    _,(shas,full_text_file) = row
    for sha in shas.split("; "):
        path = data_path_pattern.format(
        full_text_file=full_text_file,
        sha=sha
        )

        with open(path, "r") as f:
            data = json.load(f)
            if (len(data['metadata']['authors']) < 2):
                i += 1

print(i)


# ***accuracy***
# 1. no typo
# 2. relevants links
# 3. Structure and style

# 
