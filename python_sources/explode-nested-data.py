#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


CSV_FILE = "/kaggle/input/stanford-plato-corpus/simple_data.csv"


# In[ ]:


df = pd.read_csv(CSV_FILE)


# In[ ]:


df.size


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


def string_to_python_list(x):
    return eval(x) if isinstance(x,str) else None

df['related_entries_list'] = df['related_entries_list'].apply(string_to_python_list)


# In[ ]:


df['sections'] = df['sections'].apply(string_to_python_list)


# In[ ]:


df.dtypes


# # Now explode

# ### [example](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html)

# In[ ]:


example_df = pd.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': 1})
example_df


# In[ ]:


example_df.explode("A")


# # ~~Explode ***related_entries_list***~~
# ## bad idea... too many rows

# In[ ]:


# related_topic_per_row_df = df.explode("related_entries_list")


# In[ ]:


# related_topic_per_row_df.head()


# In[ ]:


# section_per_row_and_related_topic_per_row_df = related_topic_per_row_df.explode("sections")


# In[ ]:


# section_per_row_and_related_topic_per_row_df.head()


# # Just explode the sections 
# ## to later extract paragraphs

# In[ ]:


df = df.explode("sections")


# In[ ]:


df.columns


# In[ ]:


# df = df.rename(columns={"related_entries_list": "related_topic"})
df = df.rename(columns={"sections": "section"})
df = df.rename(columns={"filenames": "filename"})


# In[ ]:


df.head()


# # Now extract all the section.id & section.heading_text & section.paragraphs on the section JSON

# ### first notice the keys

# In[ ]:


df['section'].values[1]


# In[ ]:


df['section'].values[2].keys()


# In[ ]:


df['section'].values[2]['paragraphs'][0].keys()


# ## Now safely get the dictionary values

# In[ ]:


def safe_get(getter=lambda x, n: x):
    def _get(x):
        if x == None:
            return None
        elif isinstance(x, dict):
            return getter(x)
        else:
            raise NotImplementedError()
    return _get

def _id_getter(x):
    return x["id"]

def _ht_getter(x):
    return x["heading_text"]

def _p_getter(x):
    return x["paragraphs"]

def _text_getter(x):
    return x["text"]


# In[ ]:


df['section.id'] = df['section'].apply(safe_get(_id_getter))
df['section.heading_text'] = df['section'].apply(safe_get(_ht_getter))
df['section.paragraphs'] = df['section'].apply(safe_get(_p_getter))


# # Explode the section.paragraphs which is also list-of-dict

# In[ ]:


df = df.explode("section.paragraphs")


# a better name

# In[ ]:


df = df.rename(columns={"section.paragraphs": "section.paragraph"})


# # Now extract the *section.paragraph.id* and *section.paragraph.text*

# In[ ]:


df['section.paragraph.id'] = df['section.paragraph'].apply(safe_get(_id_getter))
df['section.paragraph.text'] = df['section.paragraph'].apply(safe_get(_text_getter))


# In[ ]:


df.head()


# # Now just choose the useful columns to conserve memory
# 
# ### consider that the `section` dict is no longer needed since the data was extracted. same for `section.paragraph`
# 
# ### also consider that `plain_text` duplicates the `section.paragraph.text` without the granualar section/paragraph structure

# In[ ]:


df.columns


# In[ ]:


mask = [
    'filename', 'filetype', 'topic', 'title', 'author', 'creator',
    'preamble_text', 
    #'section', 
    'related_entries_list', 
    # 'plain_text',
    'section.id', 'section.heading_text', 
    # 'section.paragraph', 
    'section.paragraph.id', 'section.paragraph.text'
]
df = df[mask]


# In[ ]:


df.size


# In[ ]:


df.shape


# In[ ]:


len(df)


# In[ ]:


df.head()


# # more tricks to conserve memory

# In[ ]:


def chunks(lst, N):
    """Yield N successive `len(lst)//N`-sized chunks from lst."""
    chunk_size = len(lst)//N
    for i in range(0, len(lst), len(lst)//N):
        yield lst[i:i + chunk_size]


# In[ ]:


TOTAL_CHUNKS = 10000


# In[ ]:


df_generator = chunks(df, TOTAL_CHUNKS)


# # Now save

# In[ ]:


CSV_FILENAME = "data_per_paragraph.csv"

# save first one
first_with_header = next(df_generator)

first_with_header.to_csv(CSV_FILENAME, mode='w', header=True)


# In[ ]:


del first_with_header
gc.collect()


# In[ ]:


i = 2
COLLECTED = gc.collect()
print(COLLECTED)
for x in df_generator:
#     if (i % 1000 == 0):
#         COLLECTED = gc.collect()
    print(f"appending... {i} / {TOTAL_CHUNKS}\r", end="")
    x.to_csv(CSV_FILENAME, mode='a', header=False)
    i+=1


# # read the output to make sure all is well

# In[ ]:


pd.read_csv(CSV_FILENAME).head()


# # What's up with the np.nan paragraphs?
# 
# It seems those paragraphs (if any) did not fit the HTML stucture assumptions when the data was extracted from the web archive. 
# 
# That's okay.
# 
# TODO: consider dealing with these files counted below...

# In[ ]:


df[df["section.paragraph.text"].isna()].groupby("filetype").count()


# In[ ]:


na_par_df = df[df["section.paragraph.text"].isna()]
(
    na_par_df.join(
        na_par_df["filename"].apply(lambda x: os.path.split(x)[1]),
        rsuffix="_____"
    ).groupby("filename_____")
    .count()
    .sort_values(["filename","filetype","topic","title","author"], ascending=False)
)


# In[ ]:




