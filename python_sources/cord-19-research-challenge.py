#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
import re


# In[ ]:


# with open("../input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json", "r") as read_file:
#     data = json.load(read_file)


# In[ ]:


# data.keys()


# In[ ]:


# data['paper_id']


# In[ ]:


# data['metadata']['authors'][0].keys()


# In[ ]:


bio=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')
pmc=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')
comm=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')
non_comm=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')


# In[ ]:


conn=pd.concat([bio,pmc,comm,non_comm])


# In[ ]:


conn.isna().sum()


# In[ ]:


#Finding any word contains transmission,incubation, environmental stability


# In[ ]:


conn_extracted=conn[conn.apply(lambda row: row.astype(str).str.contains('transmission|incubation|environmental stability').any(), axis=1)]


# In[ ]:


def clean_text(i):
    return re.sub(r"\n\n", " ", i)


# In[ ]:


conn_extracted['text']=conn_extracted['text'].apply(lambda x: clean_text(x))


# In[ ]:


conn_extracted.head(1)


# In[ ]:


meta=pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')


# In[ ]:


meta.head()


# In[ ]:


meta['has_full_text'].value_counts()


# In[ ]:


# import os
# import json
# from pprint import pprint
# from copy import deepcopy

# import numpy as np
# import pandas as pd
# from tqdm.notebook import tqdm


# In[ ]:


# biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'
# filenames = os.listdir(biorxiv_dir)
# print("Number of articles retrieved from biorxiv:", len(filenames))


# In[ ]:


# all_files = []

# for filename in filenames:
#     filename = biorxiv_dir + filename
#     file = json.load(open(filename, 'rb'))
#     all_files.append(file)


# In[ ]:


# file = all_files[0]
# print("Dictionary keys:", file.keys())


# In[ ]:


# pprint(file['abstract'])


# In[ ]:


# print("body_text type:", type(file['body_text']))
# print("body_text length:", len(file['body_text']))
# print("body_text keys:", file['body_text'][0].keys())


# In[ ]:


# print("body_text content:")
# pprint(file['body_text'][:2], depth=3)


# In[ ]:


# texts = [(di['section'], di['text']) for di in file['body_text']]
# texts_di = {di['section']: "" for di in file['body_text']}
# for section, text in texts:
#     texts_di[section] += text

# pprint(list(texts_di.keys()))


# In[ ]:


# body = ""

# for section, text in texts_di.items():
#     body += section
#     body += "\n\n"
#     body += text
#     body += "\n\n"

# print(body[:3000])


# In[ ]:


# print(format_body(file['body_text'])[:3000])


# In[ ]:


# print(all_files[0]['metadata'].keys())


# In[ ]:


# print(all_files[0]['metadata']['title'])


# In[ ]:


# authors = all_files[0]['metadata']['authors']
# pprint(authors[:3])


# In[ ]:


# for author in authors:
#     print("Name:", format_name(author))
#     print("Affiliation:", format_affiliation(author['affiliation']))
#     print()


# In[ ]:


# pprint(all_files[4]['metadata'], depth=4)


# In[ ]:


# authors = all_files[4]['metadata']['authors']
# print("Formatting without affiliation:")
# print(format_authors(authors, with_affiliation=False))
# print("\nFormatting with affiliation:")
# print(format_authors(authors, with_affiliation=True))

