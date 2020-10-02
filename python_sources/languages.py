#!/usr/bin/env python
# coding: utf-8

# # Languages

# Recently I was doing some preprocessing on the documents in this set, and noticed there are different languages in the documents. Found a [question](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/139146) on the forum about how many languages, with a pointer to a nice [language detection library](https://pypi.org/project/langdetect/). Thanks to @dannellyz for the pointer. So this kernel is a quick look at how many docs there are in different languages.
# 
# Also helps to identify some documents that seem to be corrupt or otherwise invalid. In the end, it seems there is maybe around 1-2% of documents in non-English languages. I may skip them from my further analysis unless I find a nice way to translate them..
# 
# Anway..

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from langdetect import detect
from tqdm.auto import tqdm
tqdm.pandas()


# In[ ]:


import glob, json

def load_docs(base_path):
    loaded_docs = []
    file_paths = glob.glob(base_path)
    file_names = [os.path.basename(path) for path in file_paths]
    for filepath in tqdm(file_paths):
        doc = ""
        with open(filepath) as f:
            d = json.load(f)
            for paragraph in d["body_text"]:
                doc += " "+paragraph["text"].lower()
            loaded_docs.append(doc)
    return loaded_docs


# In[ ]:


medx_docs = load_docs("/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/*.json")


# In[ ]:


import collections

def count_langs(docs):
    langs = []
    non_english_idx = []
    broken_idx = []
    for idx, doc in enumerate(tqdm(docs)):
        try:
            lang = detect(doc)
            if lang != 'en':
                non_english_idx.append((lang, idx))
        except Exception as e: 
            print(f"Error detecing lang for doc {idx}: {e}")
            broken_idx.append(idx)
        langs.append(lang)
    counts = collections.Counter(langs)
    #print(f"non-english (lang, idx): {non_english_idx}")
    #print(f"broken (idx): {broken_idx}")
    for idx in broken_idx:
        print(f"broken {idx}: {docs[idx]}")
        print()
    return counts, non_english_idx


# In[ ]:


def print_non_english(docs, indices):
    for idx in indices:
        lang = idx[0]
        doc_idx = idx[1]
        if lang != 'fr' and lang != 'es' and lang != 'de':
            #there appear to be 200-300 french and spanish documents each, so skip those
            #the rest are just a few, so might as well just take a look just for interest
            print(f"{doc_idx}, {lang}: {docs[doc_idx][:1000]}")
            print()
    


# ## Biorxiv / Medrxiv

# In[ ]:


medx_counts, medx_nes = count_langs(medx_docs)
medx_counts.most_common()


# In[ ]:


comuse_docs = load_docs("/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/*.json")


# ## Comm_Use

# In[ ]:


comuse_counts, comuse_nes = count_langs(comuse_docs)
comuse_counts.most_common()


# ## Custom License

# In[ ]:


custom_docs = load_docs("/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/*.json")


# In[ ]:


custom_counts, custom_nes = count_langs(custom_docs)
custom_counts.most_common()


# ## NonComm Use

# In[ ]:


noncom_docs = load_docs("/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/*.json")


# In[ ]:


noncom_counts, noncom_nes = count_langs(noncom_docs)
noncom_counts.most_common()


# Having run this kernel multiple times, the actual indices of the "broken" documents changes, so cannot really re-use the indices but rather seems better to run it each time. Or make a filtered dataset..

# # Top Languages

# In[ ]:


total_count = collections.Counter()
total_count += medx_counts
total_count += comuse_counts
total_count += custom_counts
total_count += noncom_counts
total_count.most_common()


# Some of the non-English identified also seem a little bit messed up, likely causing identification as something different. Such as 'cy' [seems to be](https://www.loc.gov/standards/iso639-2/php/code_list.php) for Welsh :).

# ## Medrxiv, outliers

# In[ ]:


print_non_english(medx_docs, medx_nes)


# ## CommUse, outliers

# In[ ]:


print_non_english(comuse_docs, comuse_nes)


# ## Custom License, Outliers

# In[ ]:


print_non_english(custom_docs, custom_nes)


# ## NonComm, outliers

# In[ ]:


print_non_english(noncom_docs, noncom_nes)


# # Some Valid: French, Spanish, Deutch, ...

# But I found at least many Spanish and French docs that seem to contain something meaningful. So, if you have a nice solution to translate the valid non-English docs, would be happy to hear...

# In[ ]:


def print_lang(docs, indices, lang_to_print, max):
    count = 0
    for idx in indices:
        lang = idx[0]
        doc_idx = idx[1]
        if lang != lang_to_print:
            continue
        print(f"{doc_idx}, {lang}: {docs[doc_idx][:1000]}")
        print()
        count += 1
        if count >= max:
            break
    


# ## French

# In[ ]:


print_lang(custom_docs, custom_nes, "fr", 5)


# ## Spanish

# In[ ]:


print_lang(custom_docs, custom_nes, "es", 5)


# ## German

# In[ ]:


print_lang(custom_docs, custom_nes, "de", 5)


# Thats all folks, ...

# In[ ]:




