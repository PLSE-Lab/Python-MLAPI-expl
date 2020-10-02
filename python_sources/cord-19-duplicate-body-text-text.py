#!/usr/bin/env python
# coding: utf-8

# ## tl;dr
# Roughly 5% of the articles in `CORD-19 (2020-03-13) comm_use_subset` contain duplicate, non-trivial paragraphs.
# 
# * These appear to come from a quirk in the pdf files themselves.
# * I haven't checked the other datasets
# 
# #### Quick example

# In[ ]:


get_ipython().system('grep "coexpressing of HAT or DESC1" /kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/7d4b0efca3cb904640b8fd489fdfc413a8fae264.json')


# Note that if you search in the [following PDF](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5454426/pdf/viruses-09-00114.pdf) for `coexpressing of HAT or DESC1`. there are some odd, ghost matches.

# ## Poking at the data
# ### Setup

# In[ ]:


import json
from pathlib import Path

PATH = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/')

def load_json(fpath):
    with fpath.open() as f:
        return json.load(f)

def print_paragraphs(fname, paragraphs):
    content = load_json(Path(fname))
    print('Title: {}\n'.format(content['metadata']['title']))
    for p in paragraphs:
        print('Paragraph #{0}: {1}\n'.format(p, content['body_text'][p]['text']))


# ## Example Paper
# File: `/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/7d4b0efca3cb904640b8fd489fdfc413a8fae264.json`
# 
# Title: TMPRSS2 and MSPL Facilitate Trypsin-Independent Porcine Epidemic Diarrhea Virus Replication in Vero Cells
# 
# Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5454426/
# 
# PDF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5454426/pdf/viruses-09-00114.pdf

# In[ ]:


print_paragraphs('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/7d4b0efca3cb904640b8fd489fdfc413a8fae264.json',
                [21,22,23])


# ### Finding the dupes

# In[ ]:


import hashlib
from collections import defaultdict

# Ignore very short paragraphs.
MIN_PARAGRAPH_LENGTH=10

total = 0
total_with_dupes = 0
for fname in PATH.glob('*.json'):
    total += 1
    content = load_json(fname)
    hash_to_num = defaultdict(list)
    for body_num, body_text in enumerate(content['body_text']):
        text = body_text['text'].encode('utf-8')
        # skip trivial dupes
        if len(text) < MIN_PARAGRAPH_LENGTH:
            continue
        hash_to_num[hashlib.sha256(text).hexdigest()].append(body_num)
    duplicates = []
    for nums in hash_to_num.values():
        if len(nums) > 1:
            duplicates.append(nums)
    if len(duplicates) > 0:
        total_with_dupes += 1
        print(fname)
        for d in duplicates:
            print('  duplicate paragraphs: {}'.format(d))


# In[ ]:


print(total)
print(total_with_dupes)


# In[ ]:


print_paragraphs('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/a08f5fd1ac9fc3e33a771787d584a845a8558cae.json',
                [10, 11])


# In[ ]:


print_paragraphs('/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/7d4b0efca3cb904640b8fd489fdfc413a8fae264.json',
                [21,22,23])

