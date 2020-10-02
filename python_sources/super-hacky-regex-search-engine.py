#!/usr/bin/env python
# coding: utf-8

# # Super hacky and slow regex search engine
# 
# This is just another way for me to understand what's on the data. It's uses regular expressions to browse through the papers and display sorrounding snippets on markdown

# In[ ]:


from pathlib import Path

DATA_PATH = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13/')


# In[ ]:


from tqdm import tqdm
import json

def iter_papers():
    """
    Iterate over all directories and yields all papers
    """
    dirs = 'comm_use_subset noncomm_use_subset pmc_custom_license biorxiv_medrxiv'.split()
    for dir in dirs:
        fnames = (DATA_PATH / dir / dir).glob('*')
        for fname in fnames:
            with fname.open() as f:
                content = json.load(f)
            yield content    


# In[ ]:


import re
from collections import namedtuple

Window = namedtuple('Window', ['field', 'left', 'center', 'right'])

def _get_win_pat(name, size):
    return '(?P<%s>\w+(\W+\w+){,%s})' % (name, size)

def iter_gd(paper, pat_str, left_wsize=5, right_wsize=5):
    """
    Generator for matching snippets
    """
    pat_str = pat_str.replace(' ', '\W+')
    full_pat_str = f"{_get_win_pat('left', left_wsize)} (?P<center>{pat_str}) {_get_win_pat('right', left_wsize)}"
    pat = re.compile(full_pat_str, re.I | re.M)
    
    def get(k):
        res = paper
        for kk in k.split('.'): res=res.get(kk)
        return res
        
    for key in 'abstract body_text metadata.title'.split():
        val = get(key)
#         print(key)
        if isinstance(val, list): val = '\n\n'.join([e['text'] for e in val])

#         print(val)
        for match in pat.finditer(val):
            yield Window(field=key, **match.groupdict())


# # For this first attempt, just limit the papers to those saying either covid or coronavirus on the title

# In[ ]:


it = iter_papers()

from tqdm.auto import tqdm

inserted = tqdm(desc='inserted')
relevant = []
pat = re.compile('covid|coronavirus', re.I)
for p in tqdm(iter_papers()):
    if pat.match(p['metadata']['title']): 
        relevant.append(p)
        inserted.update()


# # Snippets
# 
# Now look for snippets sorrounding the following keywords:
# 
# - transmission
# - incubation
# - environmental stability
# - nasal discharge
# - sputum
# - urine

# In[ ]:


from IPython.display import display, Markdown

dm = lambda x: display(Markdown(x))

for paper in relevant:
    matches = list(iter_gd(
        paper, 'transmission|incubation|environmental stability|nasal discharge|sputum|urine', 
        left_wsize=30, right_wsize=30
    ))
    if matches:
        dm(f"#### {paper['metadata']['title']}")
        for match in matches:
            if match.field == 'metadata.title': continue
            dm(f"{match.left} **{match.center}** {match.right}")


# In[ ]:




