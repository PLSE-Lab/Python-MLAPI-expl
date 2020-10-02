#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# %reload_ext autoreload
# %autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
from tqdm import tqdm_notebook as tqdm


# # Get Data

# In[ ]:


get_ipython().system('wget "https://drive.google.com/uc?export=download&id=1BTQQQcb0a_MWmTRdKt0EGIKk2UfpnZJK" -O KWD_characters.txt')


# In[ ]:


with open('KWD_characters.txt', 'r') as f:
    s = f.read()
#     print(s)
    data_urls = eval(s)


# In[ ]:


urls = [ it[0] for it in data_urls]
with open('strips.txt', 'w') as f:
    f.write("\n".join(urls))
    
fnames = [url[url.rfind("/")+1:] for url in urls]
with open('strips_names.txt', 'w') as f:
    f.write("\n".join(fnames))
    
labels = [[c.replace(" ", "_") for c in it[1]] for it in data_urls]
with open('labels.txt', 'w') as f:
    f.write("\n".join([' '.join(l) for l in labels]))


# In[ ]:


get_ipython().system('cat strips_urls.txt | head -n5')
get_ipython().system('cat strips_fnames.txt | head -n5')
get_ipython().system('cat strips_labels.txt | head -n5')


# In[ ]:


# TODO: Clear it.
# FROM:
# https://forums.fast.ai/t/using-download-images-retaining-source-file-name/39463/2
def download_image(url,dest, timeout=4):
    try: r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e: print(f"Error {url} {e}")

def _download_image_inner(dest, info, i, timeout=4):
    url = info[0]
    name = info[1]
#     suffix = re.findall(r'\.\w+?(?=(?:\?|$))', url)
#     suffix = suffix[0] if len(suffix)>0  else '.jpg'
    download_image(url, dest/name, timeout=timeout)

def download_images_with_names(urls:Collection[str], dest:PathOrStr, names:PathOrStr=None, max_pics:int=1000, max_workers:int=8, timeout=4):
    "Download images listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_pics]
    if names:
      names = open(names).read().strip().split("\n")[:max_pics]
    else:
      names = [f"{index:08d}" for index in range(0,len(urls))]
    info_list = list(zip(urls, names))
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout), info_list, max_workers=max_workers)


# In[ ]:


get_ipython().system('rm -rf strips')
get_ipython().system('mkdir strips')
path = Path('/tmp/strips/')
download_images_with_names('strips.txt',path ,'strips_names.txt')


# In[ ]:


os.listdir(path)[:5]


# In[ ]:


import pprint
from collections import Counter
cnt = Counter([c for group in labels for c in group])
# pprint.pprint(cnt)
good_chars = [k for k,v in cnt.items() if v > 5]
good_chars.remove('Patron')
# chars = labels[0]
filtered_labels = [[c for c in chars if c in good_chars] for chars in labels]
cnt.most_common()


# In[ ]:


# labeled_data = [(f,l)  for (f,l) in zip(fnames, labels) if l]
use_data = [bool(l) for l in labels]
labeled_data = [(f," ".join(l)) for (f,l,u) in zip(fnames, filtered_labels,use_data) if u] 
print('Total images:', len(fnames))
print('Labeled images:', len(labeled_data))


# In[ ]:


get_ipython().system('cd /tmp; tar -cf strips.tar strips')
get_ipython().system('cp /tmp/strips.tar .')


# In[ ]:


get_ipython().system('ls')


# In[ ]:




