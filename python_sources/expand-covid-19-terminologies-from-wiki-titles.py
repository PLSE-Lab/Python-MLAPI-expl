#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import urllib.parse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup


# In[ ]:


df = pd.read_csv('../input/covid19-translations/COVID-19 Translations - TWB Terminologies.tsv', sep='\t')


# In[ ]:


df.head()


# # Collect the English Wikipedia Page

# In[ ]:


y = {}
for url in tqdm(df['Wikipedia (EN)'].dropna()):
    if url:
        bsoup = BeautifulSoup(requests.get(url).content.decode('utf8'))
        y[url] = bsoup


# # Use the titles from the interlanguage links as translations

# In[ ]:


z = defaultdict(dict)

for url in tqdm(y):
    bsoup = y[url]
    for li in bsoup.find_all('li', attrs={'class': re.compile('interlanguage-link interwiki-*')}):
        link = li.find('a').attrs['href']
        lang = link.split('https://')[1].split('.')[0]
        word = urllib.parse.unquote(link.split('/')[-1])
        z[url][lang] = word.replace('_', ' ').replace('_', ' ')


# # The extended terminology

# In[ ]:


z


# In[ ]:




