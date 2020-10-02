#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from  pathlib import Path
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os
import glob
import gzip
import lzma
import  logging
logging.basicConfig(filename='out.log',format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

import collections
# Any results you write to the current directory are saved as output.
set_textes=set(glob.iglob("../input/**/*.txt",recursive=True))
set_textes.update(glob.iglob("../input/**/*.txt.*",recursive=True))
textes=sorted(set_textes)


# In[ ]:


import collections
class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
    def update(self, other):
            for i in other:
                self.add(i)


# In[ ]:


base_texte=set()


# In[ ]:


import re,itertools

def split_phrases(text,skip_start=24,skip_end=5):
    lines = ["".join(s).strip() for s in re.findall(r"^(.{5,}[?:!\.])|([?:!\.]{5,})$",text,flags=re.MULTILINE)]
    lines=lines[skip_start:-skip_end]

    gen_phrases=( re.findall(r".*?[!:.](?=\s[-A-Z]\w+ )",t) for t in lines)
    phrases=[s for s in itertools.chain.from_iterable( gen_phrases) ]


    return phrases


# In[ ]:


articles=pd.read_csv("../input/conversion-french-reddit/elements.csv.xz")

base_texte.update(articles["texte"].str.strip().values)


# In[ ]:


try:
    for c in pd.read_csv("../input/conversion-csv-tweets-presidentielle-2017/tweet_presidentielle1017.csv.xz", chunksize=4096, engine='python'):
        c=c["text"].append(c["quoted/retweeted full_text"]).fillna(value="")
        if len(c)>0:
            c=c.str.replace("@","").str.replace("#","")
            base_texte.update(c.values)
except:
    logging.exception("erreur")
    
    
    
    


# In[ ]:


tweets=pd.read_csv("../input/french-tweets-politique/tweets_unique.csv" ,low_memory=True, memory_map=True,)
tweets_text=tweets["text"].fillna(value="")

#tweets_text=tweets_text.str.replace(r"(?<=@)[^ ]*",r"\0")
tweets_text=tweets_text.str.replace(r"#",r"").str.replace(r"@",r"")
base_texte.update(tweets_text.values)


# In[ ]:


get_ipython().system('mkdir sortie')
base_texte=list(base_texte)
random.shuffle(base_texte)
import smart_open 

fichier="sortie/textes.txt.xz"
total=0
n_fichiers=0
with lzma.open(fichier,"wt",encoding='utf-8') as f:
    phrases=split_phrases("\n".join(str(s) for s in base_texte))
    print("textes csv",len(phrases))
    for p in set(phrases):
                print(p,file=f)
    for c in textes:
        n_fichiers+=1
        if n_fichiers%100==0:
            print(os.path.basename(c))
            logger.info("traitement %s %s",os.path.basename(c),c)
        with smart_open.open(c,"r",encoding='utf-8',errors='ignore') as g:
            phrases=split_phrases(g.read(int(4e5)))
            for p in OrderedSet(phrases):
                print(p,file=f)

            
            
get_ipython().system('ls   -l  -h')


# In[ ]:


max_vocab_size=int(16e6)
ft_iter=40


# In[ ]:


import gensim
from gensim.models import FastText

def ftbuidd(size,fichier=fichier,ft_iter=ft_iter,max_vocab_size=max_vocab_size,lang="fr",**kvargs):
    logger.info(f"FastText {size}")
    ft=FastText( corpus_file=fichier,
            sg=1,
            size=size,
            window=10,
            min_count=5,
            workers =8,
            negative=25,
            iter=ft_iter,
            #bucket=50000,
            max_vocab_size=max_vocab_size,
            **kvargs
           )
    ft.save(f'fastext_{lang}_{size}')
    return ft

    
ft=ftbuidd(100,fichier=fichier,ft_iter=ft_iter,max_vocab_size=max_vocab_size)



get_ipython().system('ls -l -h')


# In[ ]:


ft256=ftbuidd(256,fichier=fichier,ft_iter=ft_iter,max_vocab_size=max_vocab_size)


get_ipython().system('ls -l -h')


# In[ ]:


ftbuidd(300,fichier=fichier,ft_iter=ft_iter,max_vocab_size=max_vocab_size)

