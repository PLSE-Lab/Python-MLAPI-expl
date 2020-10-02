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

import os
import glob
import gzip,lzma,bz2 
import random
import re,itertools
# Any results you write to the current directory are saved as output.
#set_textes=set(glob.iglob("../input/**/*.txt",recursive=True))
#textes=list(set_textes)


# In[ ]:



def split_phrases(text,skip_start=150,skip_end=150):
    lines = ["".join(s).strip() for s in re.findall(r"^(.{5,}[?:!\.])|([?:!\.]{5,})$",text,flags=re.MULTILINE)]
    lines=lines[skip_start:-skip_end]

    gen_phrases=( re.findall(r".*?[!:.](?=\s[-A-Z]\w+ )",t) for t in lines)
    phrases=[s for s in itertools.chain.from_iterable( gen_phrases) ]


    return phrases


# In[ ]:


articles=pd.read_csv("../input/conversion-french-reddit/elements.csv.xz")


base_texte=set(articles["texte"].values)

with lzma.open("../input/conversion-csv-tweets-presidentielle-2017/tweets.txt.xz",mode="rt") as f:
    for t in f:
        t=t.replace("#","")
        t=t.replace("@","")
        base_texte.add(t)
        


# In[ ]:


fichier="textes.txt.bz2"


# In[ ]:




total=0
limit=4.4e9
limit=4.4e7
n=0
stored_phrases=set()
with open(fichier,"wb") as fb:
    with bz2.open(fb,"wt",encoding='utf-8') as f:
        phrases=split_phrases("\n".join(str(s) for s in base_texte))
        random.shuffle(phrases)
        for p in set(phrases):
                    print(p,file=f)
                    total+=len(p)
        stored_phrases.update(phrases)
        print(total//1024,fb.tell()//1024)

        for c in glob.iglob("../input/**/*.txt",recursive=True):
            n+=1
            if n%5==0:
                print(os.path.basename(c),total//1024,fb.tell()//1024,total/fb.tell())
            if fb.tell()>limit:
                break

            with open(c,"r",encoding='utf-8') as g:
                phrases=split_phrases(g.read())
                for p in (p for p in set(phrases) if p not in stored_phrases ):
                    print(p,file=f)
                    total+=len(p)
                    if fb.tell()>limit:
                        break
            stored_phrases.update(phrases)
        for c in glob.iglob("../input/**/*.txt.bz2",recursive=True):
            with bz2.open(c,"rt") as g:
                phrases=split_phrases(g.read())
                for p in (p for p in set(phrases) if p not in stored_phrases ):
                    print(p,file=f)
                    total+=len(p)
                    if fb.tell()>limit:
                        break
                stored_phrases.update(phrases)
        for c in glob.iglob("../input/**/*.txt.xz",recursive=True):
            with lzma.open(c,"rt") as g:
                phrases=split_phrases(g.read())
                for p in (p for p in set(phrases) if p not in stored_phrases ):
                    print(p,file=f)
                    total+=len(p)
                    if fb.tell()>limit:
                        break
                    stored_phrases.update(phrases)
        for c in glob.iglob("../input/**/*.txt.gz",recursive=True):
            with gzip.open(c,"rt") as g:
                phrases=split_phrases(g.read())
                for p in (p for p in set(phrases) if p not in stored_phrases ):
                    print(p,file=f)
                    total+=len(p)
                    if fb.tell()>limit:
                        break
                    stored_phrases.update(phrases)



            
            
get_ipython().system('ls   -l  -h')


# In[ ]:


from smart_open import smart_open
from nltk import RegexpTokenizer
class token_stream(object):
    def __init__(self,fichier):
        self.fichier=fichier
    def __iter__(self):
        toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        with smart_open(self.fichier, encoding='utf8') as f:
            for l in f:
                yield toknizer.tokenize(l)


# In[ ]:


import gensim
from gensim.models import FastText
get_ipython().system('ls   -l  -h')


ft=FastText( sentences=token_stream(fichier),
            sg=1,
            size=100,
            window=10,
            min_count=5,
            workers =8,
            iter=40,
             negative=10,
            #bucket=50000,
            #max_vocab_size=16000000
           )
ft.save('fastext_fr_100')
ft2=FastText( sentences=token_stream(fichier),
             sg=1,
             negative=10,
            size=256,
            window=10,
            min_count=5,
            workers =8,
            iter=40,
            #bucket=50000,
            #max_vocab_size=16000000
           )
ft2.save('fastext_fr_256')


get_ipython().system('ls -l -h')

