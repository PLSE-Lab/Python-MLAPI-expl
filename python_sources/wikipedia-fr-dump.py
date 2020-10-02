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
import json,requests,glob
import gensim
import zipfile
import lzma
# Any results you write to the current directory are saved as output.


# In[ ]:



#! wget    https://dumps.wikimedia.freemirror.org/frwikisource/20190301/frwikisource-20190301-pages-articles-multistream.xml.bz2 -P /tmp/
get_ipython().system(' wget    ftp://ftpmirror.your.org/pub/wikimedia/dumps/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2 -P /tmp/')
   


# In[ ]:


def fake_tk(t,*args):
  return t.split(" " )

dump_path=(glob.glob("frwiki*.xml.bz2")+glob.glob("/tmp/frwiki*.xml.bz2"))[0]
wiki = gensim.corpora.wikicorpus.WikiCorpus(dump_path,lemmatize=False ,dictionary ={} , lower=False ,
                  token_min_len=1,tokenizer_func =fake_tk )


# In[ ]:



n=1
tbuf=""
nd=1
# with zipfile.ZipFile("frwiki.zip", mode="w",compression=zipfile.ZIP_DEFLATED) as zf:
with zipfile.ZipFile("frwiki_text.zip", mode="w",compression=zipfile.ZIP_LZMA ) as zf_text:
    for text in wiki.get_texts():
                text_s=' '.join(text)
                text=bytes(text_s, 'utf-8').decode('utf-8') + '\n'

#                 zf.writestr(f"article{n}.txt",text_s)
                tbuf+=text_s
                if len(tbuf)>1.5e9:
                    zf_text.writestr(f"articles{nd}.txt",tbuf)
                    nd+=1
                    tbuf=""
                st=os.statvfs(".")

                if st.f_bavail*st.f_bsize<1e7:
                    zf_text.writestr(f"articles{nd}.txt",tbuf)
                    break



                if n%5000==0:
                  print(n)
                  print(text[:300])      
                elif n%1000==0:
                  print(n)
                n+=1
    zf_text.writestr(f"articles{nd}.txt",tbuf)

