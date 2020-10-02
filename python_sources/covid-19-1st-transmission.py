#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import json
from pathlib import Path

files = list(Path("/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/").glob("**/*.json"))


# Any results you write to the current directory are saved as output.


# In[ ]:


strAbstract =""
for file in files:
    with open(file) as f:
        jsonLoads = json.loads(f.read())
    listAbstract = jsonLoads["abstract"]
    for listItem in listAbstract:
        strAbstract = strAbstract +" "+listItem["text"] 
    file1 = open("Abstract.txt","w")
    file1.write(strAbstract) 


# In[ ]:


import spacy
import scispacy
nlp = spacy.load("en_core_sci_sm")


# In[ ]:


doc = nlp(strAbstract[0:1000000])
print(len(list(doc.sents)))


# In[ ]:


print(type(doc.ents))


# In[ ]:


from collections import Counter 
split_it = strAbstract.split() 
Counter = Counter(split_it) 
most_occur = Counter.most_common(1000000) 
  
print(most_occur) 


# def freq(str):  
#     str_list = str.split() 
#     unique_words = set(str_list) 
#     for words in unique_words : 
#         print('Frequency of ', words , 'is :', str_list.count(words)) 
# freq(strAbstract)

# from spacy import displacy
# displacy.render(next(doc.sents), style='dep', jupyter=True)
# 

# In[ ]:


strBodyText=""
for file in files:
    with open(file) as f:
        jsonLoads = json.loads(f.read()) 
    listBodyText = jsonLoads["body_text"]
    for listItem in listBodyText:
        strBodyText = strBodyText+" "+listItem["text"]
    file1 = open("BodyText.txt","w")
    file1.write(strBodyText) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




