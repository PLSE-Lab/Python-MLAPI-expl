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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import nltk
import nltk.corpus

print(os.listdir(nltk.data.find("corpora")))


# In[ ]:


from nltk.corpus import brown
brown.words()


# In[ ]:


nltk.corpus.gutenberg.fileids()


# In[ ]:


hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet


# In[ ]:


for word in hamlet[:500]:
    print(word, sep=' ', end=' ')


# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


poem = '''Where the mind is without fear and the head is held high
Where knowledge is free
Where the world has not been broken up into fragments
By narrow domestic walls
Where words come out from the depth of truth
Where tireless striving stretches its arms towards perfection
Where the clear stream of reason has not lost its way
Into the dreary desert sand of dead habit
Where the mind is led forward by thee
Into ever-widening thought and action
Into that heaven of freedom, my Father, let my country awake.'''

poem_tokens = word_tokenize(poem)
poem_tokens


# In[ ]:


from nltk.probability import FreqDist
fdist = FreqDist()

for word in poem_tokens:
    fdist[word.lower()]+=1
fdist


# In[ ]:


fdist['where']
fdist_top = fdist.most_common(5)
fdist_top


# In[ ]:


from nltk.tokenize import blankline_tokenize
poem_blank = blankline_tokenize(poem)
len(poem_blank)


# In[ ]:


from nltk.util import bigrams,trigrams,ngrams

string = "Death is not extinguishing the light; it is only putting out the lamp because the dawn has come."
quote_tokens = nltk.word_tokenize(string)


# In[ ]:


quote_trigrams = list(nltk.trigrams(quote_tokens))
quote_trigrams


# In[ ]:


from nltk.stem import PorterStemmer
pst = PorterStemmer()

(/LancasterStemmer, :, aggressive)
(/SnowballStemmer)

pst.stem("raining")


# In[ ]:


from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer

word_lem = WordNetLemmatizer()

word_lem.lemmatize('corpora')


# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[ ]:


import re
punctuation=re.compile(r'[-.?!,:;()|0-9]')


# In[ ]:


post_punctuation=[]
for words in poem_tokens:
    word=punctuation.sub("", words)
    if len(word)>0:
        post_punctuation.append(word)
        
post_punctuation


# In[ ]:


sent = "The secret of getting ahead is getting started."
sent_tokens = word_tokenize(sent)

for token in sent_tokens:
    print(nltk.pos_tag([token]))


# In[ ]:


from nltk import ne_chunk

NE_sent="MK Gandhi was born in Porbandhar, Gujarat."
NE_tokens=word_tokenize(NE_sent)

NE_tags=nltk.pos_tag(NE_tokens)

NE_NER = ne_chunk(NE_tags)
print(NE_NER )


# In[ ]:


new="I hear and I forget. I see and I remember. I do and I understand."
new_tokens = nltk.pos_tag(word_tokenize(new))
new_tokens


# In[ ]:


grammar_np = r"NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar_np)

chunk_result = chunk_parser.parse(new_tokens)
chunk_result

