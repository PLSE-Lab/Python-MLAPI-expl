#!/usr/bin/env python
# coding: utf-8

# In[6]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from nltk.chunk import ne_chunk

from bs4 import BeautifulSoup
import requests
import re

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


ex = "European authorities fined Google a record $5.1 billion on Wednesday"" for abusing its power in the mobile phone market and ordered the company to alter its practices"


# ### applying word tokenization and part-of-speech tagging to the sentence.

# In[ ]:


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
sent


# ### chunking

# In[ ]:


pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp      = nltk.RegexpParser(pattern)
cs       = cp.parse(sent)

print(cs)


# In[ ]:


iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)


# In[ ]:


ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)


# ### Entity

# In[7]:


nlp = en_core_web_sm.load()


# In[ ]:


doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])


# ### Token

# <ul>
#     <li>B ==> the token begins an entity</li>
#     <li>I ==> the token is inside an entity</li>
#     <li>O ==> the token is outside an entity</li>
#     <li>' ' ==> no entity tag is set</li>
# </ul>

# In[ ]:


pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


# # Extracting named entity from an article

# In[ ]:


def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))


# in the code below : <br>the server refuses my connection (sending too many requests from same ip address in short period of time).
# so am going to use an external file wich contains the text data**

# In[ ]:


url = "https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html""?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=""first-column-region&region=top-news&WT.nav=top-news"

ny_bb   = url_to_string(url)


# In[3]:


f = open("../input/kernel.txt","r",encoding='UTF-8',errors='ignore')
ny_bb = f.read()


# There are 188 entities in the article

# In[8]:


article = nlp(ny_bb)
len(article.ents)


# they are represented as 10 unique labels:

# In[9]:


labels = [x.label_ for x in article.ents]
Counter(labels)


# #### the three most frequent tokens : 

# In[10]:


items = [x.text for x in article.ents]
Counter(items).most_common(3)


# In[11]:


sentences = [x for x in article.sents]
print(sentences[20])


# In[12]:


displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')


# #### the above sentence and its dependencies :

# In[14]:


displacy.render(nlp(str(sentences[20])), style='dep', jupyter = True, options = {'distance': 120})


# #### verbatim, extract part-of-speech and lemmatize this sentence.

# In[15]:


[(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[20])) 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]


# In[16]:


dict([(str(x), x.label_) for x in nlp(str(sentences[20])).ents])

