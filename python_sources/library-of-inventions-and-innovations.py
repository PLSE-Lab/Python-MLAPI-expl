#!/usr/bin/env python
# coding: utf-8

# ##Simple Noun-Phrase Extraction & Wordcloud

# In[ ]:


import xml.etree.ElementTree as x
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud,STOPWORDS

d = []
s = ""
f = open("../input/ipg161011.xml")
for l in f:
    if l == "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n":
        if len(s)>0:
            d.append(s)
        s = ""
    s += l
d.append(s)

s = []

for xm in d:
    root = x.fromstring(xm)
    for e in root.iter(tag="invention-title"):
        s.append(str(e.text))

print(s[:3])


# In[ ]:


import pandas as pd
from textblob import Blobber
from nltk.tokenize.regexp import RegexpTokenizer

tb = Blobber(tokenizer=RegexpTokenizer('\d{0,2}\w{2}'))

np = []
for t in s:
    blob = tb(t)
    np.extend(blob.noun_phrases.singularize().lemmatize())

print(np[:5])


# In[ ]:


np = pd.Series(np)
np.head()


# In[ ]:


# Check for abnormally highly reoccuring terms
np.value_counts()[:25]


# The first 4 terms are stopwords - They reoccur in most patents so let's ignore them.

# In[ ]:


np_index = np.value_counts()[np.value_counts() >= 50].index
stopwords = list(np_index.values)


# In[ ]:


# Eliminate any term that contains stopwords, return other terms as continuous text
def generate_text(data):
    return ' '.join([t for t in data if not any(s in t for s in stopwords)])


# In[ ]:


wc = WordCloud(background_color='white', width=800, height=800).generate(generate_text(np))
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.imshow(wc)
plt.axis('off')
plt.show()

