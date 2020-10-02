#!/usr/bin/env python
# coding: utf-8

# # Named Entity Recognition NLTK
# 
# Named Entity Recognition is a process where an algorithm takes a string of text (sentence or paragraph) as input and identifies relevant nouns (people, places, and organizations) that are mentioned in that string.

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


# In[ ]:


#Example Sentence

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'


# In[ ]:


#Apply word tokenization and part-of-speech tagging to the sentence.
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


# In[ ]:


#Sentence filtered with Word Tokenization

sent = preprocess(ex)
print("POS_Tags for Sentence")
sent


# ### Chunking
# 
# Chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.

# In[ ]:


#Chunking Pattern 

pattern = 'NP: {<DT>?<JJ>*<NN>}'


# In[ ]:


#create a chunk parser and test it on our sentence.
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)


# In[ ]:


#Chunker formed for sentence

NPChunker = nltk.RegexpParser(pattern) 
result = NPChunker.parse(sent)
#result.draw()


# The output can be read as a tree or a hierarchy with S as the first level, denoting sentence. we can also display it graphically.

# ![nltk%20ner.PNG](attachment:nltk%20ner.PNG)

# ## Spacy
# 
# ![spacy.PNG](attachment:spacy.PNG)

# ![spacy%202.PNG](attachment:spacy%202.PNG)

# In[ ]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# One of the nice things about Spacy is that we only need to apply nlp once, the entire background pipeline will return the objects.

# In[ ]:


doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])


# In[ ]:


print([(X, X.ent_iob_, X.ent_type_) for X in doc])


# In[ ]:


from bs4 import BeautifulSoup
import requests
import re


# In[ ]:


def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))


# In[ ]:


ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)
print(article.ents)
len(article.ents)


# There are 153 entities in the article and they are represented as 8 unique labels:

# In[ ]:


labels = [x.label_ for x in article.ents]
Counter(labels)


# In[ ]:


#most common used items

items = [x.text for x in article.ents]
Counter(items).most_common(3)


# In[ ]:


sentences = [x for x in article.sents]
print(sentences[20])


# In[ ]:


#Displaying detecing the name entities, Marking down.

displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')

