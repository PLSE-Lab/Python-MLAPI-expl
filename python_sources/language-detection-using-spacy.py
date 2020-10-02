#!/usr/bin/env python
# coding: utf-8

# # Language Detection using spacy
# <li> our main goal is to keep english tweets collected from twitter about Devoxx France event</li>
# <li> the tweets in our dataset are expressed in many languages such as french, english... </li>
# <li> Here we are using per trained [spaCy models](https://spacy.io/) to detect lenguage of the tweets.</li>

# <a href="#a">1. Needed Libraries</a><br>
# <a href="#b">2. Dataset</a><br>
# <a href="#c">3. Implementation with langdetect</a><br>
# <a href="#d">4. Implementation with spacy</a><br>
# <a href="#e">5. Conclusion</a><br>

# # <a id="a">1. Needed libraries</a>

# In[ ]:


get_ipython().system('pip install spacy_cld')


# In[ ]:


import numpy as np 
import pandas as pd
import re
import spacy
from spacy_cld import LanguageDetector


import langdetect
import langid

import os
print(os.listdir("../input"))


# In[ ]:


# function for data cleaning..
def remove_xml(text):
    return re.sub(r'<[^<]+?>', '', text)

def remove_newlines(text):
    return text.replace('\n', ' ') 
    

def remove_manyspaces(text):
    return re.sub(r'\s+', ' ', text)

def clean_text(text):
    text = remove_xml(text)
    text = remove_newlines(text)
    text = remove_manyspaces(text)
    return text


# # <a id="b">2. Dataset</a>

# In[ ]:


df = pd.read_csv('../input/data.csv')
df.head()


# In[ ]:


#Loading dataset  in tweets.
tweets    = df['tweets']


# Cleaning the Data in the dataset by removing XML, extra spaces, and newlines.

# In[ ]:


for line in tweets:
    line=clean_text(line)
df.head()


# Here is the view of samll portion of cleaned dataset taken for language detection.
# #### the code below is needed in the next sections :

# # <a id="c">3. Implementation with langdetect </a>

# In[ ]:


"""
result = str(result[0])[:2] : keeping the most dominant language wich is situated 
in the 1st index, and we store the first 2 characters
"""

languages_langdetect = []

# the try except blook because there is some tweets contain links
for line in tweets:
    try:
        result = langdetect.detect_langs(line)
        result = str(result[0])[:2]
    except:
        result = 'unknown'
    
    finally:
        languages_langdetect.append(result)


# # <a id="d">4. Implementation with spacy </a>

# In[ ]:


nlp = spacy.load('en')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)


# In[ ]:


"""
doc._.languages returns : list of str
like : ['fr'] -> french
       ['en'] -> english
       [] -> empty
       ['fr','en'] -> french (the most dominant in a tweet) and english (least dominant)
"""

tweets          = df['tweets']
languages_spacy = []

for e in tweets:
    doc = nlp(e)
    # cheking if the doc._.languages is not empty
    # then appending the first detected language in a list
    if(doc._.languages):
        languages_spacy.append(doc._.languages[0])
    # if it is empty, we append the list by unknown
    else:
        languages_spacy.append('unknown')


# ### Adding a column in the dataframe containing the language of the tweet

# In[ ]:


df['languages_spacy'] = languages_spacy
df['languages_langdetect'] = languages_langdetect


# * **tweets :** denotees the tweets of the dataset
# * **languages_spacy :** It shows the sentence which were found to be in english (en) by spacy model.
# * **languages_langdetect :** It shows the language of the sentence.

# In[ ]:


df.head()


# # <a id="e">5. Conclusion</a>
# <li>spacy returns 1582 en tweets </li>
# <li>langdetect returns 1018 en tweets</li>

# In[ ]:


df['languages_spacy'].value_counts()


# In[ ]:


df['languages_langdetect'].value_counts()


# In[ ]:


df.to_csv('Detected_Languages.csv',index=False)


# In[ ]:




