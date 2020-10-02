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

# Any results you write to the current directory are saved as output.


# # Reading Conversation

# In[ ]:


convo1text = open("../input/conversation1.txt", "r").readlines()


# In[ ]:


convo1text


# # Data Cleaning 
# 
# Removing '\n' from the data

# In[ ]:


newConvo = []
for sent in convo1text:
    sent = sent.strip("\n")
    if not sent is '':
        newConvo.append(sent)
    
convo1text = newConvo
print(convo1text)


# # Appneding speaker at places where the user lists the points

# In[ ]:


for index,sent in enumerate(convo1text):
    if sent.count(':') > 0:
        previousSpeaker = sent.split(':')[0]
    if sent.count(':') == 0:
        convo1text[index] = previousSpeaker + ':' + sent
        
convo1text


# In[ ]:


conversation1 = []
for sent in convo1text:
    convo = sent.split(':')
    if len(convo) > 2:
        convo[1] += ('-').join(convo[2:])
    convo = convo[:2]
    convo[1] = convo[1].strip()
    if not convo[1] is '':
        conversation1.append([c for c in convo])


# In[ ]:


conversation1


# # Replacing Pronouns

# In[ ]:


import json
with open("../input/appos.json", "r") as read_file:
    data = json.load(read_file)
    
appos = data['appos']
appos


# In[ ]:


import re
processedConversation = []
for speaker, dialog in conversation1:
    for (key, val) in appos.items():
        dialog = dialog.lower()
        dialog = dialog.replace(key, val)
        dialog = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", dialog)
        dialog = ' ' + dialog + ' '
        dialog = re.sub(' +', ' ',dialog)
        
    processedConversation.append([speaker, dialog])
        


# In[ ]:


processedConversation


# # Conversion from 1st person to 3rd person conversation

# In[ ]:


with open("../input/thirdperson.json", "r") as read_file:
    data = json.load(read_file)
thirdperson = {}
for (key, value) in data['thirdperson'].items():
    thirdperson[' ' + key + ' '] = ' ' + value + ' '


    
thirdperson


# In[ ]:


finalConversation = []
previousSpeaker = ''
for speaker, dialog in processedConversation:
    for (key, val) in thirdperson.items():
        dialog = dialog.lower()
        dialog = dialog.replace(key, val)
        dialog = dialog.replace(' i ', ' ' +speaker + ' ')
        dialog = dialog.replace('i ', ' ' +speaker + ' ')
        dialog = dialog.replace('you', previousSpeaker if not previousSpeaker is '' else 'them' )
    finalConversation.append([speaker, dialog])
    
finalConversation


# # Summarize if it encounters something like summarize.

# Note that we will see the last occurance of lemma 'summarize'.
# 
# Then all the points that follow for the same speaker count as summary points

# In[ ]:


import spacy


# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


summary = ''
summarySpeaker = None
for speaker, dialog in finalConversation:
    if summarySpeaker is None:
        doc = nlp(dialog)
        for token in doc:
            if token.lemma_ == 'summarize':
                # this is the part where we hear someone summarize
                # point following this must be a part of the summarization
                summarySpeaker = speaker
    else:
        if speaker == summarySpeaker:
            summary = summary + dialog
        else:
            summarySpeaker = None
        
summary
            


# # Understanding topics from this summary

# ## removing stopwords

# In[ ]:


doc = nlp(summary)


# In[ ]:


spacy.displacy.render(doc, style='ent',jupyter=True)


# In[ ]:


from nltk.corpus import stopwords
from string import punctuation


# In[ ]:


_stopwords = set(stopwords.words('english') + list(punctuation) + list(' '))
_stopwords


# In[ ]:


word_sent = [word.text for word in doc if word.text not in _stopwords]
word_sent


# ## Calculate freq of these words in main text

# In[ ]:


from nltk.probability import FreqDist


# In[ ]:


main_sent = []
for speaker, dialog in finalConversation:
    dialogdoc = nlp(dialog)
    for word in dialogdoc:
        if word.text not in _stopwords:
            main_sent.append(word.text)


# In[ ]:


fd = {word:freq for word,freq in FreqDist(main_sent).items() if word in word_sent}


# In[ ]:


from heapq import nlargest


# In[ ]:


imp_words = nlargest(10, fd, key=fd.get)


# above is the list of most common words that are a part of summary

# now lets find insights based on this data

# In[ ]:


from collections import defaultdict


# In[ ]:


sents = []
for speaker, dialog in finalConversation:
    sents.append(dialog)


# In[ ]:


ranking = defaultdict(int)
for i, sent in enumerate(sents):
    doc = nlp(sent)
    for w in doc:
        if w.text in freq: 
            ranking[i] += freq[w.text]


# In[ ]:


ranking


# In[ ]:


sent_idx = nlargest(5, ranking, key=ranking.get)
sent_idx


# In[ ]:


[sents[i] for i in sent_idx]


# In[ ]:




