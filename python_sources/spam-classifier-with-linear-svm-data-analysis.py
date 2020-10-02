#!/usr/bin/env python
# coding: utf-8

# # Data-set analysis
# The data is grouped into 3 subdirectories:
# * easy_ham: messages  that  are typically quite easy to differentiate from spam
# * spam_2: spam messages
# * hard_ham: messages which are closer in many respects to typical spam

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from IPython.display import display, HTML
import email

print(os.listdir("../input"))
spam_path = '../input/spam_2/spam_2/'
easy_ham_path = '../input/easy_ham/easy_ham/'
hard_ham_path = '../input/hard_ham/hard_ham/'
# label messagges according to folder
email_files = {'spam':     os.listdir(spam_path),
               'easy_ham': os.listdir(easy_ham_path),
               'hard_ham': os.listdir(hard_ham_path)
              }
#count number of messages for each folder
print('\n spam emails: %d \n easy ham emails: %d \n hard ham emails: %d \n' %(len(email_files['spam']), len(email_files['easy_ham']), len(email_files['hard_ham']))
     )


# Visualizing one email - entire raw data

# In[18]:


path = spam_path
filename = random.sample(os.listdir(path), 1)[0]
file = open(path + filename,'r',errors='ignore')

content = file.read()
print(content)


# Visualizing the body (using email module)

# In[19]:


msg = email.message_from_string(content)
if msg.is_multipart():
    body = []
    for payload in msg.get_payload():
        # if payload.is_multipart(): ...
        body.append(payload.get_payload())
    body = ' '.join(body)

else:
    body = msg.get_payload()
print(body)


# ## Pre-processing emails [guidelines from Machine Learning Course - Andrew Ng]
# * Lowercasing
# * normalizing HTML html tags become 'html' word
# * normalizing numbers 
# * normalizing urls and email addresses
# * normalizing $
# * remove non-alphanumerical characters
# * tokenize with NLTK module
# * stemming the tokens with NLTK module

# In[41]:


import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk import FreqDist
body_pp = body.lower()
body_pp = re.sub(r"<[^<>]+>", " html ", body_pp)
body_pp = re.sub(r"[0-9]+", " number ", body_pp)
body_pp = re.sub(r"(http|https)://[^\s]*", ' httpaddr ', body_pp)
body_pp = re.sub(r"[^\s]+@[^\s]+", ' emailaddr ', body_pp)
body_pp = re.sub(r"[$]+", ' dollar ', body_pp)
body_pp = re.sub(r"[^a-zA-Z0-9]",' ', body_pp)
body_token = word_tokenize(body_pp)
stemmer = PorterStemmer()
body_stem = [stemmer.stem(token) for token in body_token]
body_freq_dict = FreqDist(body_stem)


# ## Creating a dictionary of word count with nltk module

# In[42]:


display(body_freq_dict)


# In[ ]:




