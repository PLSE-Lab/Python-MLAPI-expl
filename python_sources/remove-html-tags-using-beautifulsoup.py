#!/usr/bin/env python
# coding: utf-8

# # Remove HTML Tags using BeautifulSoup
# 
# This kernel assumes that you use Spacy to tokenize the texts. Spacy is not able to process HTML tags properly, so some preprocessing needs to be done. 
# 
# You can choose to remove punctuations, certain types of words (e.g. adjectives, adverbs) after the tokenization. Here we just focus on getting the texts ready for Spacy.

# ## Contents
# 
# 1. [tl;dr](#tl;dr)
# 2. [Development](#Development)
#   - [HTML entities](#HTML-entities)
#   - [BeautifulSoup](#BeautifulSoup)
#   - [Spacing](#Spacing)
#   - [Hyperlinks](#Hyperlinks)
#   - [Spacy](#Spacy)
# 3. [Performance](#Performance)

# In[ ]:


import os
import re
import html as ihtml

import pandas as pd
from bs4 import BeautifulSoup


# In[ ]:


input_dir = '../input'

answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))


# ## tl;dr
# 
# Apply this function before spacy tokenization:

# In[ ]:


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text)).text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)    
    return text


# ## Development

# Pick a sample:

# In[ ]:


sample_text = answers.loc[31425, "answers_body"]
sample_text


# ### HTML entities
# Unescape HTML entities:

# In[ ]:


sample_text = ihtml.unescape(sample_text)
sample_text


# ### BeautifulSoup
# Remove HTML tags (note "<apta.org>" is also removed. That's an unfortunate false positive.):

# In[ ]:


sample_text = BeautifulSoup(sample_text, "lxml").text
sample_text


# ### Spacing
# Regularize spacing:

# In[ ]:


sample_text = re.sub(r"\s+", " ", sample_text)
sample_text


# Combining what we have so far:

# In[ ]:


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"\s+", " ", text)
    return text


# ### Hyperlinks

# We can also optionally remove hyperlinks as well:

# In[ ]:


sample_text = answers.loc[13137, "answers_body"]
sample_text


# In[ ]:


sample_text = clean_text(sample_text)
sample_text


# In[ ]:


re.sub(r"http[s]?://\S+", "", sample_text)


# Final function:

# In[ ]:


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
clean_text(answers.loc[13137, "answers_body"])


# ### Spacy

# Preview the tokenization results of Spacy:

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_sm')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')
nlp.remove_pipe('tagger')

" ".join(
    [token.lower_ for token in 
     list(nlp.pipe([clean_text(answers.loc[13137, "answers_body"])], n_threads=1))[0]]
)


# ## Performance

# In[ ]:


get_ipython().run_line_magic('timeit', 'answers.loc[~answers["answers_body"].isnull(), "answers_body"].apply(clean_text)')


# It's already fast enough. No need to do parallelization.

# Double checking:

# In[ ]:


results = answers.loc[~answers["answers_body"].isnull(), "answers_body"].apply(clean_text)


# In[ ]:


answers["answers_body"][0]


# In[ ]:


results[0]

