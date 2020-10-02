#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy


# In[ ]:


get_ipython().system('ls -l ../input/')


# In[ ]:


nlp = spacy.load('en')
doc = nlp('Hello World!')
for token in doc:
    print('"' + token.text + '"')

