#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
import nltk


# In[ ]:


print(gutenberg.fileids())


# In[ ]:


for fileid in webtext.fileids():
    print(fileid,webtext.raw(fileid)[:65])


# In[ ]:


for fileid in nps_chat.fileids():
    print(fileid,nps_chat.raw(fileid)[:60])


# In[ ]:


for fileid in nps_chat.fileids():
    print(fileid,nps_chat.posts(fileid))


# In[ ]:


brown.categories()


# In[ ]:


brown.words(categories="news")


# In[ ]:


print(brown.fileids())


# In[ ]:


print(reuters.fileids())


# In[ ]:


print(inaugural.fileids())


# In[ ]:




