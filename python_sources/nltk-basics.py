#!/usr/bin/env python
# coding: utf-8

# # Basic Snippets to understand Natural Language Processing

# # Tokenizing Text

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 


# In[ ]:


sent = "Mary had a little lamb. Her fleece was white as snow."
sents = sent_tokenize(sent)
print(sents)


# In[ ]:


words = [word_tokenize(sent) for sent in sents]
print(words)


# # Stop word Removal

# In[ ]:


from nltk.corpus import stopwords
from string import punctuation


# In[ ]:


customStopWords = set(stopwords.words('english') + list(punctuation))
print(customStopWords)


# In[ ]:


wordsWOStopWords = [word for word in word_tokenize(sent) if word not in customStopWords]
print(wordsWOStopWords)


# # Identifying Bigrams

# In[ ]:


from nltk.collocations import *
finder = BigramCollocationFinder.from_words(wordsWOStopWords)


# In[ ]:


sorted(finder.ngram_fd.items())


# # Stemming and POS

# In[ ]:


text = "Mary closed on closing night when she was in the mood to close."


# In[ ]:


from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
stemmedWords = [stemmer.stem(word) for word in word_tokenize(text)]
print(stemmedWords)


# In[ ]:


nltk.pos_tag(word_tokenize(text))


# # Word Sense Disambiguation

# In[ ]:


from nltk.corpus import wordnet as wn
for ss in wn.synsets('bass'):
    print(ss, ss.definition())


# In[ ]:


from nltk.wsd import lesk
sense = lesk(word_tokenize("She stays closed the  store"), 'close')
print(sense, sense.definition())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




