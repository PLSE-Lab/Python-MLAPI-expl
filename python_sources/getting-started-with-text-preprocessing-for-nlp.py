#!/usr/bin/env python
# coding: utf-8

# # Tokenization

# In[ ]:


import nltk


# In[ ]:


text = 'Hey, I love You!'


# 
# 
# *   nltk.tokenize.WhitespaceTokenizer() separates words with spaces
# 
# 

# In[ ]:


tokenizer = nltk.tokenize.WhitespaceTokenizer()
tokens = tokenizer.tokenize(text)


# In[ ]:


tokens


#  * WordPunctTokenizer() separates using punctuation marks

# In[ ]:


tokenizer = nltk.tokenize.WordPunctTokenizer()
tokens = tokenizer.tokenize(text)


# In[ ]:


tokens


# * TreebankWordTokenizer() separates words on basis of their morphological meaning.

# In[ ]:


tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize("Hey!, What's up? _Buddy?")


# In[ ]:


tokens


# # Token Normalization
#  * Stemming
#   - heuristics that chop off suffixes
#  * Lemmatization
#   - Returns the base or dictionary form of a word

# In[ ]:


words = "feet cats wolves talkes"
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(words)


# In[ ]:


# Stemming example
stemmer = nltk.stem.PorterStemmer()
print(words)
" ".join(stemmer.stem(token) for token in tokens)


# In[ ]:


# Lemmatization example
lemmatizer = nltk.stem.WordNetLemmatizer()
print(words)
" ".join(lemmatizer.lemmatize(token) for token in tokens)


# # Other types of normalization
#   * Normalizing capital letters
#     - lowercasing the beginning of the sentence
#     - lowercasing words in titles
#     - leave mid sentence words as they are
#   * Acronyms
#     - e.t.a, E.T.A as E.T.A
#     - or use regular expression for identifying such words for every probable combination( Hard!!!)
#     

# In[ ]:





# # Summary
#   - We can think of a text as sequences 
#   - Tokenization is a process of extracting those tokens
#   - we can normalize the tokens using stemming or lemmatization
#   - casing and acronyms can also be normalised.

# In[ ]:





# If you find this helpful, kindly consider to upvote. I will be uploading basic-to-advanced NLP methodologies with more explanation. Suggestions are welcome.

# In[ ]:




