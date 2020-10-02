#!/usr/bin/env python
# coding: utf-8

# # Lemmatization 
# Stemming reduces word-forms to (pseudo)stems, whereas lemmatization reduces the word-forms to linguistically valid lemmas.
# Stemming and lemmatization helps to get the features of a unstructured text data into the structured data
# 
# # Word Lemmatization
# In the below example word lemmatization producing valid lemmas but it is not valid word as per the context of the sentence.
# for the sentence "Mary leaves the room" word lemmatization produces ['Mary', 'leaf', 'the', 'room'] lemmas.
# It is not correct as per the context.
# 
# # Line Lemmatization
# To get context of the sentence line lemmatization helps in below way. for the sentence "Mary leaves the room"
# line lemmatization produces ['mary', 'leave', 'the', 'room'] lemmas which are close to the context of the sentence
# 

# In[ ]:


from nltk.stem import WordNetLemmatizer


# In[ ]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[ ]:


sentence = "Mary leaves the room"


# In[ ]:


word_tokens = sentence.split(" ")


# In[ ]:


word_tokens


# In[ ]:


[wordnet_lemmatizer.lemmatize(test) for test in word_tokens]


# In[ ]:


get_ipython().system('pip install pywsd')


# In[ ]:


from pywsd.utils import lemmatize_sentence


# In[ ]:


lemmatize_sentence("Mary leaves the room")


# In[ ]:




