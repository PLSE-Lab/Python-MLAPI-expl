#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing
# 
# Natural Language Processing which make )computers understand the natural language. Computers can understand the structured form of data like spreadsheets and the tables in the database, but human languages, texts, and voices form an unstructured category of data, and it gets difficult for the computer to understand it, and there arises the need for Natural Language Processing.
# 
# ![NLP.png](attachment:NLP.png)

# # Tokenizing words and Sentences

# In[ ]:


import nltk
nltk.download('punkt')


# ## Tokenizing - Word Tokenizer, Sentence Tokenizer
# 
# ***Word Tokenizer*** - Word tokenization is the process of splitting a large sample of text into words.
# 
# ***Sentence Tokenizer*** - Sentence tokenization is the process of splitting text into individual sentences. 
# 
# ## lexicon and corpus
# 
# ***Lexicon*** - Will refer to the component of a NLP system that contains information (semantic, grammatical) about individual words or word strings.
# 
# ***Corpus*** - It represents a collection of (data) texts, typically labeled with text annotations: labeled corpus'''

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello, this is tokenizing. Which is helpfull? I do belive this is good way."

sent_token = sent_tokenize(text)

print("This is sent_tokenizer")
sent_token


# In[ ]:


word_token = word_tokenize(text)

print("This is word_tokenizer")
word_token

