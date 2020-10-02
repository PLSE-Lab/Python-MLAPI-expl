#!/usr/bin/env python
# coding: utf-8

# ## A thorough introduction to tokenization in Natural Language Processing. (with Python code)

# ---
# 
# ## What is tokenization ?
# 
# > Tokenization is simply breaking down text into words.
# 
# ## But how do you know what is a word in a language?
# 
# > In the English language, words are usually seperated by "space". But it's not the case in every human language(think Chinese). And that's where the problem starts.

# ---
# 
# ## Understanding _Words_
# 
# Take a look at this sentence :
# 
# 'The quick brown fox jumps over the lazy fox, and took his meal.'
# 
# * The sentence has 13 _Words_ if you don't count punctuations, and 15 if you count punctions. 
# 
# * To count punctuation as a word or not depends on the task in hand.
# 
# * For some tasks like P-O-S tagging & speech synthesis, punctuations are treated as words. (Hello! and Hello? are different in speech synthesis)

# ## But I can break a sentence into words using the split(" ") method. What's the problem?

# <p style="color:red">Why you should not use split() for tokenizaiton.</p>
# 
# If using split() on the text, the words like 'Mr. Randolf', emails like 'hello@internet.com' may be broken down as ['Mr.','Randolf'], emails may be broken down as ['hello','@','internet','.','com'].
# 
# This is not what we generally want, hence special tokenization algorithms must be used.
# 
# * Commas are generally used as word boundaries but also in large numbers (540,000).
# * Periods are generally used as sentence boundaries but also in emails, urls, salutation.

# ---
# 
# ## But I can use nltk.tokenize. What's the problem?

# No doubt the nltk tokenize API is a great way to tokenize text. But we still have a problem. We can use nltk tokenize API under the assumption that the words in our text are seperated by _spaces_. What if they are not?
# 
# - Sometimes we want tokens to be space delimited, sometimes large word tokens (New York) and more.

# ---
# 
# ### A better approach
# 
# > Let your training data tell what is a token and what is not. In other words, instead of defining tokens as word seperated by spaces or as characters, we can use our data to automatically tell what size a token must be. 
# 
# If our training corpus contains, say the words low, and lowest, but not lower, but then the word lower appears in our test corpus, our system will not know what to do with it. A solution to this problem is to use a kind of tokenization in which most tokens are words, but some tokens are frequent morphemes or other subwords like -er, so that an unseen word can be represented by combining the parts.

# ## This can be done by using the Byte Pair encoding(BPE) algorithm to update a vocabulary with new tokens.
# 
# <p style="color:green;font-weight:bold">To understand BPE in depth(and go from zero-to-hero in NLP), follow the link below.</p>
# 
# Click [here](https://github.com/samacker77/Zero-to-Hero-in-NLP) to go from Zero to Hero in NLP.

# ## The code (using nltk vs tokenizers)
# 
# > Note the speed of both of these methods

# ---
# ### Using nltk

# In[ ]:


from nltk.tokenize import word_tokenize
from datetime import datetime


# In[ ]:


sentence = 'The town was fairly large with a dozen or            so business buildings on each side of the street but, as I said, most were closed.'


# In[ ]:


def nltkTokenizer(sentence):
    start = datetime.now()
    tokens = word_tokenize(sentence)
    end = datetime.now()
    time_taken = (end-start).microseconds
    print("Tokens\n")
    print(tokens)
    print("-"*50)
    print("\nTime taken\n")
    print("-"*10)
    print(str(time_taken)+" microseconds")


# In[ ]:


nltkTokenizer(sentence)


# ---

# ### Using tokenizers

# In[ ]:


get_ipython().system('python3 -m pip install tokenizers')


# In[ ]:


# Download pre-trained vocabulary file

get_ipython().system('wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt')


# In[ ]:


from tokenizers import (BertWordPieceTokenizer)
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)


# In[ ]:


def hfTokenizer(text):
    start = (datetime.now())
    print(tokenizer.encode(text).tokens)
    end = (datetime.now())
    print("Time taken - {} microseconds".format((end-start).microseconds))


# In[ ]:


hfTokenizer(sentence)


# ## Woah! Huggingface's tokenizer is 63% faster than nltk.
# 
# ## Also note how these tokens are ready to feed to BERT with already added special tokens like [SEP], [CLS]

# ---
# 
# ### To learn more on how to go from Zero-To-Hero in NLP, check this [Github](https://github.com/samacker77/Zero-to-Hero-in-NLP) repository.
# 
# > Leave an upvote if you like this.

# In[ ]:




