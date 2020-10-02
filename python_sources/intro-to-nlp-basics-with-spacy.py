#!/usr/bin/env python
# coding: utf-8

# ### What is NLP?
# 
# NLP - **Natural Language Processing** is the process of analysing Natural Language (as in, How we humans speak) and extracting meaningful insights from the given data. NLP has become one of the very popular areas of interest due to increase in NLP and also development in Information Extraction (IE) methodologies. 
# 
# ### Sources of Natural Langauge
# 
# * Social Media  (like FB Posts/Comments, Twitter Tweets, Youtube Comments)
# * Speech Transcripts (Call Center Conversations) 
# * Voice Agents (Amazon Echo, Google Home, Apple Siri) 
# 
# ### Some Applications of NLP
# 
# * Automated Customer Service 
# * Chatbots
# * Social Listening
# * Market Trends and much more

# ### About this Dataset
# 
# This dataset contains a bunch of tweet that came with this tag **#JustDoIt** after **Nike** released the ad campaign with Colin Kaepernick that turned controversial. 
# 
# <img src="https://www.thenation.com/wp-content/uploads/2018/09/Kaepernick-Nike-Ad-sg-img.jpg" alt="drawing" width="400"/>

# ### About spaCy:
# 
# spaCy by [explosion.ai](https://explosion.ai/) is a library for advanced **Natural Language Processing** in Python and Cython.
# spaCy comes with
# *pre-trained statistical models* and word
# vectors, and currently supports tokenization for **20+ languages**. It features
# the **fastest syntactic parser** in the world, convolutional **neural network models**
# for tagging, parsing and **named entity recognition** and easy **deep learning**
# integration. It's commercial open-source software, released under the MIT license.

# ### About this Kernel:
# 
# In this Kernel, We will learn how to use *spaCy* in Python to perform a few things of NLP. 

# Let us begin our journey by loading required libraries. 
# 
# ### Loading the required Libraries

# In[ ]:


#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#print(os.listdir("../input"))
import spacy
import random 
from collections import Counter #for counting
import seaborn as sns #for visualization


# As we have seen above, *spaCy* comes with Pre-trained Language models and since our tweets are predominantly English, let us load our *en* model using the following code:
# 
# ### Loading Spacy English Model

# In[ ]:


nlp = spacy.load('en')


# Please note that you can download other language models by running a code like below in your shell or terminal
# 
# `python -m spacy download en_core_web_sm` 
# 
# and then loading using `spacy.load()`. The last argument in the above code is the name of the langauge model that's to be downloaded. 
# 
# Now that our model is successfully loaded into `nlp`, let us read our input data using `read_csv()` of `pandas`. 
# 
# ### Reading input file - Tweets

# In[ ]:


tweets = pd.read_csv("../input/justdoit_tweets_2018_09_07_2.csv")


# As with any dataset, let us do a few basics like understanding the shape (dimension) of the dataset and then see a sample row. 
# 
# ### Dimension of the input file

# In[ ]:


tweets.shape


# ### Sample Row

# In[ ]:


tweets.head(1)


# Now that we know `tweet_full_text` is the column name in which tweets are stored, let us print some sample tweets.
# 
# ### Sample Tweets Text

# For simplicity, Let us take a sample of tweets.

# In[ ]:


random.seed(888)
text = tweets.tweet_full_text[random.sample(range(1,100),10)]
text


# ### Annotation:
# 
# Let us begin our NLP journey with Lingustic Annotation, which means marking each and every word with its linguistic type like if it's a NOUN, VERB and so on. This help us in giving grammatical labels to our Text Corpus. The function `nlp()` takes only string so let us use `str()` to combine all our rows above into one long string. 

# In[ ]:


text_combined = str(text)


# In[ ]:


doc = nlp(text_combined)


# ### Tokenization 
# 
# `doc` is the annotated text (that we did using the loaded langauge model). Now, let us tokenize our text. Tokenization has been done along with the above process. We can now print the **chunks**. The tokenized parts are called **chunks**. As a naive description, tokenization is nothing but breaking the long sentences/text corpus into a small chunks (or mostly words). 

# In[ ]:


for token in doc:
    print(token)


# Since we have already done the annotation, Let us print our chunks with their Parts-of-speech tags.

# In[ ]:


for token in doc:
    print(token.text, token.pos_)


# That's good, We've got a bunch of chunks and their respective POS tags. Perhaps, we don't want to see everything but just NOUNs.  Below is the code how we can print only the nouns in the text.

# In[ ]:


nouns = list(doc.noun_chunks)
nouns


# Sometimes, we might need to tokenization based on sentences. Let's say we've got Chat Transcript from Customer Service and in that case we need to tokenize our transcript based on sentences. 

# In[ ]:


list(doc.sents)


# ### Named Entity Recognition (NER)
# 
# NER is the process of extracting Named Entities like Person, Organization, Location and other such infromation from our Text Corpus.  spaCy also has an object `displacy` that lets us visualize our text with NER. We can display Named Entities using the following code:

# In[ ]:


for ent in doc.ents:
    print(ent.text,ent.label_)


# **spaCy** also allows to visualize Named Entities along woith the Text Labels. 

# In[ ]:


spacy.displacy.render(doc, style='ent',jupyter=True)


# ### Lemmatization
# 
# Lemmetiztion is the process of retrieving the root word of the current word. Lemmatization is an essential process in NLP to bring different variants of a single word to one root word. 

# In[ ]:


for token in doc:
    print(token.text, token.lemma_)


# As you can see in the above output, words like *aligning* and *values* have been converted to their root words *align* and *value*. 

# ### Dependency Parser Visualization

# In[ ]:


spacy.displacy.render(doc, style='dep',jupyter=True)


# ### WIP! We'll get into more concepts soon!

# Also please note, Text cleaning is one of the vital preprocessing for anything NLP and this kernel hasn't addressed it. 
