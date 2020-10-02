#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Movie Synopsis Model
# 
# ---
# 
# 
# The intent of this model is to input a movie genre and have it output a movie description/synopsis.

# ## Install and Include Libraries
# Install needed libraries not prepackaged with jupyter notebooks.

# In[ ]:


#!python -m spacy download en_core_web_lg
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install sumy')


# Include the needed libraries.

# In[ ]:


import string
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sumy.summarizers import luhn
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.luhn import LuhnSummarizer 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as sumytoken
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer


# ## Read in the dataset
# Import the data using pandas into a dataframe. Drop the rows that are null. Display the top 5 entries for displaying the data.

# In[ ]:


reviews_datasets = pd.read_csv(r'/kaggle/input/movies-metadata/movies_metadata.csv')
reviews_datasets.dropna()
reviews_datasets.head(5)


# ## Genre Filtering
# Here we are filtering down the dataset into just the movies with the genre in the genre variable. We then display the top 5 entries in order to ensure we are only getting the genre chosen.

# In[ ]:


genre = "Action"
filtered_reviews = reviews_datasets[reviews_datasets['genres'].str.contains(genre)]
filtered_reviews.head(5)


# ## Creating a corpus of movie synopses
# We get a random 50 movie synopses. We then loop through and concantenate all the overview column into one single string for training.

# In[ ]:


text = filtered_reviews.sample(25)['overview'].str.cat(sep='. ')


# ## Generation of Movie Synopsis
# Here we are generating a movie synopsis based on a collection of real movie synopses. We do that by taking a random 50 movies and performing a text summarization of the overall collection of movie synopses. We could use a LexRank, LSA, or Luhn model, however, we decided on the LSA algorithm. It seemed to generate a more complete and believable story consistently.

# In[ ]:


LANGUAGE = "english"
SENTENCES_COUNT = 3
parser = PlaintextParser.from_string((text), sumytoken(LANGUAGE))
stemmer = Stemmer(LANGUAGE)
SEP = " "

def lexrank_summarizer():
    sentences = []
    summarizer_LexRank = LexRankSummarizer(stemmer)
    summarizer_LexRank.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer_LexRank(parser.document, SENTENCES_COUNT):
        sentences.append(str(sentence))
    return SEP.join(sentences)
        
def lsa_summarizer():
    sentences = []
    summarizer_lsa = Summarizer(stemmer)
    summarizer_lsa.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer_lsa(parser.document, SENTENCES_COUNT):
        sentences.append(str(sentence))
    return SEP.join(sentences)
        
def luhn_summarizer():
    sentences = []
    summarizer_luhn = LuhnSummarizer(stemmer)
    summarizer_luhn.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer_luhn(parser.document, SENTENCES_COUNT):
        sentences.append(str(sentence))
    return SEP.join(sentences)

sum1 = lexrank_summarizer()
sum2 = lsa_summarizer()
sum3 = luhn_summarizer()

print(sum1)

