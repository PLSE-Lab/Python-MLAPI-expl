#!/usr/bin/env python
# coding: utf-8

# # COVID-19 CoronaWhy NLP N-grams (Bigrams & Trigrams)

# Memory is a concern for this task so you'll see a few instances of some memory clean ups.

# ### Let's import all the tools we will need

# In[ ]:


import pandas as pd
import nltk, re, string, collections
from nltk.util import ngrams # function for making ngrams
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


# ### Now we will load the data from one of our CoronaWhy datasets.

# In[ ]:


df = pd.read_csv('/kaggle/input/coronawhy/dataset_v6.csv')


# ### Setting this column as text column to make the data easier to process.

# In[ ]:


df['text'] = df['text'].astype(str)
df['text'] = df['text'].str.lower()


# ### Filter the data by keywords.  This is recommended because there is a LOT of data to parse through.  In this section we will filter the data by anything that contains the word 'age'.

# In[ ]:


filter_keywords = ['age']
df = df[df['text'].str.contains('|'.join(filter_keywords))]


# ### Combining the text to search so we can process it for later.

# In[ ]:


text_to_search = ' '.join(df["text"])


# ### Now that we have our text loaded, let's delete the data frame to save some memory

# In[ ]:


del df


# ### Removing punctuation since we don't need that for N-grams.

# In[ ]:


# get rid of punctuation
punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text_to_search = re.sub(punctuationNoPeriod, "", text_to_search)


# ### Removing stop words.  We'll use the English stop words from NLTK plus some customized stop words we've been using for COVID-19

# In[ ]:


# let's remove stop words
# we will use the stop words provided by the NLTK
# we will also add in some customized stop words used in other places for COVID-19

customized_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 'q', 'license',
    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si', 'holder',
    'p', 'h'
]

stop_words = list(stopwords.words('english')) + customized_stop_words
print(stop_words)


# ### Let's tokenize the text and remove the stop words (this takes a while depending on the size of the data)

# In[ ]:


# let's tokenize the words
text_tokens = word_tokenize(text_to_search)
text_to_search = [word for word in text_tokens if not word in stop_words]


# ### Now we'll start with Bigrams.

# In[ ]:


# and get a list of all the bigrams
esBigrams = ngrams(text_to_search, 2)

# get the frequency of each bigram in our corpus
esBigramFreq = collections.Counter(esBigrams)

# what are the ten most popular bigrams
esBigramFreq.most_common(25)


# ### Now let's look at Trigrams.

# In[ ]:


# and get a list of all the trigrams
esTrigrams = ngrams(text_to_search, 3)

# get the frequency of each trigram in our corpus
esTrigramFreq = collections.Counter(esTrigrams)

# what are the ten most popular trigrams
esTrigramFreq.most_common(25)


# ### Cleaning up some RAM here since we don't have unlimited memory with Kaggle

# In[ ]:


del esBigrams
del esBigramFreq
del esTrigrams
del esTrigramFreq


# ### Now we will look for Bigrams and Trigrams with specific words.

# In[ ]:


search_for_word = 'age' # Text we want the Bi/Trigrams to contain

# reset the Bigrams
esBigrams = ngrams(text_to_search, 2)
esBigramFreq = collections.Counter(esBigrams)


# ### Now let's show the Bigrams containing our search word

# In[ ]:


for gram, freq in esBigramFreq.most_common():
    if gram[0] == search_for_word or gram[1] == search_for_word:
        print(gram, freq)


# ### Clean up memory again

# In[ ]:


del esBigrams
del esBigramFreq


# ### Now let's show the Trigrams containing our search word

# In[ ]:


# reset the Trigrams
esTrigrams = ngrams(text_to_search, 3)
esTrigramFreq = collections.Counter(esTrigrams)


# In[ ]:


for gram, freq in esTrigramFreq.most_common():
    if gram[0] == search_for_word or gram[1] == search_for_word or gram[2] == search_for_word:
        print(gram, freq)

