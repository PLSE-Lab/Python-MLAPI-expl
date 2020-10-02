#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, I've tried to cover the very basic steps that are required during data preprocessing and have elaborated the NLP pipeline to be used while approaching for any Natural Language problem. 
# ### Here is a pipeline shown that should be followed for any NLP problem and once you're thorough with the concepts of these underrated topics then you must proceed for trying BERT and other Transformers based models.

# ![image.png](attachment:image.png)
# Credits: [Adam Geitgey](https://medium.com/@ageitgey)

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ### Pre Processing data
# #### The following function ***preprocess_data*** covers all the preprocessing steps like removing any link in the data, emoji, hashtag, @ and stop words. It returns a sentence devoid of all the non-processable data. 

# In[ ]:


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import emoji
import string

def preprocess_data(data, remove_stop = True):
    
    data = re.sub('https?://\S+|www\.\S+', '', data)
    data = re.sub('<.*?>', '', data)
    emoj = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    emoj.sub(r'', data)
    data = data.lower()
    data = data.translate(str.maketrans('','', string.punctuation))
    data = re.sub(r'\[.*?\]', '', data)
    data = re.sub(r'\w*\d\w*','', data)
    
    words= data.split()
    
    if(remove_stop):
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        words = [w for w in words if len(w) > 2]  # remove a,an,of etc.
    
    words= ' '.join(words)
    
    return words


# #### Tokenizer function splits the sentence into tokens which can then be used for vector representation of words.
# #### Lemmatizer reduces the word to its root form. For eg: Corpora : Corpus
# #### Stemming merely deletes the suffix of the root word. For eg: Programer : Program , Programing : Program. Stemming could be sometimes inaccurate while deduction.

# In[ ]:


from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.tokenize import RegexpTokenizer


def tokenizer(words):
    tokenizer = RegexpTokenizer(r'\w+')
    words= tokenizer.tokenize(words)
    
    return words

def lemmatize(words):
    lemmatizer = WordNetLemmatizer() 
    lem= []
    for w in words:
        lem.append(lemmatizer.lemmatize(w))
    return lem

def stemming(words):
    ps = PorterStemmer() 
    stem= []
    for w in words:
        stem.append(ps.stem(w))
    return stem  


# In[ ]:


stemming(tokenizer(preprocess_data("@water #dream hi 19 :) hello where are you going be there tomorrow happening")))


# ### Using Spacy
# #### Spacy provides simple ways to perform complex operations like Dependency parsing, POS tagging, NER, Coreference resolution etc. on data. Here I'll demonstrate its usage via an example.

# In[ ]:


import spacy

nlp = spacy.load('en_core_web_lg')

text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
The City of Westminster is also an Inner London borough holding city status.
London is governed by the mayor of London and the London Assembly.
London has a diverse range of people and cultures, and more than 300 languages are spoken in the region.
"""
doc = nlp(text)


# ### Named Entity Recognition (NER)
# #### Each entity is given a category tag viz Person, Organization, Brand etc according to the spacy's dictionary. NER can easily enable machine to distinguish between "James" and "James Square". You can find out the meaning of each annotation on spacy website [here](https://spacy.io/api/annotation#named-entities).

# In[ ]:


from spacy import displacy

for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")

displacy.render(doc, style="ent") #this needs to be closed


# ### POS Tagging
# #### Categorizing each token according its appropriate Part of Speech could help in better understanding of data and can help in drawing accurate relations with other tokens. This process is called POS Tagging. You can see each tag's meaning [here](https://universaldependencies.org/docs/u/pos/).

# In[ ]:


i=0
for token in doc:
    if i<10:
        print(token.text, token.pos_)
    i+=1


# ### Dependency Parsing diagram
# I won't be covering Dependency Parsing topic in detail as it requires a separate notebook since it is an enormous topic in itself.

# In[ ]:


doc1 = nlp("Gandhiji was born in Porbandar in 1869.")

displacy.render(doc1, style="dep")

#sentence_spans = list(doc.sents) # To show large text dependency parsing, uncomment this.


# ### Word Cloud
# #### Word cloud can be used to see how frequently a particular category of words appear in our text. We will test it for seeing unique nouns as an example.

# In[ ]:


def worldcloud(word_list):
    #wordcloud = WordCloud()
    #wordcloud.fit_words(dict(count(word_list).most_common(40)))
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                
                min_font_size = 10).generate(word_list)

    fig=plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# In[ ]:


worldcloud(' '.join([token.text for token in doc if token.pos_ in ['NOUN']]))


# ### Vectorization
# #### We will be using Word2vec pretrained model to get the vector for our text. We can also use spacy's vectorizer but just wanted to demonstrate word2vec.
# 
# #### An average of all the token embeddings would be really helpful while doing further solution.

# In[ ]:


import gensim

word2vec_path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# In[ ]:


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
   
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


# In[ ]:


tokens_list= stemming(tokenizer(preprocess_data("@water #dream hi 19 :) hello where are you going be there tomorrow happening")))


# In[ ]:


tokens_list


# In[ ]:


get_average_word2vec(tokens_list, word2vec) #This will give an average of all of the token

