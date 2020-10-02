#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 


# We Will Load the data

# In[3]:


data1 = pd.read_csv("../input/SW_EpisodeIV.txt",delim_whitespace = True,header = 0,escapechar='\\')
data2 = pd.read_csv("../input/SW_EpisodeV.txt",delim_whitespace = True,header = 0,escapechar='\\')
data3 = pd.read_csv("../input/SW_EpisodeVI.txt",delim_whitespace = True,header = 0,escapechar='\\')
data = pd.concat([data1,data2,data3],axis = 0)
data.head()


# In[15]:


import re
import nltk
from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer #lemmatize the word for example "studies" and "studying" will be converted to "study"


# Disadvantages of using Stemming

# In[ ]:


print(PorterStemmer().stem("trouble"))
print(PorterStemmer().stem("troubling"))
print(PorterStemmer().stem("troubled"))


# **Cleaning the Data**
# Step 1 : We will join the data
# Step 2 : we will  remove the punctuations,numbers which will not provide any information
# Step 3 : We will remove the words whose len is less than 3 and also the stopwords

# In[ ]:


def unigram(data):
    text = " ".join(data)
    CleanedText = re.sub(r'[^a-zA-Z]'," ",text)
    CleanedText = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(CleanedText) if word not in stopwords.words("english") and len(word) > 3])
    return CleanedText


# In[ ]:


CleanedText = unigram(data['dialogue'])


# In[ ]:


from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
wordcloud = WordCloud(random_state=21).generate(CleanedText)
plt.figure(figsize = (30,15))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# In[ ]:


def ngrams(data,n):
    text = " ".join(data)
    text1 = text.lower()
    text2 = re.sub(r'[^a-zA-Z]'," ",text1)
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram


# In[ ]:


ngram = ngrams(data['dialogue'],2)
ngram[1:10]


# We have to combine the two words for better visualisation

# 

# In[ ]:


"_".join(ngram[0])


# In[ ]:


for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])


# In[ ]:


Bigram_Freq = nltk.FreqDist(ngram)


# In[ ]:


bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
plt.figure(figsize = (50,25))
plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# In[ ]:


ngram = ngrams(data['dialogue'],3)


# In[ ]:


for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])


# In[ ]:


Trigram_Freq = nltk.FreqDist(ngram)


# In[ ]:


trigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Trigram_Freq)
plt.figure(figsize = (50,25))
plt.imshow(trigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# In[7]:


lda_data = []
for i in range(0,len(data)):
    lda_data.append(data.iloc[i,]['dialogue'])
    


# In[18]:


import string
exclude = set(string.punctuation)
def clean_doc(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stopwords.words("english")])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(punc_free)])
    return normalized


# In[19]:


doc_clean = [clean_doc(doc).split() for doc in lda_data] 
doc_clean[0]


# In[30]:


import gensim
from gensim import corpora
#Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(doc_clean)
#Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
dtm = [dictionary.doc2bow(doc) for doc in doc_clean]


# In[35]:


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
#num_topics = number of topics you want to extract from the corpus
ldamodel = Lda(dtm, num_topics=5, id2word = dictionary, passes=50)


# In[36]:


print(ldamodel.print_topics(num_topics=5, num_words=5))

