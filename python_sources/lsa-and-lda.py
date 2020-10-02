#!/usr/bin/env python
# coding: utf-8

# ![photo](https://analyticssteps.com/backend/media/thumbnail/5248416/336284_1571738616_LSA-and-LDA-banner-image.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
import matplotlib                  # 2D Plotting Library        
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# In[ ]:


df = pd.read_csv("/kaggle/input/nlp-topic-modelling/Reviews.csv")


# In[ ]:


df.head()


# In[ ]:


index = df.index
number_of_rows = len(index)
print(number_of_rows)


# > As you can see amount of data is really huge so we will take small set of rows out of it so that it takes less time

# > we are only interested in Text column

# In[ ]:


df.drop(['ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score' , 'Time', 'Summary'],axis=1,inplace=True)


# In[ ]:


new1 = df[['Text']].iloc[:10000].copy()


# ### DATA CLEANING & PRE-PROCESSING
# Here I have done the data pre-processing. We can use any among the lemmatization and the stemming but i prefer to use lemmatiation. Also the stop words have been removed along with the words with length shorter than 3 characters to reduce some stray words.

# In[ ]:


def clean_text(text):
    le=WordNetLemmatizer()
    word_tokens=word_tokenize(text)
    tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
    cleaned_text=" ".join(tokens)
    return cleaned_text


# In[ ]:


#it takes time
new1['cleaned_text']=new1['Text'].apply(clean_text)


# In[ ]:


new1.head()


# You can clearly see the difference after removal of stopwords and some shorter words.

# > Now drop the unpre-processed column.

# In[ ]:


new1.drop(['Text'],axis=1,inplace=True)


# ### EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )
# A document-term matrix or term-document matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.
# 
# Some important points:-
# 
# 1) LSA is generally implemented with Tfidf values everywhere and not with the Count Vectorizer.
# 
# 2) max_features depends on your computing power and also on eval. metric (coherence score is a metric for topic model). Try the value that gives best eval. metric and doesn't limits processing power.
# 
# 3) Default values for min_df & max_df worked well.
# 
# 4) Can try different values for ngram_range.

# In[ ]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)


# In[ ]:


vect_text=vect.fit_transform(new1['cleaned_text'])


# In[ ]:


print(vect.get_feature_names())


# We can now see the most frequent and rare words in the cleaned_text column based on idf score. The lesser the value; more common is the word in the column.

# In[ ]:


print(vect_text.shape)
type(vect_text)


# In[ ]:



idf=vect.idf_


# In[ ]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['like'])
print(dd['fondant'])  # police is most common and forecast is least common among the news headlines.


# > We can therefore see that on the basis of the idf value , 'like' is the most frequent word while 'fondant' is least frequently occuring word.

# In[ ]:


new1['cleaned_text'].head()


# ### TOPIC MODELLING

# Latent Semantic Analysis (LSA)
# The first approach that I have used is the LSA. LSA is basically singular value decomposition.
# 
# SVD decomposes the original DTM into three matrices S=U.(sigma).(V.T). Here the matrix U denotes the document-topic matrix while (V) is the topic-term matrix.
# 
# Each row of the matrix U(document-term matrix) is the vector representation of the corresponding document. The length of these vectors is the number of desired topics. Vector representation for the terms in our data can be found in the matrix V (term-topic matrix).
# 
# So, SVD gives us vectors for every document and term in our data. The length of each vector would be k. We can then use these vectors to find similar words and similar documents using the cosine similarity method.
# 
# We can use the truncatedSVD function to implement LSA. The n_components parameter is the number of topics we wish to extract. The model is then fit and transformed on the result given by vectorizer.
# 
# Lastly note that LSA and LSI (I for indexing) are the same and the later is just sometimes used in information retrieval contexts.

# In[ ]:


lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[ ]:


print(lsa_top[0])
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[ ]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# > Similalry for other documents we can do this. However note that values don't add to 1 as in LSA it is not probabiltiy of a topic in a document.

# In[ ]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Now we can get a list of the important words for each of the 10 topics as shown. For simplicity here I have shown 10 words for each topic.

# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# ### Latent Dirichlet Allocation (LDA)
# LDA is the most popular technique.The topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.[](http://)

#  To understand LDA in detail please refer to [this](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/) blog.
#  

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
# n_components is the number of topics


# In[ ]:


lda_top=lda_model.fit_transform(vect_text)


# In[ ]:


print(lda_top.shape)  # (no_of_doc,no_of_topics)
print(lda_top[0])


# In[ ]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)


# > Note that the values in a particular row adds to 1. This is beacuse each value denotes the % contribution of the corressponding topic in the document.

# In[ ]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# As we can see Topic 0 is dominantly present in document 0.

# In[ ]:


print(lda_model.components_[0])
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# Most important words for a topic. (say 5 this time.)

# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# To better visualize words in a topic we can see the word cloud. For each topic top 25 words are plotted.

# In[ ]:


from wordcloud import WordCloud
# Generate a word cloud image for given topic
def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda_model.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:25]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=900, height=600).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


# In[ ]:


# topic 0
draw_word_cloud(0)

