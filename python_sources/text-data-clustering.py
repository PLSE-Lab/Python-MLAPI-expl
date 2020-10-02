#!/usr/bin/env python
# coding: utf-8

# Hi there! This is my first kernel dealing with textual data so any constructive feedabacks are higly appreciated.
# 
# This dataset contains data of over 7 topics namely biology, robotics, cryptography, diy, travel, cooking, robotics and physics extracted from Stack Exchange. Each of these topics except physics have been classified as to which topic data belongs. So our task is to do predictions on unseen physics questions.
# 
# Since our data won't be related to each other for example tags in travel won't be related to tags in cryptography hence I will be using unsupervised learning on physics dataset which is the test dataset. 

# In[ ]:


#Importing all the neccesary libraries
import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import regex as re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer


# Making dictionary to put all the data in the same hood.

# In[ ]:


data={'bio':pd.read_csv('../input/biology.csv',index_col=0),
      'robo':pd.read_csv('../input/robotics.csv',index_col=0),
      'cryp':pd.read_csv('../input/crypto.csv',index_col=0),
      'diy':pd.read_csv('../input/diy.csv',index_col=0),
      'cooking':pd.read_csv('../input/cooking.csv',index_col=0),
      'travel':pd.read_csv('../input/travel.csv',index_col=0),
      'test':pd.read_csv('../input/test.csv',index_col=0),
     }
data['robo']


# **Text data preprocessing steps**
# 
# 1- Data Cleaning(either using regex or BeautifulSoup): 
# a) Removing HTML characters. 
# b) Removing punctuation. 
# c) Decoding encoded data.
# d) Split attached words.
# e) Removing URLs. 
# f) Apostrophe removal.
# g) Removing Expressions. 
# h) Uppercase & Lowercase letters 
# i) Numbers such as amounts and data.
# 
# 2- Data Tokenization(using word_tokenize in nltk.tokenize) 
# Segregation of text into individual words i.e tokens.
# 
# 3- Stopword Removal(using stopwords in nltk.corpus)
# Discarding too common words or words which are not going to be helpful in our analysis.
# 
# 4- Stemming(using WordNetLemmatizer in nltk.stem) 
# Combining different variants of words into a single parent word that conveys same meaning.
# 
# 5-Vectorization (either using TfidVectorizer or Countvectorizer in sklearn.feature_extraction.text or word embeddings) Changing text data into vector format.
# 

# In[ ]:


stops = set(stopwords.words("english"))


# In[ ]:


def clean_content(table):
    content = table.content
    #Converting text to lowercase characters
    content = content.apply(lambda x: x.lower())
    #Removing HTML tags
    content = content.apply(lambda x: re.sub(r'\<[^<>]*\>','',x))
    #Removing any character which does not match to letter,digit or underscore
    content = content.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
    #Removing space,newline,tab
    content = content.apply(lambda x: re.sub(r'\s',' ',x))
    #Removing punctuation
    content = content.apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
    #Tokenizing data
    content = content.apply(lambda x: word_tokenize(x))
    #Removing stopwords
    content = content.apply(lambda x: [i for i in x if i not in stops])
    return(content)


# Doing the cleaning process on title as well

# In[ ]:


def clean_title(table):
    title = table.title
    title = title.apply(lambda x: x.lower())
    title = title.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
    title = title.apply(lambda x: re.sub(r'\s',' ',x))
    title = title.apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
    title = title.apply(lambda x: word_tokenize(x))
    title = title.apply(lambda x: [i for i in x if i not in stops])
    return(title)


# Applying operations on data

# In[ ]:


for df in data:
    data[df].content = clean_content(data[df])


# In[ ]:


for df in data:
    data[df].title = clean_title(data[df])


# In[ ]:


data['robo']


#  Visualizing our cleaned robo data using WordCloud

# In[ ]:


text = ' '
for x in data['robo'].content:
    for y in x:
        text+=' '+y


# In[ ]:


plt.figure(figsize=(8,10))
wc = WordCloud(max_words=1000,random_state=1).generate(text)
plt.imshow(wc)
plt.show()


# WordCloud for cooking data

# In[ ]:


cooking = ' '
for x in data['cooking'].title:
    for y in x:
        cooking+=' '+y


# In[ ]:


plt.figure(figsize=(8,10))
wf = WordCloud(background_color='white',max_words=1000,random_state=1).generate(cooking)
plt.imshow(wf)
plt.show()


# WordCloud for cryptography

# In[ ]:


crypt = ' '
for i in data['cryp'].content:
    for j in i:
        crypt+=' '+j


# In[ ]:


plt.figure(figsize=(8,10))
wg = WordCloud(background_color='black',max_words=1000,random_state=1).generate(crypt)
plt.imshow(wg)
plt.show()


# **Stemming ** 

# In[ ]:


wordnet = WordNetLemmatizer()
data['test'].title = data['test'].title.apply(lambda x:[wordnet.lemmatize(i,pos='v') for i in x])
data['test'].content = data['test'].content.apply(lambda x:[wordnet.lemmatize(i,pos='v') for i in x])


# Finally let's see how our test data look likes

# In[ ]:


tst = ' '
for i in data['test'].title:
    for j in i:
        tst+=' '+j     


# WordCloud for physics dataset

# In[ ]:


plt.figure(figsize=(8,10))
phy = WordCloud(background_color='white',max_words=1000,random_state=1).generate(tst)
plt.imshow(phy)
plt.show()


# Vectrorizing data using TfidVectrorizer which uses the concept of term frequency and inverse document frequency to get rid of all non-consequential tokens from being vectorized.
# For more details see https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms

# In[ ]:


def identity_tokenizer(text):
  return text
vect = TfidfVectorizer(tokenizer=identity_tokenizer,lowercase=False)
x = vect.fit_transform(data['test'].title.values)


# In[ ]:


indices = np.argsort(vect.idf_)[::-1]
features = vect.get_feature_names()
top_n = 50
top_features = [features[i] for i in indices[:top_n]]
top_features


# **k-means clustering**
# 
# In general, k-means is the first choice for clustering because of its simplicity. Here, the user has to define the number of clusters (Post on how to decide the number of clusters would be dealt later). The clusters are formed based on the closeness to the center value of the clusters. The initial center value is chosen randomly. K-means clustering is top-down approach, in the sense, we decide the number of clusters (k) and then group the data points into k clusters.

# In[ ]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
model.fit(x)


# In[ ]:


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(20):
    print ("Cluster %d:" % i,)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)
    


# So from the output we can infer following points:
# 
# Cluster 1 classifies text related to 'angular momentum', 'torque' which can be associated to 'motor'
# 
# Cluster 2 is related to 'visible light source' 
# 
# Cluster 3 deal with 'kinetic' and ' potential' energy which can be used to explain 'energy conservation'
# 
# Cluster 4 possibly relates to 'physics equations'
# 
# Cluster 10 has term like 'singularity' which is related to 'black hole'
# 
# Similarly we can draw other conclusions too.
# 
# Any feedbacks to improve it further are appreciated.
# 
# Thank you!

# In[ ]:




