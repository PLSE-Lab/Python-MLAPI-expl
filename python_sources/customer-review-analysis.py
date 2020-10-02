#!/usr/bin/env python
# coding: utf-8

# Customer Review Analysis using NLP/Text Analytics and spaCy  
# 
#  1. Data Analysis
#  2. Sentiment Analysis
#  3. Data Clustering
#  4. Word2vec
# 
# *** Facing some error in spaCy and word2vec**Working on it **

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from __future__ import unicode_literals


# In[ ]:


Amazon_Meta_Data = pd.read_csv('../input/Amazon_Unlocked_Mobile.csv', encoding='utf-8')


# In[ ]:


Amazon_Meta_Data.head(2)


# In[ ]:


Amazon_Meta_Data.columns


# In[ ]:


Amazon_Meta_Data.dtypes


# In[ ]:


Reviews = Amazon_Meta_Data['Reviews']
len(Reviews)


# **Top Review Counts With Brand**

# In[ ]:


Brand_Name = Amazon_Meta_Data['Brand Name'].str.upper()
Brand_Name.value_counts().head(10)


# **Mean and Median Price In Given Data**

# In[ ]:


Price = Amazon_Meta_Data['Price']
Price.mean()


# In[ ]:


Price.median()


# In[ ]:


get_ipython().run_line_magic('time', '')
table = pd.pivot_table(Amazon_Meta_Data,
            values = ['Price'],
            index = ['Brand Name'], 
                       columns= [],
                       aggfunc=[np.mean, np.median], 
                       margins=True)

#table


# **Review Ranting**

# In[ ]:


Customer_Ratings = Amazon_Meta_Data.groupby(
    'Brand Name'
    ).Rating.agg(
        ['count', 'min', 'max']
    ).sort_values(
        'count', ascending=False
    )
Customer_Ratings.head(15)


# In[ ]:


Product_Ratings = Amazon_Meta_Data.groupby(
    'Product Name'
    ).Rating.agg(
        ['count', 'min', 'max']
    ).sort_values(
        'count', ascending=False
    )
Product_Ratings.head(15)


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sample_review = Reviews[:10]    


# In[ ]:


sentiment = SentimentIntensityAnalyzer()
  


# ****Review Sentiment Score Using NLTK ****

# In[ ]:


for sentences in sample_review:
    sentences
    ss = sentiment.polarity_scores(sentences)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]))
    print(sentences)   


# **Kmeans Clutering **

# In[ ]:


Cluster_Data = pd.read_csv('../input/Amazon_Unlocked_Mobile.csv',nrows=6000)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
Cluster_Data.columns


# **Data Cleaning** 

# In[ ]:


from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.corpus import stopwords
def remove_stopword(word):
    return word not in words

from nltk.stem import WordNetLemmatizer
Lemma = WordNetLemmatizer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

Cluster_Data['NewReviews'] = Cluster_Data['Reviews'].str.lower().str.split()
Cluster_Data['NewReviews'] = Cluster_Data['NewReviews'].apply(lambda x : [item for item in x if item not in stop])
#Cluter_Data['NewReviews'] = Cluter_Data["NewReviews"].apply(lambda x : [stemmer.stem(y) for y in x])


# In[ ]:


Cluster_Data['Cleaned_reviews'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
for line in lists]).strip() for lists in Cluster_Data['NewReviews']] 


# **Columns**

# In[ ]:


Cluster_Data.columns


# **TF_IDF**

# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)


# In[ ]:


model = vectorizer.fit_transform(Cluster_Data['Cleaned_reviews'].str.upper())


# **KMeans**

# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=5,init='k-means++',max_iter=200,n_init=1)


# In[ ]:


km.fit(model)
terms = vectorizer.get_feature_names()
order_centroids = km.cluster_centers_.argsort()[:,::-1]
for i in range(5):
    print("cluster %d:" %i,end='')
    for ind in order_centroids[i,:10]:
        print(' %s' % terms[ind],end='')
    print()    


# In[ ]:





# In[ ]:





# In[ ]:




