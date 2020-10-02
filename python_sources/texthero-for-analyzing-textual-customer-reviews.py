#!/usr/bin/env python
# coding: utf-8

# ![Texthero.png](attachment:Texthero.png)

# 

# ![rep.PNG](attachment:rep.PNG)

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


#Installing the required package
pip install texthero


# In[ ]:


#If you have already installed it and want to upgrade to the last version type
#pip install texthero -U


# In[ ]:


import texthero as hero
import pandas as pd
import csv
import matplotlib as plt


# In[ ]:


#Importing only the Product Name and the textual reviews of the customer into the Dataframe for Text Mining
df = pd.read_csv('/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')

df['Product']=df['name']
df['Review']=df['reviews.text']

df=df[['Product','Review']]


# In[ ]:


df.head(5)


# In[ ]:


#Step 1: Cleaning the textual reviews with hero.clean
df['clean_text'] = hero.clean(df['Review'])


# In[ ]:


#Step 2: This step performs the same cleaning as Step 1. However, it takes into account only certain pre-processing cleaning steps as indicated "custom"
from texthero import preprocessing

custom       = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_digits,
                   preprocessing.remove_punctuation,
                   preprocessing.remove_diacritics,
                   preprocessing.remove_stopwords,
                   preprocessing.stem]
df['clean_text'] = hero.clean(df['Review'])


# In[ ]:


#Step 3: Adding Term Frequency vector and PCA values into the data
df['tfidf_clean_text'] = hero.tfidf(df['clean_text'])
df['pca_tfidf_clean_text'] = hero.pca(df['tfidf_clean_text'])


# In[ ]:


#Step 4: This step peforms the same function as Step1 and Step 3 combined together with the help of pipe function in pandas
df['pca'] = (
    
            df['clean_text']
            .pipe(hero.clean)
            .pipe(hero.tfidf)
            .pipe(hero.pca)
   )


# In[ ]:


df.head()


# In[ ]:


#Scatter plot of PCA vector values
hero.scatterplot(df, col='pca', color='Product', title="PCA Amazon Product Reviews")


# ![PCA.PNG](attachment:PCA.PNG)

# In[ ]:


#Pulling the top words by product
NUM_TOP_WORDS = 10
df.groupby('Product')['clean_text'].apply(lambda x: hero.top_words(x)[:NUM_TOP_WORDS])

#If you take the first product, All-New Fire HD 8 Tablet- you can see that customers feel great about the product and its easy to use


# In[ ]:


#Plot 2: Plotting K-means
#pipe is the Pandas function used when chaining together functions

df['tfidf2'] = (
    df['Review']
    .pipe(hero.clean)
    .pipe(hero.tfidf)
)
df['kmeans_labels'] = (
    df['tfidf2']
    .pipe(hero.kmeans, n_clusters=5) #Defining 5 clusters (Forcing the algorithm to use only 5 clusters)
    .astype(str)
)
df['pca_k'] = df['tfidf2'].pipe(hero.pca)
hero.scatterplot(df, 'pca_k', color='kmeans_labels', title="K-means Plot for Amazon Customer Reviews")


# ![k%20means.PNG](attachment:k%20means.PNG)

# In[ ]:


#TextHero library we used is still in beta version and future release is expected to come with more features
#With just few lines of code, TextHero made is easier to analyze the data, group them into clusters and get insights from the data

