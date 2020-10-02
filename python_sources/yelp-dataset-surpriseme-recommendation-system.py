#!/usr/bin/env python
# coding: utf-8

# Sometimes, you don't want Yelp to recommend "similar" restaurants per se, but rather restaurants that worth exploring given your rating of the current restaurant. The problem with many recommendation systems including those in popular apps is their focus on *similarity* -- not the differences. Ultimately, however, I don't want to eat the same or have the same experience, but rather I would like to find a "surprisingly good" restaurant that may be a little different. This notion might be a bit hard to quantify, but in this tutorial, I would like to explore very simple techniques for doing this using Yelp Dataset.

# We will explore the techniques later in details, but here I would like to highlight the core idea: First, we want to find a way to represent reviews using a bag-of-words representation. After doing so, we will also represent categories using a one-hot encoding representation. Then, we can manipulate those representations to find similarities and differences while balancing the weights of the two. Note that the core idea assume that you are more likely to love a restaurant if its reviews are similar to the reviews of the restaurants you already love.

# Let's begin by importing libraries and making sure we only deal with valid data.

# In[ ]:


import os
import re
import string

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


df_yelp_business = pd.read_json('../input/yelp_academic_dataset_business.json', lines=True)
df_yelp_business.fillna('NA', inplace=True)
# we want to make sure we only work with restaurants -- nothing else
df_yelp_business = df_yelp_business[df_yelp_business['categories'].str.contains('Restaurants')]
print('Final Shape: ',df_yelp_business.shape)


# Now we bring the reviews and perform some preprocessing on those reviews..

# In[ ]:


df_yelp_review_iter = pd.read_json('../input/yelp_academic_dataset_review.json', chunksize=100000, lines=True)


# Because reviews are too big, we will read them in chunks, and make sure we delete reviews of places that are not in our list of businesses filtered earlier. Note here we choose 5 chunks, but we could have chosen any number (larger numbers will give MemoryError later on).

# In[ ]:


df_yelp_review = pd.DataFrame()
i=0
for df in df_yelp_review_iter:
    df = df[df['business_id'].isin(df_yelp_business['business_id'])]
    df_yelp_review = pd.concat([df_yelp_review, df])
    i=i+1
    print(i)
    if i==4: break


# Also make sure we only get businesses that already show up in our review list and delete the rest.

# In[ ]:


df_yelp_business = df_yelp_business[df_yelp_business['business_id'].isin(df_yelp_review['business_id'])]


# In[ ]:


print('Final businesses shape: ', df_yelp_business.shape)
print('Final review shape: ', df_yelp_review.shape)


# Now we want to processes reviews in a reasonable way. The following function is adopted from [here](https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb) which really does a good deal to preprocess the text.

# In[ ]:


def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    return text


# The next step will apply those transformations. Note that it will take a couple of minutes to finish.

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_yelp_review['text'] = df_yelp_review['text'].apply(clean_text)")


# Now we want to vectorize both reviews and categories. Note that min_df and max_df arguments in both.

# In[ ]:


vectorizer_reviews = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
vectorized_reviews = vectorizer_reviews.fit_transform(df_yelp_review['text'])


# In[ ]:


print(vectorized_reviews.shape)


# Show top 100 vocabularies:

# In[ ]:


' | '.join(vectorizer_reviews.get_feature_names()[:100]) # only the first 100


# In[ ]:


vectorizer_categories = CountVectorizer(min_df = 1, max_df = 1., tokenizer = lambda x: x.split(', '))
vectorized_categories = vectorizer_categories.fit_transform(df_yelp_business['categories'])


# In[ ]:


print(vectorized_categories.shape)


# We also show 100 categories..

# In[ ]:


' | '.join(vectorizer_categories.get_feature_names()[:100]) # only the first 100


# We will use sparse representations to make dot products easier to speed up dot products (and also save memory).

# In[ ]:


get_ipython().run_cell_magic('time', '', "from scipy import sparse\nbusinessxreview = sparse.csr_matrix(pd.get_dummies(df_yelp_review['business_id']).values)")


# Let's print out the shapes of the matrices we have prepared and make sure they make sense (by matching their dimensions):

# In[ ]:


print('restuarants x categories: \t', vectorized_categories.shape) 
print('restuarants x reviews: \t\t' , businessxreview.shape) 
print('reviews x words: \t\t', vectorized_reviews.shape)


# Now we are ready to choose a seed restaurant and find  other restaurants that might be as good as the seed restaurant. We make sure to choose a restaurant with good number of reviews and ratings.

# In[ ]:


# to choose a restaurant, just copy the business id and paste it in the next cell
# you can always rerun the cell to choose another restuarant. 
df_yelp_business.sample(10)


# In[ ]:


business_choose = 'aUrOyWFKxKeVXiFzwbTXSA' # vegan, vegetarian, cafes


# First, we pull up the reivews and then show some of them

# In[ ]:


new_reviews = df_yelp_review.loc[df_yelp_review['business_id'] == business_choose, 'text']


# In[ ]:


print('\n'.join([r[:100] for r in new_reviews.tolist()])) # restaurant reviews


# Then we pull up the categories:

# In[ ]:


new_categories = df_yelp_business.loc[df_yelp_business['business_id'] == business_choose, 'categories']


# In[ ]:


new_categories.tolist() #  restaurant categories


# Here, we compute two sets of distancecs: we compute the correlation distance of the average vectorized reviews to all the reviews, and compute the correlation distance between this category and all other categories. The category trick will be clearer when we see the results.

# In[ ]:


from scipy.spatial.distance import cdist
# find most similar reviews
dists1 = cdist(vectorizer_reviews.transform(new_reviews).todense().mean(axis=0), 
              vectorized_reviews.T.dot(businessxreview).T.todense(), 
               metric='correlation')
# find most similar categories
dists2 = cdist(vectorizer_categories.transform(new_categories).todense().mean(axis=0), 
              vectorized_categories.todense(), 
               metric='correlation')


# Now we combine the two sets of distances and take the average of those (we can take other metrics such as min or max, depending on your priority).

# In[ ]:


# combine the two vectors in one matrix
dists_together = np.vstack([dists1.ravel(), dists2.ravel()]).T


# In[ ]:


dists_together


# In[ ]:


# this is a key cell: how are we going to prioritize ?
dists = dists_together.mean(axis=1)


# In[ ]:


dists


# Let's select the closest 10 restaurants to the seed restaurant.

# In[ ]:


# select the closest 10
closest = dists.argsort().ravel()[:10]


# Here is our seed restaurant:

# In[ ]:


df_yelp_business.loc[df_yelp_business['business_id']== business_choose, ['business_id', 'categories', 'name', 'stars']]


# Now let's see what the top matches:

# In[ ]:


df_yelp_business.loc[df_yelp_business['business_id'].isin(df_yelp_business['business_id'].iloc[closest]), ['business_id', 'categories', 'name', 'stars']]


# Although many of those seem to come from the same category (Vegetarian and Vegan), there is a considerable variation in those categories (Ethiopian, African, Asian Fusion, .. etc). Most importantly, they all seem highly rated.
# 
# Would you go and try those out and be little surprised ? It is really up to you.
