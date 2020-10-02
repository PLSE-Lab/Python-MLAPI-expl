#!/usr/bin/env python
# coding: utf-8

# ## Motivation for this Notebook
# 
# If I was a business owner, I would want to know how my customers are generally feeling. After reading a couple of reviews, you can start to pick up on some trends but who has the time to go through all of the comments to get a full picture of what people are saying about the company? Well luckily we have the power of NLP and Machine Learning algorithms that can do this compiling and grouping for us. Here I try to get a better look into 'average' reviews for a particular business and what's being said in them by implementing kMeans clustering.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import textblob as tb


# In[ ]:


review_df = pd.read_csv('../input/yelp_review.csv')
business_df = pd.read_csv('../input/yelp_business.csv')


# Adding the name of the business. 

# In[ ]:


name_df = business_df[['business_id', 'name']]


# In[ ]:


review_df = pd.merge(review_df, name_df, how = 'left', left_on = 'business_id', right_on = 'business_id')
review_df.head()


# I want to stem the words so we're not getting various forms of words that basically have the same meaning. I also tokenize the tokens so we're only getting words, including those with apostrophes. Below is a function to pass through as an argument in the TfidfVectorizer to override the tokenizing and to add the stemming. 

# In[ ]:


snowball = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [snowball.stem(word) for word in tokenizer.tokenize(text.lower())]


# Below is a function that will vectorize the words of the corpus (in this case, the group of reviews for the business). The function returns the resulting tfidf matrix and the complete list of words used in all of the reviews. 

# In[ ]:


def vectorize_reviews(reviews):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize, max_features = 1000)
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    return X, words


# Each cluster that kMeans finds is a general topic of the reviews as a whole and is represented by words or groups of words. Each dimension in the cluster center coordinates is the relative frequency for a word in that cluster. We can find the indices of the words with highest frequency in each cluster and these indices correspond to their respective word in the array of tokens. That way we can take a look at the words that represent the clusters the most and get an idea of what the latent topics are. 

# In[ ]:


def print_clusters(company_id, K = 8, num_words = 10):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    reviews = company_df['text'].values
    X, words = vectorize_reviews(reviews)
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' +           company_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# Let's try this out with an actual business. I love Tacos El Gordo so let's focus on the location in Downtown Las Vegas. 
# 
# First taking a look at the distribution of the ratings at this location. A lot of higher stars so we can probably expect that the ratings will have some pretty positive things said in them. Let's see if we can take a peek...
# 
# For now I randomly chose 4 clusters and the top 12 words in each one. Although we can see words that are commonly used, it's sort of difficult to put together what people are really saying. The first cluster might represent the positive feelings the customers generally have. The second seems to represent the long lines but how people still think it's worth it despite the wait time. The other clusters could be talking about the top types of tacos they like to order

# In[ ]:


#Tacos El Gordo in Downtown Las Vegas
bus_id = 'CiYLq33nAyghFkUR15pP-Q'
company_df = review_df[review_df['business_id'] == bus_id]
sns.countplot(x = company_df['stars'])


# In[ ]:


print_clusters(bus_id, K = 5, num_words = 12)


# For my initial analysis, I did not look at any ngrams. Including ngrams will allow us to see what groups of adjacent words in reviews are used frequently in reviews and more importantly, will give us a clearer and structured understanding of what people are saying; rather than just looking at single words in random order, hopefully the algorithm can identify groups of words that will be more coherent.
# 
# Adjusting some of the TfidfVectorizer arguments...

# In[ ]:


def vectorize_reviews2(reviews):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize,                         min_df = 0.0025, max_df = 0.05, max_features = 1000, ngram_range = (1, 3))
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    return X, words


# In[ ]:


def print_clusters2(company_id, K = 8, num_words = 10):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    reviews = company_df['text'].values
    X, words = vectorize_reviews2(reviews)
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' +           company_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# In the clusters above, it was hard to put the words together to form an idea of what was being said. For example, in cluster 3, are people saying it's a great place? Great food? Great price? Using ngrams let's us figure that out a little better by including groups of adjacent words in our vectorizer. 
# 
# Looking at the clusters below (again random K and num_words), people are saying it's worth the wait, they talk about the al pastor tacos a lot, etc. It still might not be the best view but it's better than guessing putting the words together on our own.

# In[ ]:


print_clusters2(bus_id, K = 3, num_words = 12)


# I attempted to use the elbow plot to determine what K is optimal to use. We want to choose the value of k that will minimize the within cluster variance, which is what the elbow plot graphs. However, we unfortunately don't always get a clear elbow, or an elbow at all as in this case, which means the data might not be easy to cluster or something else. It could be that since there's a skew in higher stars, that it's hard to cluster with more positive reviews. It would also be interesting to look at the silhouette score to see if that can better identify a value of K. 

# In[ ]:


def elbow_plot(X, k_start, k_end):
    
    distortions = []
    K = range(k_start, k_end + 1)
    for k in K:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(K, distortions)
    plt.xticks(K)
    plt.title('Elbow curve')
    


# No elbow :/

# In[ ]:


reviews = company_df['text'].values
X, words = vectorize_reviews2(reviews)
elbow_plot(X, 1, 10)


# Let's look at a business that has a wider distribution of star values. I'll choose Ginseng Korean BBQ II.

# In[ ]:


agg = review_df.groupby('name').filter(lambda x: len(x) > 100)
agg = agg.groupby('name')['stars'].mean()
agg[agg == 3.0]


# In[ ]:


review_df[review_df['name'] == '"Ginseng Korean BBQ II"']


# Even with ngrams, it's still a little hard to understand what people are saying. Sounds like some people think there's great food, while others think it's the worst korean food and poor service. This could potentially help the owner identify problems they need to address.

# In[ ]:


bus_id2 = 'EkuSy_kM8dpGrlb2pTxCBw'
company_df2 = review_df[review_df['business_id'] == bus_id2]
sns.countplot(x = company_df2['stars'])


# In[ ]:


print_clusters2(bus_id2, K = 3, num_words = 20)


# I was hoping with the wider range of review types that there would be a clearer elbow plot but doesn't look like it...

# In[ ]:


reviews2 = company_df2['text'].values
X2, words2 = vectorize_reviews2(reviews2)
elbow_plot(X, 1, 10)


# ## Sentiment Analysis
# 
# Just taking a look at average polarity and subjectivity the two restaurants we looked at have. Both have positive polarity but Tacos El Gordo has a higher average, which makes sense since it has more positive reviews.

# In[ ]:


def calc_polarity(text):
    blob = tb.TextBlob(text)
    return blob.sentiment.polarity

def calc_subjectivity(text):
    blob = tb.TextBlob(text)
    return blob.sentiment.subjectivity


# In[ ]:


def get_pol_sub(company_id):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    company_df['polarity'] = company_df['text'].apply(calc_polarity)
    company_df['subjectivity'] = company_df['text'].apply(calc_subjectivity)
    
    print('Company:' + company_name + '\nMean Polarity: ' + str(company_df['polarity'].mean())          + '\nMean Subjectivity: ' + str(company_df['subjectivity'].mean()))


# In[ ]:


get_pol_sub(bus_id)


# In[ ]:


get_pol_sub(bus_id2)


# ## Further Investigation
# 
# That was a preliminary analysis to get an idea of what common things are said about your business. 
# 
# We could further investigate by looking at a couple of reviews that have words found in our clusters to get that 'better understanding' of what's really being said. That way, you can see what things you're doing well and things you might want to improve on. Another thing is to look at only reviews with 3 or so stars and less (or above) and cluster from there to narrow down bad and good topics. 
# 
# Any suggestions with the ngrams or analysis would be great!

# In[ ]:




