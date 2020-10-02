#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Imports (code & data)
import re
import pandas as pd
import yake_helper_funcs as yhf
from datetime import datetime, timedelta
from math import sqrt, floor
from sklearn.cluster import SpectralClustering
import numpy as np
import itertools
from matplotlib import pyplot as plt
import removing_polite_posts as rpp
from flashtext.keyword import KeywordProcessor
import string

forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")

# read in pre-tuned vectors
vectors = pd.read_csv("../input/fine-tuning-word2vec-2-0/kaggle_word2vec.model", 
                      delim_whitespace=True,
                      skiprows=[0], 
                      header=None
                     )

# set words as index rather than first column
vectors.index = vectors[0]
vectors.drop(0, axis=1, inplace=True)


# In[ ]:


## Utility functions

# get vectors for each word in post
# TODO: can we vectorize this?
def vectors_from_post(post):
    all_words = [] 

    for words in post:
        all_words.append(words) 
        
    return(vectors[vectors.index.isin(all_words)])


# create document embeddings from post
def doc_embed_from_post(post):
    test_vectors = vectors_from_post(post)

    return(test_vectors.mean())

# explore our posts by cluster
def get_keyword_set_by_cluster(number):
    cluster_index = list(clustering.labels_ == number)
    return(list(itertools.compress(keyword_sets, cluster_index)))

# get sample post info by #
def get_post_info_by_cluster(number, 
                             data,
                             cluster):
    return(data[cluster.labels_ == number])

# remove HTML stuff
# https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return(re.sub(clean, '', text))

# remove "good", "nice", "thanks", etc
def remove_thanks(text):
    text = text.lower()
    
    text = re.sub("nice", "", text)
    text = re.sub("thank.*\s", " ", text)
    text = re.sub("good","", text)
    text = re.sub("hi", "", text)
    text = re.sub("hello", "", text)
    
    return(text)

def polite_post_index(forum_posts):
    '''Pass in a list of fourm posts, get
    back the indexes of short, polite ones.'''
    
    polite_indexes = []
    
    # create  custom stop word list to identify polite forum posts
    stop_word_list = ["no problem", "thanks", "thx", "thank", "great",
                      "nice", "interesting", "awesome", "perfect", 
                      "amazing", "well done", "good job"]

    # create a KeywordProcess
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(stop_word_list)

    # test our keyword processor
    for i,post in enumerate(forum_posts):
        post = post.lower().translate(str.maketrans({a:None for a in string.punctuation}))
        
        if len(post) < 100:
            keywords_found = keyword_processor.extract_keywords(post.lower(), span_info=True)
            if keywords_found:
                polite_indexes.append(i)

    return(polite_indexes)


# In[ ]:


## Hyperprameters

# number of clusters currently based on the square root of the # of posts
days_of_posts = 1


# # Preprocessing posts

# In[ ]:


# For sample posts, get forum title and topic title
# based on queries from https://www.kaggle.com/pavlofesenko/strategies-to-earn-discussion-medals
topics = pd.read_csv('../input/meta-kaggle//ForumTopics.csv').rename(columns={'Title': 'TopicTitle'})
forums = pd.read_csv('../input/meta-kaggle/Forums.csv').rename(columns={'Title': 'ForumTitle'})

df1 = pd.merge(forum_posts[['ForumTopicId', 'PostDate', 'Message']], topics[['Id', 'ForumId', 'TopicTitle']], left_on='ForumTopicId', right_on='Id')
df1 = df1.drop(['ForumTopicId', 'Id'], axis=1)

forum_posts = pd.merge(df1, forums[['Id', 'ForumTitle']], left_on='ForumId', right_on='Id')
forum_posts = forum_posts.drop(['ForumId', 'Id'], axis=1)
forum_posts.head()


# In[ ]:


# parse dates
forum_posts['Date'] = pd.to_datetime(forum_posts.PostDate, format="%m/%d/%Y %H:%M:%S")

# posts from the last X days
start_time = datetime.now() + timedelta(days=-days_of_posts)  

# forum posts from last week (remember to convert to str)
sample_post_info = forum_posts.loc[forum_posts.Date > start_time]
sample_posts = sample_post_info.Message.astype(str)

# reindex from 0
sample_posts.reset_index(drop=True)
sample_post_info.reset_index(drop=True)

# remove html tags
sample_post_info.Message = sample_post_info.Message    .astype(str)    .apply(remove_html_tags)
sample_posts = sample_posts.apply(remove_html_tags)

# remove polite posts (make sure you remove HTML tags first)
polite_posts = sample_posts.index[polite_post_index(sample_posts)]
# posts aren't being dropped 
sample_posts = sample_posts.drop(polite_posts)
sample_post_info = sample_post_info.drop(polite_posts)

# number of posts
num_of_posts = sample_posts.shape[0]

# Number of clusters is square root of the # of posts (rounded down)
number_clusters = floor(sqrt(num_of_posts))


# In[ ]:


# extact keywords & tokenize
#keywords = yhf.keywords_yake(sample_posts, )
keywords_tokenized = yhf.tokenizing_after_YAKE(sample_posts)
keyword_sets = [set(post) for post in keywords_tokenized]


# # Get word vectors for keywords in post

# In[ ]:


# create empty array for document embeddings
doc_embeddings = np.zeros([num_of_posts, 300])

# get document embeddings for posts
for i in range(num_of_posts):
    embeddings = np.array(doc_embed_from_post(keyword_sets[i]))
    if np.isnan(embeddings).any():
        doc_embeddings[i,:] = np.zeros([1,300])
    else:
        doc_embeddings[i,:] = embeddings


# # Clustering!

# In[ ]:


# the default k-means label assignment didn't work well
clustering = SpectralClustering(n_clusters=number_clusters, 
                                assign_labels="discretize",
                                n_neighbors=number_clusters).fit(doc_embeddings)


# In[ ]:


# look at distrobution of cluster labels
pd.Series(clustering.labels_).value_counts()


# In[ ]:


for i in range(number_clusters):
    
    print(f"Cluster {i}:\n")
    print(get_post_info_by_cluster(i, 
                                   data = sample_post_info,
                                   cluster = clustering))
    print("\n")
    


# In[ ]:


for i in range(number_clusters):
    
    print(f"Cluster {i}:\n")
    print(get_keyword_set_by_cluster(i))
    print("\n")


# # Refining clustering
# 
# Steps:
# 
# 1. Drop empty clusters
# 2. Identify large clusters (2 times more than expected)
# 3. Recluster those clusters (# clusters = sqrt # posts)
# 
# 

# In[ ]:


# count of posts/cluster
cluster_counts = pd.Series(clustering.labels_).value_counts()

# get clusters bigger than expected
max_cluster_size = number_clusters * 2
big_clusters = cluster_counts[cluster_counts > max_cluster_size]


# In[ ]:


# sub-cluster first (biggest) cluster
cluster_label = big_clusters.index[0]

sub_sample = sample_post_info[clustering.labels_ == cluster_label]
sub_cluster_embeddings = doc_embeddings[clustering.labels_ == cluster_label]

number_sub_clusters = floor(sqrt(sub_sample.shape[0]))

sub_cluster = SpectralClustering(n_clusters=number_sub_clusters, 
                                 assign_labels="discretize", 
                                 n_neighbors=number_sub_clusters).fit(sub_cluster_embeddings)


# In[ ]:


# see how it looks
for i in range(number_sub_clusters):

    print(f"Cluster {i}:\n")
    print(get_post_info_by_cluster(i, data = sub_sample, 
                                   cluster = sub_cluster))
    print("\n")


# In[ ]:


pd.Series(sub_cluster.labels_).value_counts()


# # Word clouds

# In[ ]:


from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


# TODO why do I see thank you?
posts_as_string = sample_post_info    .Message    .to_string(index=False)

# shouldn't have to do this b/c I removed polite posts earlier
posts_as_string = remove_thanks(posts_as_string)

# Generate a word cloud image
wordcloud = WordCloud().generate(posts_as_string)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Going forward
# 
# Biggest problem: redundent clusters
# 
# Possible solutions: 
# 
# * Remove very short posts
# * Don't include posts on kernels
# * Build filter for removing short "thanks!" type posts
# * Start w/ sentiment analys & put all very high sentiment posts in a single bin

# # Visualization brain storming
# 
# Slides on text visualizatoin: https://courses.cs.washington.edu/courses/cse512/15sp/lectures/CSE512-Text.pdf
# 
# * Bigram based method, reporting the two terms with the median freuquency
# * term saliency, normalize by freq of most common term log(tf_w) / log(tf_the) (and then some sort of regression?)
# * Termite-based model: Topics as columns, terms as rows and weight visualiation of term distinctivenes as KL divergence p(T|term)/p(T|any_term)
