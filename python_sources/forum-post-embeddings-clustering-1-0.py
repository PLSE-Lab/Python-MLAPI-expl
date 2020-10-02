#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Imports (code & data)
import re
import pandas as pd
#import yake_helper_funcs as yhf
from datetime import datetime, timedelta
from math import sqrt, floor
from sklearn.cluster import SpectralClustering
import numpy as np
import itertools
from flashtext.keyword import KeywordProcessor
import string
import nltk
import math
import join_forum_post_info as jfpi
import kaggle_specific_embed_cluster as embed_cluster
import get_surprising_words as surprising_words

# forum-wide frequency info for identifying surprising words
frequency_table = pd.read_csv("../input/kaggle-forum-term-frequency-unstemmed/kaggle_lex_freq.csv",
                             error_bad_lines=False)


# In[ ]:


## Utility functions

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


def polite_post_index(forum_posts):
    '''Pass in a list of fourm posts, get
    back the indexes of short, polite ones.'''
    
    polite_indexes = []
    
    # create  custom stop word list to identify polite forum posts
    stop_word_list = ["no problem", "thanks", "thx", "thank", "great",
                      "nice", "interesting", "awesome", "perfect", 
                      "amazing", "well done", "good job", "good work",
                      "congrats", "+1", "you're welcome", "good one",
                      "you are welcome", "good", "wow", "congrats",
                      "thnx", "my pleasure", "congratulations", "welcome",
                      "brilliant"
                     ]

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
days_of_posts = 4

# do you want to inclue comments on notebooks/scripts?
include_kernel_comments = False


# # Preprocessing posts

# In[ ]:


# read in and join forum post info
forums_info_df, forum_posts_df, forum_topics_df = jfpi.read_in_forum_tables()

# data validation
jfpi.check_column_names_forum_posts(forum_posts_df)
jfpi.check_column_names_forum_forums(forums_info_df)
jfpi.check_column_names_forum_topics(forum_topics_df)

# join info from different tables
posts_and_topics_df = jfpi.join_posts_and_topics(forum_posts_df, forum_topics_df)
forum_posts = jfpi.join_posts_with_forum_title(posts_and_topics_df, forums_info_df)


# In[ ]:


# remove kernel comments if desired
if include_kernel_comments == False:
    forum_posts = forum_posts[forum_posts["ForumTitle"] != "Kernels"]

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
sample_posts = sample_posts.drop(polite_posts)
sample_post_info = sample_post_info.drop(polite_posts)


# In[ ]:


# add URLs of each post to info

# add column for URLS
sample_post_info["url"] = ""

# info w/ forum titles and URL abbreviations
forum_titles = ["Kaggle Forum","Getting Started", "Product Feedback", 
                "Questions & Answers", "Datasets", "Learn"]
forum_title_abbrvs = {"Kaggle Forum":"general",
                "Getting Started":"getting-started",
                "Product Feedback":"product-feedback",
                "Questions & Answers":"questions-and-answers",
                "Datasets":"data",
                "Learn":"learn-forum"}

# add URLs to posts in main forums
for index, row in sample_post_info.iterrows():
   if row["ForumTitle"] in forum_titles:
        forum_name = row["ForumTitle"]
        forum_abbrv = forum_title_abbrvs[forum_name]
        post_id = row["ForumPostId"]
        topic_id = row["ForumTopicId"]
        
        post_url = (f'<a href="https://www.kaggle.com/{forum_abbrv}/{topic_id}#{post_id}">kaggle.com/{forum_abbrv}/{topic_id}#{post_id}</a>')
                
        sample_post_info.at[index,'url'] = post_url


# In[ ]:


# number of posts
num_of_posts = sample_posts.shape[0]

# number of clusters is square root of the # of posts (rounded down)
number_clusters = floor(sqrt(num_of_posts))

# how many posts are we clustering?
print(f"You're looking at {num_of_posts} posts.")


# In[ ]:


# check out the first few rows:
sample_post_info.head()


# # Embed & cluster forum posts

# In[ ]:


clustering = embed_cluster.get_spectral_clusters(sample_posts=sample_posts)


# # Generate HTML report of clusters

# In[ ]:


# count of posts/cluster
cluster_counts = pd.Series(clustering.labels_).value_counts()


# In[ ]:


# look at distrobution of cluster labels
size_df = pd.Series(clustering.labels_).value_counts().to_frame()

size_df = size_df.rename(columns={0: "size"})

size_df['characteristic_words'] = 0
size_df['characteristic_words'] = size_df['characteristic_words'].astype(object)

size_df["cluster_label"] = size_df.index

size_df['link_to_posts'] = ""

for index, row in size_df.iterrows():
    current_cluster_label = row["cluster_label"]
    link_to_posts = (f'<a href="#anchor_{current_cluster_label}">Link to posts</a>')
    size_df.at[index,'link_to_posts'] = link_to_posts

for i in size_df.index:
    words = surprising_words.get_surprising_words(cluster_index=i,
                                 post_data=sample_post_info, 
                                 cluster_object=clustering).tolist()
    size_df.at[i,'characteristic_words'] = words


# In[ ]:


size_df


# In[ ]:


# write suprising words & contents of cluster in file
with open("cluster_report.html", 'w', encoding="utf-8") as file:
    # file header
    file.writelines('<meta charset="UTF-8">\n')
    
    # add cluster info
    file.write(size_df.drop(["cluster_label"], axis=1).to_html(escape=False))
    file.write("\n")
                    
    for i in range(number_clusters):
        if i in size_df.index:
            file.write(f'\n<h2 id="anchor_{i}">Cluster {i}:</h2>\n')

            cluster_info = get_post_info_by_cluster(i,
                                                    data = sample_post_info,
                                                    cluster = clustering)

            cluster_info = cluster_info.drop(['ForumTopicId','PostDate','ForumPostId',
                                              'ForumId', 'Date'], axis=1)

            # truncate posts
            cluster_info.Message = cluster_info.Message.apply(lambda x: x[:100])

            file.write(cluster_info.to_html(escape=False))

