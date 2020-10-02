# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
## Imports (code & data)
import re
import pandas as pd
from datetime import datetime, timedelta
from math import sqrt, floor
from sklearn.cluster import SpectralClustering
import numpy as np
import itertools
from matplotlib import pyplot as plt
from flashtext.keyword import KeywordProcessor
import string
import nltk
import math

# %% [code]
#pip install csvvalidator

#from csvvalidator import *

# %% [markdown]
# # For sample posts, get forum title and topic title
# # based on queries from https://www.kaggle.com/pavlofesenko/strategies-to-earn-discussion-medals
# forum_topics_df = pd.read_csv('../input/meta-kaggle/ForumTopics.csv').rename(columns={'Title': 'TopicTitle'})
# forums_info_df = pd.read_csv('../input/meta-kaggle/Forums.csv').rename(columns={'Title': 'ForumTitle'})
# forum_posts_df = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")

# %% [code]
def check_column_names(input_df, column_names):
    """Header check doesn't work, don't use
    
    TODO: fix
    """
    # create a validator object using the column names
    posts_validator = CSVValidator(column_names)
    posts_validator.add_header_check()
    
    posts_validation = posts_validator.validate(input_df)
    
    if posts_validation == []:
        return(True)
    else:
        return(posts_validation)

# %% [code]
#check_column_names(input_df=forum_posts_df,
#                  column_names=('ForumTopicId','Message'))


# %% [code]
def join_posts_and_topics(forum_posts_df, forum_topics_df):
    """Join info from the ForumTopics & ForumMessages tables in MetaKaggle"""
    posts_and_topics_df = pd.merge(forum_posts_df[['ForumTopicId', 'PostDate', 'Message']], 
                   forum_topics_df[['Id', 'ForumId', 'TopicTitle']], 
                   left_on='ForumTopicId', right_on='Id')
    posts_and_topics_df = posts_and_topics_df.drop(['ForumTopicId', 'Id'], axis=1)
    
    return(posts_and_topics_df)

def join_posts_with_forum_title(posts_and_topics_df,
                                forum_info_df):
    """Join info from join_posts_and_topics() output and MetaKaggle Forums table """
    forum_posts = pd.merge(posts_and_topics_df, 
                           forum_info_df[['Id', 'ForumTitle']], 
                           left_on='ForumId', right_on='Id')
    forum_posts = forum_posts.drop(['ForumId', 'Id'], axis=1)
    
    return(forum_posts)

# %% [markdown]
# posts_and_topics_df = join_posts_and_topics(forum_posts_df=forum_posts_df,
#                                               forum_topics_df=forum_topics_df)
# forum_posts = join_posts_with_forum_title(posts_and_topics_df=posts_and_topics_df,
#                                          forum_info_df=forums_info_df)
#     
# forum_posts.head()