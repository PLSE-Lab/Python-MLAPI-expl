# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
import pandas as pd

# %% [code]
def read_in_forum_tables():
    """Read in the three forum-related files from MetaKaggle"""
    forums_info_df = pd.read_csv('../input/meta-kaggle/Forums.csv').rename(columns={'Title': 'ForumTitle'})
    forum_posts_df = pd.read_csv("../input/meta-kaggle/ForumMessages.csv").rename(columns={'Id': 'ForumPostId'})
    forum_topics_df = pd.read_csv('../input/meta-kaggle/ForumTopics.csv').rename(columns={'Title': 'TopicTitle'})
    
    return forums_info_df, forum_posts_df, forum_topics_df

# %% [code]
# data validation: check for the appropriate columns needed to join across
# all the .csv files we need for our 

def check_column_names(input_df, column_names):
    """Check that all names in provided array are in dataframe"""
    if input_df.columns.isin(column_names).sum() != len(column_names):
        raise ValueError(f'Expected column in {column_names} is missing')
        
def check_column_names_forum_posts(forum_posts_df):
    """Check for columns from ForumMessages table in MetaKaggle"""
    forum_posts_columns = ['ForumTopicId', 'PostDate', 'Message', 'ForumPostId']
    check_column_names(input_df=forum_posts_df,
                      column_names=forum_posts_columns)

def check_column_names_forum_topics(forum_topics_df):
    """Check for columns from ForumTopics table in MetaKaggle"""
    forum_topics_columns = ['Id', 'ForumId', 'TopicTitle']
    check_column_names(input_df=forum_topics_df,
                      column_names=forum_topics_columns)
    
def check_column_names_forum_forums(forum_info_df):
    """Check for columns from Forums table in MetaKaggle"""
    forum_info_columns = ['Id', 'ForumTitle']
    check_column_names(input_df=forum_info_df,
                      column_names=forum_info_columns)

# %% [code]
# functions for joining tables to easily get info out

def join_posts_and_topics(forum_posts_df, forum_topics_df):
    """Join info from the ForumTopics & ForumMessages tables in MetaKaggle"""
    posts_and_topics_df = pd.merge(forum_posts_df[['ForumTopicId', 'PostDate', 'Message', 'ForumPostId']], 
                   forum_topics_df[['Id', 'ForumId', 'TopicTitle']], 
                   left_on='ForumTopicId', right_on='Id')
    posts_and_topics_df = posts_and_topics_df.drop(['Id'], axis=1)
    
    return(posts_and_topics_df)

def join_posts_with_forum_title(posts_and_topics_df,
                                forum_info_df):
    """Join info from join_posts_and_topics() output and MetaKaggle Forums table """
    forum_posts = pd.merge(posts_and_topics_df, 
                           forum_info_df[['Id', 'ForumTitle']], 
                           left_on='ForumId', right_on='Id')
    forum_posts = forum_posts.drop(['Id'], axis=1)
    
    return(forum_posts)
