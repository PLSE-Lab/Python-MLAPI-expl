#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import in our custom functions
import join_forum_post_info as jfpi
import pylint


# # First, let's make sure all our functions work the way we think they should...

# In[ ]:


# read in the three tables we want from metakaggle
forums_info_df, forum_posts_df, forum_topics_df = jfpi.read_in_forum_tables()


# In[ ]:


# check column names; if all's well you should have no output
jfpi.check_column_names_forum_forums(forums_info_df)
jfpi.check_column_names_forum_posts(forum_posts_df)
jfpi.check_column_names_forum_topics(forum_topics_df)

# join tablesforums_info_df
posts_and_topics_df = jfpi.join_posts_and_topics(forum_posts_df, forum_topics_df)
forum_posts = jfpi.join_posts_with_forum_title(posts_and_topics_df, forums_info_df)


# In[ ]:


# check joined file
forum_posts.head()


# ## Now let's use a linter to check our code against a style guide

# In[ ]:


get_ipython().system('pylint join_forum_post_info')

