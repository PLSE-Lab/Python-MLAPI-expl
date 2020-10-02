#!/usr/bin/env python
# coding: utf-8

# # HackerNews Dataset Analysis
# This dataset contains all stories and comments from [HackerNews](https://news.ycombinator.com/news)  from its launch in 2006. Each story contains a story id, the author that made the post, when it was written, and the number of points the story received. <br>
# Hacker News is a social news website focusing on computer science and entrepreneurship. It is run by Paul Graham's investment fund and startup incubator, Y Combinator. In general, content that can be submitted is defined as "anything that gratifies one's intellectual curiosity". <br>
# 
# There are 4 tables in this public dataset:
# - stories (a.k.a posts)
# - comments
# - full
# - full_2015 <br>
# <br>
# * An example [post](https://news.ycombinator.com/item?id=8596682) from HackerNews.
# * An example of a [comment](https://news.ycombinator.com/item?id=8597333) from HackerNews.
# * A post can have many comments.
# * A comment can have many comments.
# * Posts can have scores.
# * Posts have rankings showing their popularity.
# 
# 
# So the structure is:
# * Story
#   * comment
#     * comment
#       * comment
# 
# **Now the question is, how this structure represented in the tables ?**

# # 1) Understanding the structure between *stories* and *comments* tables
# Before diving deeper in the queries lets understand the structure of the tables by investigating following questions:
# * Parent_id column in the comments table: 
#   * what do they represent? 
#   * do they only represent parent id of the comment or the parent id of the story?
# * How much of a comments does a story has, which column represents it?
# * Are the comment and story ids globally unique?

# In[ ]:


import os
import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Stories Table - First Rows

# In[ ]:


# view first rows of stories table
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# Construct a reference to the "stories" table
table_ref_stories = dataset_ref.table("stories")

# API request - fetch the table
table_stories = client.get_table(table_ref_stories)

# Preview the first five lines of the "stories" table
client.list_rows(table_stories, max_results=5).to_dataframe()


# **Stories**
# * `id` is the unique key of the story.
# * `descendants` is the number of all comments that a story has.
# * Points of a story is represented as score in the table.

# ## Stories Table - Attributes and Field Data Types

# In[ ]:


table_stories.schema


# ## Comments Table - First Rows

# In[ ]:


# Construct a reference to the "comments" table
table_ref_comments = dataset_ref.table("comments")

# API request - fetch the table
table_comments = client.get_table(table_ref_comments)

# Preview the first five lines of the "comments" table
client.list_rows(table_comments, max_results=5).to_dataframe()


# ## Comments Table - Attributes and Field Data Types

# In[ ]:


table_comments.schema


# ## Parent column in the `comments` table, what does it represent?

# In[ ]:


# look at the parent_id column in the comments table
# then search for parent_ids who does not represent a comment_id in the comments table
# if this results are no empty, it means parent column represent parend comment id and one another id
query_for_id_selection = """ 
                            SELECT DISTINCT(parent)
                            FROM `bigquery-public-data.hacker_news.comments`
                            WHERE parent NOT IN (
                                                    SELECT id
                                                    FROM `bigquery-public-data.hacker_news.comments`)
            
"""
# set quota not to exceed limits
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

# create job to execute query
query_for_id_selection_job = client.query(query_for_id_selection, job_config=safe_config)

#load results into a dataframe
query_for_id_result = query_for_id_selection_job.to_dataframe()

# turn dataframe to a a series, then list, then list of strings to use in the query
non_comment_ids = ",".join(map(str,query_for_id_result.head(20).parent.tolist()))


# In[ ]:


# function to write query results to a dataframe
# without exceeding the 1 GB quota per query
def query_to_dataframe(query_name):
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
    query_name_job = client.query(query_name, job_config=safe_config)
    return query_name_job.to_dataframe()


# In[ ]:


print("Number of parent_ids that are not comment_ids:", query_for_id_result.parent.nunique())


# **Conclusion 1:** There are 485055 parent_ids that does not represent to a comment_id. Lets have a look at some of them.

# In[ ]:


non_comment_ids


# In[ ]:


# are parent_ids that does not represent a comment_id, do they belong to a story?
non_comment_id_query = """ 
                            SELECT *
                            FROM `bigquery-public-data.hacker_news.stories` AS stories
                            WHERE id IN ({})
                    """.format(non_comment_ids)

# print query results
query_to_dataframe(non_comment_id_query).head()


# **Conclusion 2:** Some of the non_comment_ids are story_ids.

# ## How much of comments does a story has, which column represents it?

# In[ ]:


# look at parent id row 4318042 to see how many comments are dependant on it
# in the stories table, number of  descendants are 113
query_number_of_comments = """
                            SELECT *
                            FROM `bigquery-public-data.hacker_news.comments`
                            WHERE parent = 4318042
"""

print("Number of comments that story 4318042 has:", len(query_to_dataframe(query_number_of_comments)))
# it did not give 122 rows because of the tree structure.
# it only give the first level comment id which is 12.


# * When number of descendants were 113, number of comments belong to that story is 53.
# * This is an expected result because of the tree structure that post and comments have.
# * 53 is the number of first level comments created for that story, 113 is the number of all sub-comments that a story has. <br>
# 
# **Conclusion 3:** Descendants attribute are the total number of comments including sub-comments that a story has.

# ## Are id columns in the stories and comments globally unique ?

# In[ ]:


id_query = """ 
                SELECT id
                FROM `bigquery-public-data.hacker_news.stories`
                WHERE id IN (
                            SELECT id
                            FROM `bigquery-public-data.hacker_news.comments`
                            )
"""

query_to_dataframe(id_query)


# **Conclusion 4:** Ids are globally unique among stories and comments table.

# # 2) With the conclusions in mind, some interesting questions to ask:
# * Recent studies have found that many forums tend to be dominated by a very small fraction of users. Is this true of Hacker News?
# * Hacker News has received complaints that the site is biased towards Y Combinator startups. Do the data support this?
# * What is the average number of daily comments created per day?
# * What is the number of users that HackerNews had over years?
# * How long does it take for a post to receive comment?
# * How many of the comments receive sub-comments ?
# * Is it more common for users to first create post or provide comments?
# * For the users who joined the site in January 2014. When did they post their first story or comment, if ever?
# * How many distinct users posted on October, 2015?
# * What is the moving average (within the 15 day window) of number of posts created in each post category? 
# * What is the rank of the stories based on scores, created in the same day ?
# 

# ## a) Recent studies have found that many forums tend to be dominated by a very small fraction of users. Is this true of Hacker News?
# * number of posts created
# * number of comments created
# 
# shows a sign of being a active and dominated user in the HackerNews.
# 

# In[ ]:


# to have a look we are going to count the number of comments and posts created 
# for not deleted and dead ones
active_users_query = """
                        WITH active_users_from_stories AS (
                            SELECT author,
                                COUNT(*) AS number_of_stories
                            FROM `bigquery-public-data.hacker_news.stories`
                            WHERE deleted IS NOT TRUE AND dead IS NOT TRUE
                            GROUP BY author
                            ORDER BY number_of_stories DESC
                        ),
                        active_users_from_comments AS (
                            SELECT author,
                                COUNT(*) AS number_of_comments
                            FROM `bigquery-public-data.hacker_news.comments`
                            WHERE deleted IS NOT TRUE AND dead IS NOT TRUE
                            GROUP BY author
                            ORDER BY number_of_comments DESC
                        )
                        SELECT active_users_from_comments.author,
                            number_of_stories,
                            number_of_comments
                        FROM active_users_from_stories
                        FULL JOIN active_users_from_comments
                         ON active_users_from_stories.author = active_users_from_comments.author
                        ORDER BY number_of_stories DESC
                        LIMIT 10
"""
# if you change last ORDER BY clause to number_of_comments will list the users who commented most
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
active_users_query_job = client.query(active_users_query, job_config=safe_config)
active_users_query_result = active_users_query_job.to_dataframe()
active_users_query_result


# In[ ]:


# look at the total number of stories in the whole hackernews which is not dead or deleted
total_stories = """
                    SELECT COUNT(id)
                    FROM `bigquery-public-data.hacker_news.stories`
                    WHERE deleted IS NOT TRUE AND dead IS NOT TRUE
"""
total_stories_df = client.query(total_stories).result().to_dataframe()

percentage = active_users_query_result.number_of_stories.sum()/total_stories_df.f0_.iloc[0]

print("Most active users in terms of number of stories created have created the {} of the whole stories"
      .format(round(percentage,2)))


# ## b) Hacker News has received complaints that the site is biased towards Y Combinator startups. Do the data support this?
# 
# Here are some of the most popular YCombinator startups:
# 
# * AirBnB
# * Stripe
# * Dropbox
# * Zapier
# * Reddit
# and so on..

# In[ ]:


# before providing an answer to this question investigate and understand full table

# Construct a reference to the "full" table
table_ref_full = dataset_ref.table("full")

# API request - fetch the table
table_full = client.get_table(table_ref_full)

# Preview the first five lines of the "full" table
client.list_rows(table_full, max_results=5).to_dataframe()


# In[ ]:


# if those keywords occur in the full table's text or title column we can say they got covered
coverage_query = """WITH startup_ranking_score AS (
                    SELECT CASE 
                        WHEN title LIKE "%Airbnb%" OR text LIKE "%Airbnb%" THEN "Airbnb"
                        WHEN title LIKE "%Stripe%" OR text LIKE "%Stripe%" THEN "Stripe"
                        WHEN title LIKE "%Dropbox%" OR text LIKE "%Dropbox%" THEN "Dropbox"
                        WHEN title LIKE "%Zapier%" OR text LIKE "%Zapier%" THEN "Zapier"
                        WHEN title LIKE "%Reddit%" OR text LIKE "%Reddit%" THEN "Reddit"
                        END AS popular_startup_name,
                    ranking,
                    score
                    FROM `bigquery-public-data.hacker_news.full`
                                                                )
                    SELECT popular_startup_name,
                        SUM(ranking) AS total_ranking,
                        SUM(score) AS total_score
                    FROM startup_ranking_score
                    GROUP BY popular_startup_name
                    ORDER BY total_score DESC
"""
query_to_dataframe(coverage_query)


# YCombinator startups got covered by HackerNews. However, most of the stories belong to other subjects showing there is no bias towards those most popular startups.

# ## c) What is the average number of daily comments created per day?

# In[ ]:


# to look at this first total number of comments generated per day will be investigated
# then the average of the year will be aggregated
average_daily_comments_per_year = """ WITH total_comments_generated_per_day AS (
                                        SELECT EXTRACT(DAYOFYEAR FROM time_ts) AS day,
                                            EXTRACT(YEAR FROM time_ts) AS year,
                                            COUNT(id) AS total_comments
                                        FROM `bigquery-public-data.hacker_news.comments`
                                        GROUP BY year, day
                                        )
                                        SELECT year, 
                                            AVG(total_comments) AS average_daily_comments
                                        FROM total_comments_generated_per_day
                                        GROUP BY year
                                        ORDER BY year
"""
query_to_dataframe(average_daily_comments_per_year)


# Looks like popularity of HackerNews increased constantly and dramatically from 2006 to 2015.

# ## d) What is the number of users that HackerNews had over years?

# In[ ]:


# this question will be investigated in the full table 
number_of_users = """
                    SELECT EXTRACT(YEAR FROM timestamp) AS year,
                        COUNT(DISTINCT f.by) AS number_of_users
                    FROM `bigquery-public-data.hacker_news.full` AS f
                    WHERE timestamp IS NOT NULL
                    GROUP BY year
                    ORDER BY year     
"""
query_to_dataframe(number_of_users)


# Number of users constantly growed, supports the idea of the increasing popularity of HackerNews.

# ## e) How long does it take for a post to receive comment?

# In[ ]:


# to answer this question stories and comments tables will be joined and 
# and timedifference of the time_ts will be investigated
time_to_receive_comment = """
                            WITH time_difference AS (
                                SELECT stories.id AS story_id,
                                    MIN(TIMESTAMP_DIFF(comments.time_ts, stories.time_ts, SECOND)) AS second
                                FROM `bigquery-public-data.hacker_news.stories` AS stories
                                LEFT JOIN `bigquery-public-data.hacker_news.comments` AS comments
                                    ON stories.id = comments.parent
                                GROUP BY story_id
                                ORDER BY second ASC)
                            SELECT *
                            FROM time_difference
                            WHERE second >= 0
"""
query_to_dataframe(time_to_receive_comment).head(10)


# It takes only seconds a story to receive a comment.

# In[ ]:


print("Average number of hours passed for a story to receive a comment:",
      round(query_to_dataframe(time_to_receive_comment).second.mean()/3600,2))


# ## f) How many of the comments receive sub-comments ?

# In[ ]:


# to answer this question we are going to look at the ratio of 
# comments receiving subcomments to all comments

# total number of comments in the comments table
total_num_comments = """
                        SELECT COUNT(DISTINCT id)
                        FROM `bigquery-public-data.hacker_news.comments`
                    """

# total number of comments having sub_comments in the comments table
total_num_comments_w_sub_comments = """ WITH comments_w_subcomment_list AS (
                                            SELECT id,
                                                CASE
                                                    WHEN id IN (
                                                        SELECT DISTINCT(parent)
                                                        FROM `bigquery-public-data.hacker_news.comments`) 
                                                            THEN 1
                                                    ELSE 0
                                                END AS is_commented
                                            FROM `bigquery-public-data.hacker_news.comments`)
                                        SELECT SUM(is_commented)
                                        FROM comments_w_subcomment_list
"""


# In[ ]:


percent_of_commented_comments = 100 * (query_to_dataframe(total_num_comments_w_sub_comments).f0_.iloc[0] 
                                      / query_to_dataframe(total_num_comments).f0_.iloc[0])

print("Percentage of comments with replies {}".format(round(percent_of_commented_comments,2)))


# ## g) Is it more common for users to first create post or provide comments ?

# In[ ]:


# we are going to investigate this question in the union of comments and stories table
# with necessary attributes
user_and_creation_date = """ WITH authors_creation_times AS (
                                SELECT author, 
                                    time_ts,
                                        CASE
                                            WHEN author IS NOT NULL THEN "story"
                                                ELSE NULL
                                        END AS type
                                FROM `bigquery-public-data.hacker_news.stories` AS stories
                                UNION ALL
                                SELECT author,
                                    time_ts,
                                        CASE
                                            WHEN author IS NOT NULL THEN "comment"
                                                ELSE NULL
                                    END AS type
                                FROM `bigquery-public-data.hacker_news.comments` AS comments),
                            first_creation_date AS (
                                SELECT author,
                                    type,
                                    MIN(time_ts) AS first_activity_time
                                FROM authors_creation_times
                                GROUP BY author, type)
                            SELECT type,
                                COUNT(author) AS number_of_users
                            FROM first_creation_date
                            GROUP BY type        
"""
query_to_dataframe(user_and_creation_date)


# People tend to comment first in the HackerNews. However, there is no significant difference between number of users first created a comment or story.

# ## h) For the users who joined the site in January 2014, When did they post their first story or comment, if ever?

# In[ ]:


# to answer this question
# first users who make their first activity on HackerNews on January 2014 will be identified
# then their activity will be matched from full table
# to make the query more efficient CTEs will be used rather than joining multiple tables at once

users_w_first_activity_2014_01 = """WITH users_from_2014_01 AS (
                                        SELECT f.by AS author,
                                            MIN(timestamp) AS first_activity
                                        FROM `bigquery-public-data.hacker_news.full` AS f
                                        WHERE timestamp >= '2014-01-01' AND timestamp < '2014-02-01'
                                        GROUP BY f.by)
                                    SELECT users_from_2014_01.author,
                                        users_from_2014_01.first_activity,
                                        f.type
                                    FROM users_from_2014_01
                                    LEFT JOIN `bigquery-public-data.hacker_news.full` AS f
                                    ON users_from_2014_01.author = f.by 
                                        AND users_from_2014_01.first_activity = f.timestamp   
                                """
query_to_dataframe(users_w_first_activity_2014_01).head(10)


# First activities of the users who discovered HackerNews on January 2914, is to create a job post or pollopt.

# ## i) How many distinct users posted on October, 2015?

# In[ ]:


# before answering this question lets look at the first rows of full_201510 table
# Construct a reference to the "full" table
table_ref_full = dataset_ref.table("full_201510")

# API request - fetch the table
table_full = client.get_table(table_ref_full)

# Preview the first five lines of the "full" table
client.list_rows(table_full, max_results=5).to_dataframe()


# In[ ]:


# to answer this we are going to use full_201510 table
users_posted_201510 = """ 
                        SELECT COUNT(DISTINCT f.by) AS number_of_users
                        FROM `bigquery-public-data.hacker_news.full_201510` AS f
                     """
query_to_dataframe(users_posted_201510)


# ## j) What is the moving average (within the 15 day window) of number of posts created from 2018 and onwards, in each post category? 

# In[ ]:


# number of posts created temporary table will be created using full table
# and then moving averages per date and post category will be calculated
# using analytic functions
moving_average_query = """ WITH num_posts_per_day_type AS ( 
                            SELECT EXTRACT(DATE FROM timestamp) AS date,
                                type,
                                COUNT(id) AS num_posts
                            FROM `bigquery-public-data.hacker_news.full` 
                            WHERE timestamp >= "2018-01-01"
                            GROUP BY date, type
                            )
                          SELECT date,
                            type,
                            AVG(num_posts) OVER (
                                PARTITION BY type
                                ORDER BY num_posts
                                ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING) AS moving_average
                          FROM num_posts_per_day_type
                          ORDER BY date
"""
query_to_dataframe(moving_average_query).head(10)


# Number of posts created in story and comment category have a significantly higher moving averages than job, poll and pollopt category.

# ## k) What is the rank of the stories based on scores, created in the same day ?

# In[ ]:


# scores will be grouped and order per day created from scores table
# and a rank will be assigned using analytic functions
scores_query = """
                    SELECT id,
                        score,
                        EXTRACT(DATE FROM time_ts) AS date,
                        RANK() OVER(
                            PARTITION BY EXTRACT(DATE FROM time_ts)
                            ORDER BY score) AS score_rank
                    FROM `bigquery-public-data.hacker_news.stories`
                    WHERE score >= 0   
"""
query_to_dataframe(scores_query).head(10)

