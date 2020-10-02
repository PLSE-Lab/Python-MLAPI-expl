#!/usr/bin/env python
# coding: utf-8

# **[Advanced SQL Home Page](https://www.kaggle.com/learn/advanced-sql)**
# 
# ---
# 

# # Exercises
# ### 3) Initial questions and answers, Part 2
# 
# Now you'll address a more realistic (and complex!) scenario.  To answer this question, you'll need to pull information from *three* different tables!  This syntax very similar to the case when we have to join only two tables.  For instance, consider the three tables below.
# 
# ![three tables](https://i.imgur.com/OyhYtD1.png)
# 
# We can use two different **JOINs** to link together information from all three tables, in a single query.
# 
# ![double join](https://i.imgur.com/G6buS7P.png)
# 
# With this in mind, say you're interested in understanding users who joined the site in January 2019.  You want to track their activity on the site: when did they post their first questions and answers, if ever?
# 
# Write a query that returns the following columns:
# - `id` - the IDs of all users who created Stack Overflow accounts in January 2019 (January 1, 2019, to January 31, 2019, inclusive)
# - `q_creation_date` - the first time the user posted a question on the site; if the user has never posted a question, the value should be null
# - `a_creation_date` - the first time the user posted a question on the site; if the user has never posted a question, the value should be null
# 
# Note that questions and answers posted after January 31, 2019, should still be included in the results.  And, all users who joined the site in January 2019 should be included (even if they have never posted a question or provided an answer).
# 
# The query from the previous question should be a nice starting point to answering this question!  You'll need to use the `posts_answers` and `posts_questions` tables.  You'll also need to use the `users` table from the Stack Overflow dataset.  The relevant columns from the `users` table are `id` (the ID of each user) and `creation_date` (when the user joined the Stack Overflow site, in DATETIME format).

# ## 1. Connect to BiqQuery

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()


# ## 2. Let's choose some users and check how their activity looks like in all 3 tables
# 
# User **11486952** - there are answers but no questions<br/>
# User **10904707** - there are questions but no answers <br/>
# User **10977933** - there are both answers and questions <br/>
# User **11040013** - there are no answers and no questions <br/>
# User **10600000** - join the site before Jan 1

# In[ ]:


query = """
         SELECT 'table: USERS' as table,
                 id AS user_id,
                 MIN(creation_date) as min_date,
                 'date of joinning Stack Overflow' as date_description
             FROM `bigquery-public-data.stackoverflow.users`
             WHERE id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             GROUP BY table, user_id
         UNION ALL
         SELECT 'table: POSTS_QUESTIONS' as table,
                 owner_user_id AS user_id,
                 MIN(creation_date) as min_date,
                 'date of first question' as date_description
             FROM `bigquery-public-data.stackoverflow.posts_questions`
             WHERE owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             GROUP BY table, user_id
         UNION ALL
         SELECT 'table: POSTS_ANSWERS' as table,
                 owner_user_id AS user_id,
                 MIN(creation_date) as min_date,
                 'date of first answer' as date_description
             FROM `bigquery-public-data.stackoverflow.posts_answers`                     
             WHERE owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             GROUP BY table, user_id
                    """

user = client.query(query).result().to_dataframe()
user.sort_values('user_id')


# ## 3. Now, let's see the result for them if we use sugested solution from Kaggle

# In[ ]:


three_tables_query = """
             SELECT u.id AS id,
                 MIN(q.creation_date) AS q_creation_date,
                 MIN(a.creation_date) AS a_creation_date
             FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                 FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                     ON q.owner_user_id = a.owner_user_id 
                 RIGHT JOIN `bigquery-public-data.stackoverflow.users` AS u
                     ON q.owner_user_id = u.id
             WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
             AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             GROUP BY id
                    """

client.query(three_tables_query).result().to_dataframe()


# We've got 4 users - it's OK, cause user 10600000 join Stack Overflow earlier, so he's out.<br/><br/>
# 
# **But what about user 11486952? There should be a date in *a_creation_date*.**

# ## 4. What went wrong?
# 
# Let's look on the FULL JOIN result

# In[ ]:


three_tables_query = """
         SELECT q.owner_user_id AS q_id,
             a.owner_user_id as a_id,
             MIN(q.creation_date) AS q_creation_date,
             MIN(a.creation_date) AS a_creation_date
         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
             FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                 ON q.owner_user_id = a.owner_user_id 
         WHERE q.owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             or a.owner_user_id IN (11486952, 10904707, 10977933, 11040013, 10600000)
         GROUP BY q_id,a_id
                    """

client.query(three_tables_query).result().to_dataframe()


# **Look at last row.**<br/>
# Think what is happening when we try to RIGHT JOIN USERS using condition:<br/>
#     ON q.owner_user_id = u.id<br/>
# That's right we use q_id column. Instead we should write:<br/>
#     ON **COALESCE**(q.owner_user_id, a.owner_user_id) = u.id<br/>
# But we don't know this function yet ;) Let's try that solution.

# In[ ]:


three_tables_query = """
             SELECT u.id AS id,
                 MIN(q.creation_date) AS q_creation_date,
                 MIN(a.creation_date) AS a_creation_date
             FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                 FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                     ON q.owner_user_id = a.owner_user_id 
                 RIGHT JOIN `bigquery-public-data.stackoverflow.users` AS u
                     ON COALESCE(q.owner_user_id, a.owner_user_id) = u.id
             WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
             AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)
             GROUP BY id
                    """

client.query(three_tables_query).result().to_dataframe()


# Looks good!

# ## 5. Finally, our LEFT JOIN solution

# In[ ]:


# Your code here
three_tables_query = """
            SELECT u.id,
                MIN(q.creation_date) AS q_creation_date,
                MIN(a.creation_date) AS a_creation_date
            FROM `bigquery-public-data.stackoverflow.users` AS u
            LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                ON u.id = q.owner_user_id
            LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON u.id = a.owner_user_id 
            WHERE u.creation_date >= '2019-01-01' AND u.creation_date < '2019-02-01' 
            AND u.id IN (11486952, 10904707, 10977933, 11040013, 10600000)
            GROUP BY u.id
                     """


client.query(three_tables_query).result().to_dataframe()

