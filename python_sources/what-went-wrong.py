#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()


# In[ ]:


correct_query = """
                SELECT u.id AS id,
                    MIN(q.creation_date) AS q_creation_date,
                    MIN(a.creation_date) AS a_creation_date
                FROM `bigquery-public-data.stackoverflow.users` AS u
                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                        ON u.id = a.owner_user_id 
                    LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                        ON q.owner_user_id = u.id
                WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
                GROUP BY id
                """

correct_result = client.query(correct_query).to_dataframe()
correct_result.count(axis=0)


# In[ ]:


another_correct_query = """
                        SELECT u.id AS id,
                            MIN(q.creation_date) AS q_creation_date,
                            MIN(a.creation_date) AS a_creation_date
                        FROM `bigquery-public-data.stackoverflow.users` AS u
                            LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                                ON u.id = a.owner_user_id 
                            LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                                ON q.owner_user_id = u.id
                        WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
                        GROUP BY id
                        """

another_correct_result = client.query(another_correct_query).to_dataframe()
another_correct_result.count(axis=0)


# In[ ]:


incorrect_query = """
                  SELECT u.id AS id,
                      MIN(q.creation_date) AS q_creation_date,
                      MIN(a.creation_date) AS a_creation_date
                  FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                      FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                          ON q.owner_user_id = a.owner_user_id
                      RIGHT JOIN `bigquery-public-data.stackoverflow.users` AS u
                          ON q.owner_user_id = u.id
                  WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
                  GROUP BY id
                  """

incorrect_result = client.query(incorrect_query).to_dataframe()
incorrect_result.count(axis=0)


# ### What went wrong?
# 
# In the incorrect query, all users who have not yet asked a question will have a missing entry in the "a_creation_date" column (even if they have asked an answer!).  This accounts for the 5643 extra missing entries in the result returned by the incorrect query:

# In[ ]:


sum(correct_result.q_creation_date.isnull() & correct_result.a_creation_date.notnull())


# ### Why?
# Let's look at a visual example of what went wrong with the query in `incorrect_query`.  Notice that the user with ID 3 is an example of a user who provided an answer, but didn't ask a question.
# 
# ![tut1_plots_you_make](https://i.imgur.com/5Xf5TzG.png)
