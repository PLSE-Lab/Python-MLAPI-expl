#!/usr/bin/env python
# coding: utf-8

# So, which is more popular among python enthusiasts: **tabs** or **spaces**?
# 
# Inspired by a [hilarious scene](https://www.youtube.com/watch?v=V7PLxL8jIl8) from Silicon Valley, the question is there and it is expecting a sound answer.
# ![tabs_preferences](https://img.devrant.com/devrant/rant/r_109448_5NyDp.jpg)
# 
# Let's find out.

# In[ ]:


# import our bq_helper package
import bq_helper


# Explore github datasets: show available tables

# In[ ]:


# create a helper object for our bigquery dataset
github = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
# print a list of all the tables in the hacker_news dataset
github.list_tables()


# We'll use `sample_contents` table which is a 10% random sample from all github data, and should be perfectly representative to get an answer.

# In[ ]:


github.head('sample_contents', num_rows=3)


# Building a query. Inspired by [Hayato's kernel[](http://)](https://www.kaggle.com/hayatoy/most-common-indentation-space-count-in-python-code/notebook).

# In[ ]:


q_tab_or_space = ('''
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.py" )
SELECT
  Indentation,
  COUNT(Indentation) AS number_of_occurence
FROM (
  SELECT
    CASE
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^\t+")))>=1 THEN 'Tab'
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +")))>=1 THEN 'Space'
        ELSE 'Other'
    END AS Indentation
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^\s+")
  GROUP BY
    id )
GROUP BY
  Indentation
ORDER BY
  number_of_occurence DESC
''')


# A good practise is to estimate the query beforehand when working with Bigdata: _how much data would the query scan before getting the results?

# In[ ]:


github.estimate_query_size(q_tab_or_space)


# Our query would scan through roughly 24 Gigs. Which is fine.

# Onto running the query:

# In[ ]:


tab_or_space_df = github.query_to_pandas(q_tab_or_space)
tab_or_space_df


# Completing with a nice visual.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
# setting default seaborn aesthetic style
sns.set()
# define plot and figure size
fig = plt.figure(figsize=(10,6))
# Add a subplot
ax = fig.add_subplot(111)
bars = tab_or_space_df.plot(kind='bar', x='Indentation', y='number_of_occurence', ax=ax)
ax.set_ylabel('Occurence',fontsize=12,alpha=0.75)
ax.set_xlabel('Indentation',fontsize=12,alpha=0.75)
ax.set_title('Tabs vs Spaces\n(github python files)')
ax.set_ylim(0, tab_or_space_df['number_of_occurence'].max()*1.2)
plt.xticks(rotation=0);

for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, (height+1500), '{}'.format(height), 
                 ha='center', color='black', fontsize=12, alpha=0.75)

plt.show()


# Oh wow. That is a surprise to me, as someone who fav Tabs! I find it hard to believe that people use space as an indentation much more often than tabs!
# 
# Please let me know, what do you prefer?
