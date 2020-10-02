#!/usr/bin/env python
# coding: utf-8

# # State of Javascript in 2018

# In this notebook I will explore Javascript trends for 2018. I will try to see which libraries and forntend frameworks are the most used ones, how are developers using JS etc.
# 
# __NOTICE__: dataset was last updated on September 2nd 2018. I will update notebook once data is updated the rest of 2018.

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import exponnorm
import numpy as np
from bq_helper import BigQueryHelper


# Initializing BigQuery helper:

# In[ ]:


stack_overflow = BigQueryHelper(active_project="bigquery-public-data", dataset_name="stackoverflow")


# List of all tables in Stackoverflow dataset

# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()


# ## Basic stats

# First, I would like to see total number of questions in 2018 and total number of questions related to Javascript

# In[ ]:


total_questions_count_query = '''
    select count(*) as total_number_of_questions 
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018
'''
total_questions_count = stack_overflow.query_to_pandas_safe(total_questions_count_query)
total_questions_count


# In[ ]:


total_js_questions_count_query = '''
    select count(*) as number_of_javascript_questions 
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018 and
        tags like '%javascript%'
'''
total_js_questions_count = stack_overflow.query_to_pandas_safe(total_js_questions_count_query)
total_js_questions_count


# So there was 1546724 questions overall and 165979 questions related to Javascript. So arround 10% of all questions on Stackoverflow in 2018 were about Javascript.

# In[ ]:


questions_per_day_query = '''
    select count(id) as q_count, extract(day from creation_date) as day, extract(month from creation_date) as month
        from `bigquery-public-data.stackoverflow.posts_questions` 
        where extract(year from creation_date) = 2018 and
        tags like '%javascript%'
        group by day, month
'''
questions_per_day = stack_overflow.query_to_pandas_safe(questions_per_day_query)
questions_per_day.head()


# ### Working days vs weekends

# In[ ]:


questions_per_day.q_count = questions_per_day.q_count.values.astype(int)
pivoted_table = questions_per_day.pivot('day', 'month', 'q_count')

plt.figure(figsize=(16,12))
sns.heatmap(data=pivoted_table, annot=True, fmt='.0f', linewidths=.5)


# We can observe interesting pattern from the heatmap above. Developers create less questions on weekends then on working days.

# ## Answers and comments to Javacript questions

# In[ ]:


answers_to_js_count_query = '''
    select id, accepted_answer_id, answer_count, comment_count
    from `bigquery-public-data.stackoverflow.posts_questions`
    where extract(year from creation_date) = 2018 and
    tags like '%javascript%'
'''
answers_to_js_count = stack_overflow.query_to_pandas_safe(answers_to_js_count_query)
answers_to_js_count.head()


# In[ ]:


votes_and_views_js_count_query = '''
    select id, favorite_count, view_count
    from `bigquery-public-data.stackoverflow.posts_questions`
    where extract(year from creation_date) = 2018 and
    tags like '%javascript%'
'''
votes_and_views_js_count = stack_overflow.query_to_pandas_safe(votes_and_views_js_count_query)
votes_and_views_js_count = votes_and_views_js_count.fillna(0)
votes_and_views_js_count.head()


# In[ ]:


answers_to_js_count['has_accepted_answer'] = answers_to_js_count.apply(lambda r: ~np.isnan(r.accepted_answer_id), axis=1)
answers_to_js_count.groupby('has_accepted_answer').count()


# There is 97976 javacript questions with accepted answer and 68003 without one.

# ### Distribution of number of answers

# Distribution of number of answers

# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(answers_to_js_count.
             answer_count.values,
             fit=exponnorm,
             bins=20,
             axlabel='Number of answers',
             kde=False,
             rug=True)


# Distribution of number of comments

# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(answers_to_js_count.comment_count,
             bins=40,
             fit=exponnorm,
             axlabel='Number of comments',
             kde=False,
             rug=True)


# 
# ## Tags

# In this section we want to see which tags are most used with `javascript` tag

# In[ ]:


tag_js_query = '''
    select id, tags
        from `bigquery-public-data.stackoverflow.posts_questions`
            where extract(year from creation_date) = 2018 and
            tags like '%javascript%'
'''
tags_raw = stack_overflow.query_to_pandas_safe(tag_js_query)
tags_raw.head()

rows_list = []
for _, rows in tags_raw.iterrows():
    tag = rows.tags.split('|')
    for t in tag:
        if t != 'javascript':
            row = {'question_id': rows.id, 'tag': t}
            rows_list.append(row)
tags_per_question = pd.DataFrame(rows_list)
tags_per_question.head()


# In table bellow we can see which tags/technologies were most used with `javascript` in 2018.

# In[ ]:


tag_count = tags_per_question.groupby('tag').count().sort_values(by='question_id', ascending=False)
tag_count.head(20)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x=tag_count.index[0:20], y=tag_count.question_id[0:20])


#  Most used tags are `jquery` and `html`. In top 5 there is also`node.js`(javascript runtime based on V8 engine) and `reactjs`(frontend library created by Facebook).

# ### Question views and answers for top `javascript` tags.

# In[ ]:


top20tags = tag_count.head(20).index.values
answers_tags = pd.merge(answers_to_js_count, tags_per_question, left_on='id', right_on='question_id', how='left')
views_tags = pd.merge(votes_and_views_js_count, tags_per_question, left_on='id', right_on='question_id', how='left')


# In the table bellow we will calculate following numbers for questions with 20 most used tags:
# * average answer count
# * average comment count
# * percentage of questions with highest rate of accepted answers

# In[ ]:


answers_tag_grouped = answers_tags[answers_tags.tag.isin(top20tags)][['tag', 'answer_count', 'comment_count', 'has_accepted_answer']].groupby('tag')
avg_answer_count = answers_tag_grouped.mean().sort_values('has_accepted_answer', ascending=False)
avg_answer_count


# In[ ]:


answers_top20_tags = answers_tags[answers_tags.tag.isin(top20tags)]
views_top20_tags = views_tags[views_tags.tag.isin(top20tags)]


# In[ ]:


plt.figure(figsize=(20, 8))
sns.stripplot(data=answers_top20_tags, x='tag', y='answer_count', jitter=True)


# ### Relation between answers and comments

# In[ ]:


tag_scatter_data = answers_tags[['tag', 'answer_count', 'comment_count']].groupby('tag').sum()
tag_scatter_data = tag_count.join(tag_scatter_data, how='left')
tag_scatter_data['tag'] = tag_scatter_data.index.values

sns.jointplot('answer_count', 'comment_count', data=tag_scatter_data, height=10, kind='reg')


# ### View counts and favorite count for tags 

# In following table we want to see which tags have the biggest number of views.

# In[ ]:


avg_views = views_top20_tags[['tag', 'favorite_count', 'view_count']].groupby('tag').mean()
avg_views = avg_views.sort_values('view_count', ascending=False)

f = plt.figure(figsize=(16, 12))

ax = f.add_subplot(2, 1, 1)
plt.xticks(rotation=90)
sns.barplot(x = avg_views.index, y=avg_views.view_count, ax=ax)
plt.subplots_adjust(hspace=1)

avg_views = avg_views.sort_values('favorite_count', ascending=False)
ax = f.add_subplot(2, 1, 2)
plt.xticks(rotation=90)
sns.barplot(x = avg_views.index, y=avg_views.favorite_count, ax=ax)


# As we can see from the bar chart aboe and table, questions with tags `ecmascript-6`, `webpack`  have biggest number of views and are followed by frontend frameworks `angular` and `vue.js`.

# #### Ratio between number of views and answers for some tag

# In[ ]:


merged_avg_answer_view = pd.merge(avg_views, avg_answer_count, left_index=True, right_index=True)
merged_avg_answer_view = merged_avg_answer_view[['view_count', 'answer_count']].copy()
merged_avg_answer_view['tag'] = merged_avg_answer_view.index.values
merged_avg_answer_view['ratio'] = merged_avg_answer_view.view_count/merged_avg_answer_view.answer_count
merged_avg_answer_view.sort_values('ratio', ascending=False)


# On chart bellow we can see distribution of top 20 tags by average view counts and average answers. 

# In[ ]:


sns.jointplot('answer_count', 'view_count', merged_avg_answer_view, height=6, kind='kde')


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(x='answer_count', y='view_count', hue='tag', data=merged_avg_answer_view, palette='hls', s=100)


# We can see that `webpack` questions have bad ratio of views against answers(a lot of views and relatively small number answers). Tags `regex` and `arrays` have smaller number of view counts but (relatively) larger number of answers.

# ## Usage by geolocation

# In[ ]:


user_location_query = '''
    select u.location, q.tags
    from `bigquery-public-data.stackoverflow.posts_questions` q
    left join `bigquery-public-data.stackoverflow.users` u on q.owner_user_id = u.id
        where extract(year from q.creation_date) = 2018 and
        q.tags like '%javascript%'
'''

geo_locations = stack_overflow.query_to_pandas_safe(user_location_query)
geo_locations.head()


# In[ ]:


geo_tag_list = []
for _, rows in geo_locations.iterrows():
    tag = rows.tags.split('|')
    for t in tag:
        if t != 'javascript':
            row = {'location': rows.location, 'tag': t}
            geo_tag_list.append(row)

geo_tag_data = pd.DataFrame(geo_tag_list)


# In[ ]:


geo_tag_data.head()


# ### Frontend frameworks

# In[ ]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# #### Angular, React, Vue

# In 3  tables bellow we have top locations of users from which questions were created. Tables are shown for `angular`, `reactjs` and `vue.js` respectively.

# In[ ]:


angular_loc = geo_tag_data[geo_tag_data.tag == 'angular'].groupby('location').count().sort_values('tag', ascending=False).head(10)
react_loc = geo_tag_data[geo_tag_data.tag == 'reactjs'].groupby('location').count().sort_values('tag', ascending=False).head(10)
vue_loc = geo_tag_data[geo_tag_data.tag == 'vue.js'].groupby('location').count().sort_values('tag', ascending=False).head(10)

display_side_by_side(angular_loc, react_loc, vue_loc)


# ### Node.js and PHP

# In[ ]:


node_loc = geo_tag_data[geo_tag_data.tag == 'node.js'].groupby('location').count().sort_values('tag', ascending=False).head(10)
php_loc  = geo_tag_data[geo_tag_data.tag == 'php'].groupby('location').count().sort_values('tag', ascending=False).head(10)
display_side_by_side(node_loc, php_loc)


# In[ ]:




