#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px


# ## 1. Data Loading

# In[ ]:


df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')


# In[ ]:


df.head()


# ## 2. Data cleaning

# In[ ]:


missing_values_count = df.isnull().sum()

# nan by columns
missing_values_count


# In[ ]:


# removing duplicates
df.drop_duplicates()


# In[ ]:


# replace value of is_paid column
mask = df.applymap(type) != bool
d = {True: 'Paid', False: 'Free'}
df = df.where(mask, df.replace(d))


# In[ ]:


# rename subject column to be categories
df = df.rename(columns={"subject": "categories"})


# ## 3. Course categories

# In[ ]:


# print the total number of unique categories
num_categories = df['categories'].nunique()
print('Number of categories = ', num_categories)


# In[ ]:


# count the number of apps in each 'Category' and sort them for easier plotting
df.groupby('categories')['course_id'].count()

fig = px.bar(df.groupby('categories')['course_id'].count(), x=(df.groupby('categories')['course_id'].count()).values, y=(df.groupby('categories')['course_id'].count()).index, color=(df.groupby('categories')['course_id'].count()).index, orientation='h')
fig.update_layout(
    title="Total courses in each category",
    xaxis_title="Total courses",
    yaxis_title="Category"
)
fig.show()


# ## 4. Number of courses on paid courses vs free courses

# In[ ]:


(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False)


# In[ ]:


fig = px.bar((df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False), x=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).index, y=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).values, text=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).values, color=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).index)
fig.update_layout(
    title="Number of paid courses vs free courses",
    xaxis_title="Type",
    yaxis_title="Total Courses"
)
fig.show()


# ## 5. Number of subscribers on paid courses vs free courses

# In[ ]:


(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False)


# In[ ]:


fig = px.bar((df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False), x=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).index, y=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).values, text=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).values, color=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).index)
fig.update_layout(
    title="Number of paid subscribers vs free subscribers",
    xaxis_title="Type",
    yaxis_title="Total subscribers"
)
fig.show()


# ## 6. Course price

# In[ ]:


fig = px.bar(df.groupby('price')['course_id'].count(), y=(df.groupby('price')['course_id'].count()).values, x=(df.groupby('price')['course_id'].count()).index, text=(df.groupby('price')['course_id'].count()).values)
fig.update_layout(title="Price distribution",
                 xaxis_title="Price",
                 yaxis_title="Frequency"
                 )
fig.show()


# ## 7. Course price, number of reviews, content duration, and year published

# ### 7.1 Does the price of a course affect people to subscribe that course?

# In[ ]:


fig = px.scatter(df, x="price", y="num_subscribers", color="categories")
fig.update_layout(title="Course price and number of subscribers",
                 xaxis_title="Price",
                 yaxis_title="Number of subscribers"
                 )
fig.show()


# 7.2 Does the number of reviews of a course affect people to subscribe that course?

# In[ ]:


fig = px.scatter(df, x="num_reviews", y="num_subscribers", color="categories")
fig.update_layout(title="Course number of reviews and number of subscribers",
                 xaxis_title="Number of reviews",
                 yaxis_title="Number of subscribers"
                 )
fig.show()


# ### 7.3 Does the content duration of a course affect people to subscribe that course?

# In[ ]:


fig = px.scatter(df, x="content_duration", y="num_subscribers", color="categories")
fig.update_layout(title="Course content duration and number of subscribers",
                 xaxis_title="Content duration",
                 yaxis_title="Number of subscribers"
                 )
fig.show()


# ## 7.4 Does the year of course publish affect people to subscribe that course?

# In[ ]:


fig = px.scatter(df, x="published_timestamp", y="num_subscribers", color="categories")
fig.update_layout(title="Year of course and number of subscribers",
                 xaxis_title="Year",
                 yaxis_title="Number of subscribers"
                 )
fig.show()

