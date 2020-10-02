#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install joypy -q')


# In[ ]:


import joypy
import pandas as pd
import numpy  as np
import plotly.express  as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib import cm
plt.style.use('ggplot')


# # Read Data

# In[ ]:


# Read data
courses_df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')

# Print sample
courses_df.sample(5).reset_index(drop=True).style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})


# In[ ]:


# Split content_time column into unit and value
courses_df['content_time_value'] = courses_df['content_duration'].str.split(' ').str[0]
courses_df['content_time_unit']  = courses_df['content_duration'].str.split(' ').str[1]

# Inspect contentless cases
courses_df.query("content_time_unit == 'questions'")


# Surprisingly, some courses have no lectures at all!
# 
# Its just a few of them, and they are essentialy paid assessments.
# We'll remove them as the duration does not apply for them.

# In[ ]:


# Remove undesired rows
courses_df = courses_df.drop([93,95,847,970,2066],axis=0).reset_index(drop=True)

# Fix content duration column
courses_df['content_multiplier'] = np.where(courses_df['content_time_unit'] == 'mins',1/60,1)
courses_df['content_duration']   = courses_df['content_time_value'].astype('float') * courses_df['content_multiplier']


# In[ ]:


def fix_paid_columns(x):
    if x == 'TRUE':
        return('True')
    elif x == 'FALSE':
        return('False')
    else:
        return(x)


# In[ ]:


courses_df['is_paid']    = courses_df['is_paid'].apply(fix_paid_columns)
courses_df['engagement'] = courses_df['num_reviews'] / courses_df['num_subscribers']


# In[ ]:


courses_df.to_csv('clean_dataset.csv',index=False)


# # General Exploration

# Let's start by seeing how the courses are distributed along the dataset.

# ## % of Topics in the Dataset

# In[ ]:


temp_df = pd.DataFrame(courses_df['subject'].value_counts()).reset_index()

fig = go.Figure(data=[go.Pie(labels=temp_df['index'],
                             values=temp_df['subject'],
                             hole=.7,
                             title = '% of Courses by Subject',
                             marker_colors = px.colors.sequential.Blues_r,
                            )
                     
                     ])
fig.update_layout(title='Amount of Courses by Subject')
fig.show()


# ## Course Duration Distribution

# In[ ]:


fig = px.box(courses_df,
       x='content_duration',
       y='is_paid',
       orientation='h',
       color='is_paid',
       title='Duration Distribution Across Type of Course',
       color_discrete_sequence=['#03cffc','#eb03fc']
      )


fig.update_layout(showlegend=False)
fig.update_xaxes(title='Content Duration')
fig.update_yaxes(title='Paid Course')
fig.show()


# We can see that **paid courses** have a higher duration, with an average of **2.5 hours**, whereas **free courses** have a median of **1.5 hours**.
# It is also worth noting that duration varies considerabily more on paid courses as well.
# 
# Let's see that same plot stratified by subject.

# In[ ]:


fig = px.box(courses_df,
       x='content_duration',
       y='subject',
       orientation='h',
       color='is_paid',
       title='Duration Distribution Across Subject and Type of Course',
       color_discrete_sequence=['#03cffc','#eb03fc']
      )


fig.update_xaxes(title='Content Duration')
fig.update_yaxes(title='Course Subject')
fig.show()


# # Exploring Paid Courses

# In[ ]:


# Filter paid courses
paid_courses_df = courses_df.query("price != 'Free'").sort_values('num_reviews',ascending=False)
paid_courses_df['price'] = paid_courses_df['price'].astype('float32')


# ## How are Course Prices Distributed?

# In[ ]:


fig = px.box(paid_courses_df,
             x     = 'subject',
             y     = 'price',
             color = 'subject',
             title = 'Course Prices x Subject',
             color_discrete_sequence = ['#03cffc','#0362fc','#eb03fc','#0ecc83'],
             hover_name = 'course_title',
            )

fig.update_layout(showlegend=False)
fig.update_yaxes(range=[0,220], title='Course Price')
fig.update_xaxes(title='Course Subject')
fig.show()


# In[ ]:


# Ridgeline Plot
fig = joypy.joyplot(paid_courses_df,
                    by      = 'subject',
                    column  = 'price',
                    figsize = (16,10),
                    grid    = 'both',
                    linewidth = 1,
                    colormap  = cm.winter,
                    fade      = True,
                    title     = 'Price Distribution Across Subjects',
                    overlap   = 2
                   )
plt.show()


# ## Top 25 Most Popular Paid Courses

# In[ ]:


top25_paid = paid_courses_df.sort_values("num_subscribers", ascending=False)[0:25].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()
fig = px.bar(top25_paid,
               y    = 'index',
               x    = 'num_subscribers',
               orientation = 'h',
               color       = 'num_subscribers',
               hover_name  = 'course_title',
               title       = 'Top 25 Most Popular Courses (by number of subscribers)',
               opacity     = 0.8,
               color_continuous_scale = px.colors.sequential.ice,
               height = 800,
              )

fig.update_layout(showlegend=False)
fig.update_xaxes(title='Number of Subscribers')
fig.update_yaxes(title='Course Title',showticklabels=False)
fig.show()


# # Exploring Free Courses

# In[ ]:


free_courses_df = courses_df.query("price == 'Free'").sort_values('num_reviews',ascending=False)


# In[ ]:


fig = px.scatter(free_courses_df,
       size = 'num_lectures',
       x    = 'num_subscribers',
       y    = 'num_reviews',
       trendline = 'ols',
       facet_col ='subject',
       color = 'subject',
       color_discrete_sequence = ['#03cffc','#0362fc','#eb03fc','#0ecc83'],
       hover_name= 'course_title',
       title='Engagement: Number of Reviews x Number of Subscribers'
      )

fig.update_layout(showlegend=False)
fig.update_xaxes(title='Number of Subscribers')
fig.update_yaxes(title='Number of Reviews')
fig.show()


# Our plot is being dragged too much due to some outliers.
# These outliers have either a great amount of reviews (>2000) or great number of subscribers (>10000).
# 
# It is interesting to point out that some courses are anomalies in terms of reviews x subscribers.
# 
# 

# In[ ]:


fig = px.scatter(free_courses_df.query("num_reviews <= 600 and num_subscribers <= 10000"),
       size = 'num_lectures',
       x    = 'num_subscribers',
       y    = 'num_reviews',
       trendline = 'ols',
       facet_col = 'subject',
       color     = 'subject',
       color_discrete_sequence = ['#03cffc','#0362fc','#eb03fc','#0ecc83'],
       hover_name = 'course_title',
       title      = 'Engagement: Number of Reviews x Number of Subscribers - Filtered'
      )

fig.update_layout(showlegend=False)
fig.update_xaxes(title='Number of Subscribers')
fig.update_yaxes(title='Number of Reviews')
fig.show()


# ## Top 25 Most Popular Free Courses

# In[ ]:


top25_free = free_courses_df.sort_values("num_subscribers", ascending=False)[0:25].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()
fig = px.bar(top25_free,
               y    = 'index',
               x    = 'num_subscribers',
               orientation = 'h',
               color       = 'num_subscribers',
               hover_name  = 'course_title',
               title       = 'Top 25 Most Popular Courses (by number of subscribers)',
               opacity     = 0.8,
               color_continuous_scale = px.colors.sequential.Aggrnyl,
               height = 800,
              )

fig.update_layout(showlegend=False)
fig.update_xaxes(title='Number of Subscribers')
fig.update_yaxes(title='Course Title',showticklabels=False)
fig.show()


# ### To be Added
# - Reviews/subscribers - engagement
# - Top 10 courses by subject
# - Top quality / price
# - Sunburst: subject + is_paid
# - Correlation: Price, reviews, subscribers, duration
