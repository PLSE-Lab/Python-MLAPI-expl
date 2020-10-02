#!/usr/bin/env python
# coding: utf-8

# This notebook is a detailed exploratory analysis of the Trending YouTube Video Statistics in specifically for the US dataset. The purpose of this notebook is to understand the various dimensions of the data and answer questions to figure out what makes a video trend. The questionnaire listed below is the reference I used to create this notebook. I have answered all these questions through the notebook and at the end have also provided a brief cheat sheet for YouTubers with the guidelines on how to get a video trending.
# 
# All the visualizations have been done through plotly as they are interactive, beautiful and awesome. I also did it as it's a great way to explore the plotly syntax.
# 
# Hope you enjoy reading this notebook and I would love your feedback or questions in the comment section.

# ### Questions to be answered
# - Which metric is most important for a video to start trending? most views, likes, dislikes, comments?
# - How are metrics like views, likes, dislikes, comments correlated with each other?
# - What category of videos trend the most?
# - What channels are performing the best? Channels with the most number of trending videos?
# - What videos were trending for the most number of days?
# - What is the best month, day, or hour to publish for a video to trend?
# - What are the most common words in trending video titles? Does length of title correlate with a video trending?
# - What factors to keep in mind to get a video trending (apart from the content obviously)?

# ### Notebook Steps
# - Import libraries and read data
# - Basic exploration
# - Data preprocessing
# - Extracting some new features
# - Splitting the dataset (full, first, last)
# - top 10 videos (views, likes, dislikes, comments)
# - bottom 10 videos (views, likes, dislikes, comments)
# - Channel analysis
# - Distribution by views, likes, dislikes, comments
# - Category analysis  
# - Visualize videos by number o trending days
# - Percentage of videos with comments/ratings disabled or removed
# - Best time to publish a video
# - Working with titles and tags
# - Cheat sheet

# ### Import libraries and read data

# In[ ]:


import numpy as np 
import pandas as pd
import json

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_columns', 50)


# In[ ]:


df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')


# In[ ]:


df.head()


# ### Basic data exploration

# In[ ]:


df.info()


# - 40,949 entries
# - 16 columns
# - trending_date and publish_time dtype needs to converted to datetime
# - split publish_time to date and time
# - insert category names for category_id
# - description has some missing data

# In[ ]:


df.describe()


# # Data Preprocessing

# ### Converting date and time columns to datetime

# In[ ]:


df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')


# In[ ]:


df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[ ]:


df.insert(5, 'publish_date', df['publish_time'].dt.date)


# In[ ]:


df['publish_time'] = df['publish_time'].dt.time


# In[ ]:


df['publish_date'] = pd.to_datetime(df['publish_date'])


# ### Processing category_id

# In[ ]:


id_to_cat = {}

with open('/kaggle/input/youtube-new/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_cat[category['id']] = category['snippet']['title']


# In[ ]:


id_to_cat


# In[ ]:


df['category_id'] = df['category_id'].astype(str)


# In[ ]:


df.insert(5, 'category', df['category_id'].map(id_to_cat))


# # Extracting some useful features

# ### publish_to_trend_days
# A new feature that shows the number of days it took for a video to start trending from the publish date

# In[ ]:


df.insert(8, 'publish_to_trend_days', df['trending_date'] - df['publish_date'])


# ### publish_month
# A new columns with the month the video was published. It will help us do analysis of the best month to publish a video.

# In[ ]:


df.insert(7, 'publish_month', df['publish_date'].dt.strftime('%m'))


# ### publish_day
# A new column with the day of the week the video was published as a string. It will help us do analysis of the best day of the week to publish a video.

# In[ ]:


df.insert(8, 'publish_day', df['publish_date'].dt.strftime('%a'))


# ### publish_hour
# A new column with the hour feature. It will halp us do analysis of the publish hour with the most trending videos

# In[ ]:


df.insert(10, 'publish_hour', df['publish_time'].apply(lambda x: x.hour))


# In[ ]:


# Let's take a look at our dataframe with the new features
df.head()


# # Splitting the dataset to full, first and last

# ### Duplicated entries

# In[ ]:


df['video_id'].nunique()


# In[ ]:


len(df['video_id'])


# Out of 40949 videos, only 6351 are unique as these videos were trending on multiple days. Creating two splits of this dataset for when the video 'first' started trending and when it was 'last' trending.

# In[ ]:


print(df.shape)
df_last = df.drop_duplicates(subset=['video_id'], keep='last', inplace=False)
df_first = df.drop_duplicates(subset=['video_id'], keep='first', inplace=False)
print(df_last.shape)
print(df_first.shape)


# In[ ]:


print(df['video_id'].duplicated().any())
print(df_last['video_id'].duplicated().any())
print(df_first['video_id'].duplicated().any())


# ### Adding total_trend_days feature to df_last

# In[ ]:


trend_days = df.groupby(['video_id'])['video_id'].agg(total_trend_days=len).reset_index()
df_last = pd.merge(df_last, trend_days, on='video_id')


# ### From here on,  we will only be using df_last as the dataset for analysis as we have extracted the one useful piece of information that the entire dataset had which is the number of days a particular video trends. The df_last dataset does not have any repetitions and is a more accurate representation of the data we need for this analysis.

# In[ ]:


df_last.head()


# In[ ]:


df_last.info()


# # Visualize top 10 by feature
# 
# We'll be creating a function to visualize the top 10 videos by feature and then call the function for any feature we desire

# In[ ]:


def top_10(df, col, num=10):
    sort_df = df.sort_values(col, ascending=False).iloc[:num]
    
    fig = px.bar(sort_df, x=sort_df['title'], y=sort_df[col])
    
    labels = []
    for item in sort_df['title']:
        labels.append(item[:10] + '...')
        
    fig.update_layout(title = {'text':'Top {} videos with the highest {}'.format(num, col),
                           'y':0.95,
                           'x':0.4,
                            'xanchor':'center',
                            'yanchor':'top'},
                 xaxis_title='',
                 yaxis_title=col,
                     xaxis = dict(ticktext=labels))
  
    fig.show()
    
    return sort_df[['video_id', 'title', 'channel_title','category', col]]


# ### Top 10 videos with the highest views

# In[ ]:


top_10(df_last, 'views', 10)


# ### Top 10 videos with the highest likes

# In[ ]:


top_10(df_last, 'likes')


# ### Top 10 videos with the highest dislikes

# In[ ]:


top_10(df_last, 'dislikes')


# ### Top 10 videos with the highest comment count

# In[ ]:


top_10(df_last, 'comment_count')


# - 3 entries in all the 4 features, shows that videos with high views are also prone to higher engagement
# - 6 common entries in top 10 views and likes, shows a high correlation between views and likes
# - 3 common entries in top 10 comment_count and dislikes, shows a correlation between comments and dislikes
# - 3 common entries in top 10 comment_count and likes, shows a correlation between comments and likes
# - comment_count has a correlation with views, likes and dislikes
# - likes and views are dominated by the music category and some from entertainment
# - dislikes and comments are a mix of entertainment, music, people & blogs (Logan Paul Vlogs categorised as Nonprofits & Acitvism is essentially people & blogs)

# # Visualize bottom 10 by feature
# Making some customizations to this function since we're populating data for lowest by feature, it is obvious that the entries with ratings_disabled=True will have 0 likes and dislikes, and entries with comments_disabled=True will have 0 comments by default. So we make sure that when we are searching for lowest likes, dislikes, and comments, ratings and comments haven't been disabled. 

# In[ ]:


def bottom_10(df, col, num=10):
    
    if col == 'likes' or col == 'dislikes':
        sort_df = df[df['ratings_disabled'] == False].sort_values(col, ascending=True).iloc[:num]
    elif col == 'comment_count':
        sort_df = df[df['comments_disabled'] == False].sort_values(col, ascending=True).iloc[:num]
    else:
        sort_df = df.sort_values(col, ascending=True).iloc[:num]
    
    fig = px.bar(sort_df, x=sort_df['title'], y=sort_df[col])
    
    labels = []
    for item in sort_df['title']:
        labels.append(item[:10] + '...')
        
    fig.update_layout(title = {'text':'Bottom {} videos with the lowest {}'.format(num, col),
                           'y':0.95,
                           'x':0.4,
                            'xanchor':'center',
                            'yanchor':'top'},
                 xaxis_title='',
                 yaxis_title=col,
                     )
  
    fig.show()
    
    return sort_df[['video_id', 'title', 'channel_title','category', 'total_trend_days', 'publish_to_trend_days', 'views', 'likes', 'dislikes', 'comment_count', 'ratings_disabled', 'comments_disabled']]


# ### Bottom 10 videos with the lowest views

# In[ ]:


bottom_10(df_last, 'views')


# ### Bottom 10 videos with the lowest likes

# In[ ]:


bottom_10(df_last, 'likes')


# ### Bottom 10 videos with the lowest dislikes

# In[ ]:


bottom_10(df_last, 'dislikes')


# ### Bottom 10 videos with the lowest comments

# In[ ]:


bottom_10(df_last, 'comment_count')


# This is a mystery
# - Why are these videos trending that have such low engagement i.e. views, likes, dislikes, comments?
# - It's not that these videos started trending immediately after publishing, they range from a couple of days to hundreds in some cases
# - In cases where ratings and comments have been disabled, there's a possibility that there was an influx of dislikes or negative comments which triggered it to trend leading the publisher to disable them, also, once it's disabled we cannot view the engagement as the ratings and comments then just say 0
# - But videos are trending even in cases where the ratings and comments haven't been disabled and have minimal engagement
# - A lot of these videos have trended for 1-3 days, it could be a glitch in the YouTube algorithm that triggered it to trend, and fixed once identified
# - Another thing to note is that none of these bottom 10 entries are from the music category but from a variety of categories like people& blogs, travel & events, entertainment etc.
# - Maybe something to do with the title and tags

# # Channel Analysis

# ### Channels with the most number of trending videos

# In[ ]:


top_channels = df_last.groupby(['channel_title'])['channel_title'].agg(code_count=len).sort_values("code_count", ascending=False)[:20].reset_index()

fig = px.bar(top_channels, x=top_channels['channel_title'], y=top_channels['code_count'])

fig.update_layout(title = {'text':'Channels with the most trending videos',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Number of trending videos')

fig.show()


# It's evident that channels that create the most content have the most trending videos as well. Even though channels in the music category have the most views and likes, it's channels in the sports and entertainment category that post a video regularly and thus have more trending videos

# ### Channels with the most views

# In[ ]:


top_channels_views = df_last.groupby(['channel_title'])['views'].agg(total_views=sum).sort_values("total_views", ascending=False)[:20].reset_index()

fig = px.bar(top_channels_views, x=top_channels_views['channel_title'], y=top_channels_views['total_views'])

fig.update_layout(title = {'text':'Channels with the most views',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Total views')

fig.show()


# ### Channels with the most likes

# In[ ]:


top_channels_likes = df_last.groupby(['channel_title'])['likes'].agg(total_likes=sum).sort_values("total_likes", ascending=False)[:20].reset_index()

fig = px.bar(top_channels_likes, x=top_channels_likes['channel_title'], y=top_channels_likes['total_likes'])

fig.update_layout(title = {'text':'Channels with the most likes',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Total likes')

fig.show()


# ### Channels with the most dislikes

# In[ ]:


top_channels_dislikes = df_last.groupby(['channel_title'])['dislikes'].agg(total_dislikes=sum).sort_values("total_dislikes", ascending=False)[:20].reset_index()

fig = px.bar(top_channels_dislikes, x=top_channels_dislikes['channel_title'], y=top_channels_dislikes['total_dislikes'])

fig.update_layout(title = {'text':'Channels with the most dislikes',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Total dislikes')

fig.show()


# ### Channels with most comments

# In[ ]:


top_channels_comments = df_last.groupby(['channel_title'])['comment_count'].agg(total_comments=sum).sort_values("total_comments", ascending=False)[:20].reset_index()

fig = px.bar(top_channels_comments, x=top_channels_comments['channel_title'], y=top_channels_comments['total_comments'])

fig.update_layout(title = {'text':'Channels With Most Comments',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Total Comments')

fig.show()


# But when it comes down to views, likes, dislikes, and comments, the music and entertainment channels do the best

# # Distribution of data in views, likes, dislikes, and comments

# In[ ]:


print("Views quantiles")
print(df_last['views'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Likes quantiles')
print(df_last['likes'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Disikes quantiles')
print(df_last['dislikes'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')
print('Comments quantiles')
print(df_last['comment_count'].quantile([.01,.25,.5,.75,.99]))
print('---------------------------')


# The data is extremely skewed with a big gap between the first 75% and the last 25%
# - 75% of views are less than 1.5 M and the last 25% goes upto 2.5 B
# - 75% of likes are less than 40 K and the last 25% goes upto 800 K
# - 75% of dislikes are less than 1500 and the last 25% goes upto 35 K
# - 75% of comments are less than 4000 and the least 25% goes upto 80 K

# ### Logarithmizing the data
# We will logarithmize the data to fix the above mentioned problem as the they will play a big role in a boxplot

# In[ ]:


df_last['views_log'] = np.log(df_last['views'] + 1)
df_last['likes_log'] = np.log(df_last['likes'] + 1)
df_last['dislikes_log'] = np.log(df_last['dislikes'] + 1)
df_last['comments_log'] = np.log(df_last['comment_count'] + 1)


# # Category Analysis

# ### Number of videos sorted by category

# In[ ]:


fig = px.histogram(df_last, x=df_last['category'])

fig.update_layout(title = {'text':'Number of videos sorted by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='Count',
                 template='seaborn')

fig.show()


# As expected, entertainment and music have the most trending content

# ### Views distribution by category

# In[ ]:


fig = px.box(df_last, x=df_last['category'], y=df_last['views_log'])

fig.update_layout(title = {'text':'Views distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='views_log',
                 template='seaborn')

fig.show()


# ### Likes distribution by category

# In[ ]:


fig = px.box(df_last, x=df_last['category'], y=df_last['likes_log'])

fig.update_layout(title = {'text':'Likes distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='likes_log',
                 template='seaborn')

fig.show()


# ### Dislikes distribution by category

# In[ ]:


fig = px.box(df_last, x=df_last['category'], y=df_last['dislikes_log'])

fig.update_layout(title = {'text':'Dislikes distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='dislikes_log',
                 template='seaborn')

fig.show()


# ### Comments distribution by category

# In[ ]:


fig = px.box(df_last, x=df_last['category'], y=df_last['comments_log'])

fig.update_layout(title = {'text':'Comments distribution by category',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='',
                 yaxis_title='comments_log',
                 template='seaborn')

fig.show()


# - Music and gaming have have the highest engagement in all the quartiles
# - Fourth quartile: Music is the highest in all 4 metrics
# - Third quartile: Music is the highest in all 4 metrics
# - Second quartile: Gaming and music are the highest in all 4 metrics
# - First quartile: Music , comedy and gaming are the highest in all 4 metrics
# - Min value: Education is the highest in all 4 metrics

# # Visualize videos by number of trending days

# In[ ]:


video_trend = df_last.groupby('total_trend_days')['total_trend_days'].agg(count=len).sort_values('count', ascending=False).reset_index()

fig = px.bar(video_trend, x=video_trend['total_trend_days'], y=video_trend['count'])

fig.update_layout(title = {'text':'Number of videos arranged by trending days',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='Total trend days',
                 yaxis_title='Number of videos',
                 template='seaborn')

fig.show()   


# - Most videos trend for less than 10 days
# - As the trend days increase, the number of videos reduce 
# - Vieos trending for days 5,6,7 seems like the sweet spot for most
# - Max nuber of trending days is 30
# - There's only one video that has trended for 30 days

# # Percentage of videos with comments/ratings disabled or removed

# In[ ]:


com = 100.0*len(df_last[df_last['comments_disabled'] == True]) / len(df_last['comments_disabled'])
rat = 100.0*len(df_last[df_last['ratings_disabled'] == True]) / len(df_last['ratings_disabled'])
err = 100.0*len(df_last[df_last['video_error_or_removed'] == True]) / len(df_last['video_error_or_removed'])

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Bar(x=[com], name='Percentage of videos with comments disabled'))
fig.add_trace(go.Bar(x=[rat], name='Percentage of videos with ratings disabled'))
fig.add_trace(go.Bar(x=[err], name='Percentage of videos with error or removed'))

fig.update_layout(title = {'text':'Percentage of videos with comments/ratings disabled or removed',
                           'y':0.9,
                           'x':0.4},
                 xaxis_title='Percentage',
                 yaxis_title='')

fig.show()


# - Only 1.65% of the videos have comments disabled
# - Only 0.5% of the videos have ratings disabled
# - Only 0.06% of the videos have been removed
# - These percentages are too small to draw any conclusions from

# # Best time to publish a video

# ### Videos published arranged by months

# In[ ]:


best_month = df_last.groupby('publish_month')['publish_month'].agg(count=len).sort_values('count', ascending=False).reset_index()

fig = px.bar(best_month, x=best_month['publish_month'], y=best_month['count'])

fig.update_layout(title = {'text':'Videos published arranged by months',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='Publish months',
                 yaxis_title='Count',
                 xaxis = dict(
        tickmode = 'array',
        tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
                  

fig.show()


# - Looks like most trending videos are published in Jan and Dec
# - Very few videos published in Jul, Aug, Sep, Oct are trending
# - In other words, winter is a better time to publsih than summer

# ### Videos published arranged by day of the week

# In[ ]:


best_day = df_last.groupby('publish_day')['publish_day'].agg(count=len).sort_values('count', ascending=False).reset_index()

fig = px.bar(best_day, x=best_day['publish_day'], y=best_day['count'])

fig.update_layout(title = {'text':'Videos published arranged by day of the week',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='Publish days',
                 yaxis_title='Count') 

fig.show()


# - Its better to post on a weekday as oppose to a weekend

# ### Videos published arranged by hour of the day

# In[ ]:


best_hour = df_last.groupby('publish_hour')['publish_hour'].agg(count=len).reset_index()

count = best_hour['count']
bins = [0, 155, 223, 327, 578]
labels = ['Bad','Decent','Good' ,'Great']

colors = {'Decent': 'orange',
          'Bad': 'red',
          'Good': 'lightgreen',
          'Great': 'darkgreen'}

# Build dataframe
color_df = pd.DataFrame({'y': count,
                   'x': range(len(count)),
                   'label': pd.cut(count, bins=bins, labels=labels)})

fig = go.Figure()

bars = []
for label, label_df in color_df.groupby('label'):
    bars.append(go.Bar(x=label_df.x,
                       y=label_df.y,
                       name=label,
                       marker={'color': colors[label]}))

go.FigureWidget(data=bars)


# - A great time to publish a video is 1:00 pm to 6:00 pm
# - A good time to publish a video is 7:00 pm to 11:00 pm
# - A decent time to publish a videos is 12:00 am to 5:00 am
# - A bad time to publish a videos is 6:00 am to 11:00 am
# - In other words, it's better to post in the second half of the day

# # Working with titles and tags

# ## Creating some new title features

# ### title_length and no_of_tags

# In[ ]:


df_last['title_length'] = df_last['title'].apply(lambda x: len(x))


# In[ ]:


df_last['no_of_tags'] = df_last['tags'].apply(lambda x: len(x.split('|')))


# ### Number of videos sorted by title_length

# In[ ]:


fig = px.histogram(df_last, x=df_last['title_length'])

fig.update_layout(title = {'text':'Number of videos sorted by title length',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='Title length',
                 yaxis_title='Count')

fig.show()


# - Normally distributed
# - Most trending videos have 30 - 60 characters in thier title

# ### Number of videos sorted by number of tags

# In[ ]:


fig = px.histogram(df_last, x=df_last['no_of_tags'])

fig.update_layout(title = {'text':'Number of videos sorted by number of tags',
                           'y':0.95,
                           'x':0.5},
                 xaxis_title='Number of Tags',
                 yaxis_title='Count')

fig.show()


# - A bunch of videos have only 1 tag
# - 150 - 200 videos have tags between 5 and 30 each
# - Lesser videos have tags more than 40

# ### Most common words in video titles

# In[ ]:


eng_stopwords = set(stopwords.words('english'))


# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='black',
                     stopwords=stopwords,
                     max_words=150,
                     max_font_size=40,
                     ).generate(str(df_last['title']))

plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.title('Most common words in title', fontsize=15)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### Most common tags

# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',
                     stopwords=stopwords,
                     #max_words=150,
                     max_font_size=50,
                     ).generate(str(df_last['tags']))

plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.title('Most common tags', fontsize=15)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### Follow these steps next time you upload a video on YouTube and you have just might get it trending
# 
# - Make sure the content is something in Entertainment or Music
# - Upload the video between November and February
# - Upload it on a weekday preferrably Wednesday
# - Upload it between 1:00 pm and 6:00 pm
# - Make sure the title is between 30 and 60 characters
# - Don't include more than 30 tags
