#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import json
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/youtube-new/INvideos.csv')
data.head()


# In[ ]:


print('Number of rows of in our dataset is {}'.format(data.shape[0]))
print('Number of columns of in our dataset is {}'.format(data.shape[1]))


# In[ ]:


# let's check the data type of all attributes present in our dataset
data.info()


# ### Processing the dates ( By looking at the trending_date and publish_time columns, we found that these columns are not in the correct datetime format)

# In[ ]:


data['trending_date'] = pd.to_datetime(data['trending_date'], format='%y.%d.%m')
data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
data[['trending_date','publish_time']].head()


# In[ ]:


# Separating columns for publish_date and publish_time
data['publish_date'] = data['publish_time'].dt.date
data['publish_time'] = data['publish_time'].dt.time
data[['trending_date','publish_date','publish_time']].head()


# ### Now,we can see that our date&time columns are in proper format, we will use them later when we will perform statistical analysis over time.

# In[ ]:


# We had seen that there are boolean columns too in our dataset so let's take a look on them.
bool_cols=[]
for cols in data.columns:
    if data[cols].dtype == 'bool':
        bool_cols.append(cols)
bool_cols    


# In[ ]:


for i in bool_cols:
    print(data[i].value_counts())


# ### We can see that we have very less number of True values in our bool type columns so it will good to remove these columns while modelling our dataset because these columns shows no variance and adds no information in predicting number of likes(also,it can be found that these variables  highly correlated with each other and this gives one more point why we should drop them)

# In[ ]:


print('The number of unique videos in our dataset is {}'.format(len(data['video_id'].unique()))) #prints no. of unique videos
print('The number of unique channels in our dataset is {}'.format(len(data['channel_title'].unique()))) #prints no. of unique channels
print('The number of unique category_id in our dataset is {}'.format(len(data['category_id'].unique()))) #prints no. of unique catgories


# ### 1. A number of videos appear multiple times in our dataset, as they were trending across multiple days. 
# ### 2. There are some channels whose videos were in trending more than one day.
# ### 3. We are dealing with 17 types of  video categories in our dataset,let's take a closer look at them.

# In[ ]:


# Processing category_id column
id_to_categ={}
with open('../input/youtube-new/IN_category_id.json', 'r') as f:
    categ_data = json.load(f)
    for category in categ_data['items']:
        id_to_categ[category['id']] = category['snippet']['title']


# In[ ]:


#data['category_id'].astype(str)
data['category'] = data['category_id'].map(id_to_categ)
data[['category_id','category']].head()


# ## Here, we have retrieved categories for each category_id from json file.

# In[ ]:


# Now,we will keep those columns on which we will do statistical analysis
df = data[['title','category','views','likes','dislikes','comment_count']]
df.head()


# In[ ]:


# Plotting a pie chart with plotly
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)

categ_counts = pd.DataFrame(df['category'].value_counts())
groups = categ_counts.index
values = categ_counts['category']
trace = go.Pie(labels=groups, values=values,
              hoverinfo='label+percent', textinfo='value')
#Plot
iplot([trace])


# ### From this pie-chart,we can see that around 50% of videos are from Entertainment category so it can be concluded that if any video belongs to entertainment category,there are more chances to go it in trending list.

# In[ ]:


# Now, let's take a look at the distributions of all numerical attributes
num_cols = ['views','likes','dislikes','comment_count']
for i in num_cols:
    plt.figure()
    sns.distplot(np.log(df[i]+1))
    plt.show()


# ### By above plots, it can be inferred that all numerical atrributes have a log-normal distribution

# In[ ]:


for i in num_cols:
    sns.barplot(df[i], df['category'])
    plt.show()


# ## Takeaway Points
# #### 1. Gaming videos are the most  viewed videos among trending videos.
# #### 2. Education videos are most less viewed videos among all trending videos.
# #### 3. Pets&Animals videos are most liked videos among all trending videos.
# #### 4. Gaming videos are most disliked videos among all trending videos.
# #### 5. Science&Tech videos are most commented videos among all trending videos.
# #### 6. Travel&Events videos are most less commented videos among all trending videos.

# ## Box Plots With Plotly

# In[ ]:


import plotly.express as px


fig = px.box(df, x="category", y="views")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# ### Takeaway Points:
# ### 1. There are a lot of outliers present in the Entertainment category, also most viewed video is from Entertainment category.
# ### 2. Most viewed video is from entertainment category having 125 million views

# In[ ]:


fig = px.box(df, x="category", y="likes")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# ### Takeaway Points:
# ### 1. As expected, most liked video is also from entertainment category and  chances are high that it's the same video having highest no. of views.

# In[ ]:


fig = px.box(df, x="category", y="dislikes")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# ### Most Disliked Video is also from entertainment category

# In[ ]:


fig = px.box(df, x="category", y="comment_count")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# ### 1.Most commented video is also from the entertainment category and chances are very high that one particular video have highest no of views,likes,dislikes and comments.
# ### 2. Hence, it can be inferred that if a video have very high views, there are more chances that it could be most liked,disliked and commented video

# In[ ]:


# Let's find out which is the most viewed,liked,disliked and commented video in trending list
print('Most viewed videos is {}'.format(df.loc[df[['views']].idxmax()]['title']))
print('Most liked videos is {}'.format(df.loc[df[['likes']].idxmax()]['title']))
print('Most disliked videos is {}'.format(df.loc[df[['dislikes']].idxmax()]['title']))
print('Most commented videos is {}'.format(df.loc[df[['comment_count']].idxmax()]['title']))


# **Now,we can clearly see that there was a particular video having highest no of views,likes,dislikesand comments.**

# In[ ]:


df['like_dislike_ratio'] = df['likes'] / df['dislikes']
df.head()


# In[ ]:


# Now,deleting all the duplicates videos from our dataset as we had seen that there were videos trending across multiple days
print('Shape of our dataset with duplicated videos is {}'.format(data.shape))
my_df = data[~data.video_id.duplicated(keep='last')]
print('Shape of our dataset after deleting duplicated videos is {}'.format(my_df.shape))
my_df.index.duplicated().any()


# In[ ]:


# Plotting the statistics for top10 most viewed videos
most_viewed = my_df.sort_values(by='views', ascending=False).head(10)
Title = most_viewed['title']

trace1 = go.Bar(x=Title,
                y=most_viewed['likes'],
                name='No.Of Likes')
trace2 = go.Bar(x=Title,
                y=most_viewed['dislikes'],
                name='No.Of Dislikes')
trace3 = go.Bar(x=Title,
                y=most_viewed['comment_count'],
                name='No.Of Comment')
data = [trace1, trace2, trace3]
layout = go.Layout(barmode='group')
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


corr_mat = df.corr()
sns.heatmap(corr_mat, annot=True, linewidths=0.5, fmt='.2f', cmap='YlGnBu')
plt.show()


# ### From above plot, we can see a very strong positive correlation between views vs likes and comment_count vs likes.

# In[ ]:


# Let's plot a scatter plot to visualize relationship between Views and Likes.
plt.figure(figsize=(20,10))
plt.scatter(df['views'], df['likes'], c=df['like_dislike_ratio'], cmap='summer', edgecolor='black',
           linewidth=1, alpha=0.4)
cbar = plt.colorbar()
cbar.set_label('Like-dislike Ratio')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.title('Views vs Likes of trending videos')
plt.xlabel('No Of views')
plt.ylabel('No of likes')
plt.grid()
plt.show()


# ### 1.From above plot, it can be inferred that as views on a video increases, no of likes also increases linearly.
# ### 2.The colorbar indicates the intensity of like-dislike ratio more bluish the color,more liked the video is.
# ### 3.As all these are trending videos,they have generally very high no of likes as comapred to dislikes.

# In[ ]:




