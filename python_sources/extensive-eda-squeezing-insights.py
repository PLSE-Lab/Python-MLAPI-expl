#!/usr/bin/env python
# coding: utf-8

# ### Note: I chose to do EDA On Youtube Statistics of India. To perform EDA on any other country's dataset, just replace the file name and enjoy!

# In[ ]:


# Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly
from wordcloud import WordCloud, STOPWORDS


import json
from datetime import datetime


# In[ ]:


# Getting our data

youtube = pd.read_csv("../input/youtube-new/INvideos.csv")


# **Let's look at some of the rows of our data.**

# In[ ]:


youtube.head()


# **Let's look at the number of rows and columns in our data.**

# In[ ]:


print("Number of rows in data :",youtube.shape[0])
print("Number of columns in data :", youtube.shape[1])


# **Now, let's see what columns are there in our data.**

# In[ ]:


youtube.columns


# **Let's look at the number of unique values in each column.**

# In[ ]:


print(youtube.nunique())


# **Finally, let's get some of the information about our data.**

# In[ ]:


youtube.info()


# In[ ]:


temp = youtube.copy() # creating a copy just in case


# **We can easily notice that all columns except the 'description' have non-null values so let's do a sanity check!**

# In[ ]:


youtube[youtube.description.isnull()].head()


# In[ ]:


desc = youtube['description'].isnull().sum()/(len(youtube))*100

print(f"Description column has {desc.round(2)}% null values.")


# **Indeed, the some channels did not even bother to write a description of their video or maybe this is just a data collection fault.**

# # Data Cleaning

# **Let's change the description to something which can describe that description is indeed empty. - We will replace the description with an empty string.**

# In[ ]:


# Replacing all the NaN values to a empty string

youtube["description"] = youtube["description"].fillna(value="")


# **I don't know if you can notice, but the time and date columns looks odd and verbose! Especially the time column. So, let's clean them and make them readable by parsing it using datetime module function to_datetime.**

# In[ ]:


# Making format of date and time better

youtube.trending_date = pd.to_datetime(youtube.trending_date, format='%y.%d.%m')
youtube.publish_time = pd.to_datetime(youtube.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')
youtube.category_id = youtube.category_id.astype(str)

youtube.head()


# **Much better! But if we want to watch a video we don't refer to with categoryID 5 but rather using a category. So, why not our data can do this? Let's create a new column 'category' using our json file which contains categories for each ID and map both of them.** 

# In[ ]:


# creating a new category column by loading json 

id_to_category = {}

with open('../input/youtube-new/IN_category_id.json' , 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']
        
youtube['category'] = youtube['category_id'].map(id_to_category)


# In[ ]:


youtube.head()


# **Awesome! Now, let's see what video categories do we have in our data.**

# In[ ]:


# Looking at each category and number of unique values

youtube['category'].value_counts()


# **I am still not statisfied with the data because I personally like to look at likes and dislikes beacuse what If we have a lot of likes but also a lot of dislikes? So, Is there anything which can quantify both of them at the same time? Yes!!**
# 
# **Let's create a new column 'likes_dislikes_ratio' and note that there can be vids with 0 dislikes and likes so we will have to handle that to avoid odd values**

# In[ ]:


zero_dislikes = len(youtube.dislikes)-youtube.dislikes.astype(bool).sum(axis=0)
zero_likes = len(youtube.likes)-youtube.likes.astype(bool).sum(axis=0)

print(f"There are {zero_likes} videos with 0 likes.")
print(f"There are {zero_dislikes} videos with 0 dislikes.")


# **So, if we have any zero in denominator(dislikes) then we will keep the its value same as number of likes and if we don't then we will calculate the ratio.** 

# In[ ]:


# this will hold all the ratios
likes_dislikes = {}

for i in range(len(youtube['likes'])):
    
    # if the value of dislikes is not zero
    if youtube['dislikes'][i]!=0:
        
        # compute the ratio
        likes_dislikes[i]=youtube['likes'][i]/youtube['dislikes'][i]
        
    else:
        
        # simply use the likes value
        likes_dislikes[i]=youtube['likes'][i]
        
youtube['likes_dislikes_ratio'] = likes_dislikes.values()


# In[ ]:


youtube.head()


# **A number of videos can appear multiple times in our dataset, as they were trending across multiple days. Thus, we will remove these duplicated entries for now, and only keep the last entry of each video, as that entry will be the most recent statistic.**

# In[ ]:


print(f"Does the data contain duplicate video_ids? - {youtube.video_id.duplicated().any()}")


# In[ ]:


print(f"Before Deduplication : {youtube.shape}")
youtube = youtube[~youtube.video_id.duplicated(keep='last')]
print(f"After Deduplication : {youtube.shape}")

print(f"Does the data contain duplicate video_ids now? - {youtube.video_id.duplicated().any()}")


# **Let's look at the descriptive properties of the data**
# 
# **I don't like to look at large scientific notations and float values when they are not. So, here I am using a custom formatter to make the values look a bit nicer. If you don't understand this code, it is fine.**

# In[ ]:


# Creating a custom formatter for pandas describe function

import contextlib
import numpy as np
import pandas as pd
import pandas.io.formats.format as pf
np.random.seed(2015)

pd.set_option('display.max_colwidth', 100)

@contextlib.contextmanager
def custom_formatting():
    orig_float_format = pd.options.display.float_format
    orig_int_format = pf.IntArrayFormatter

    pd.options.display.float_format = '{:0,.2f}'.format
    class IntArrayFormatter(pf.GenericArrayFormatter):
        def _format_strings(self):
            formatter = self.formatter or '{:,d}'.format
            fmt_values = [formatter(x) for x in self.values]
            return fmt_values
    pf.IntArrayFormatter = IntArrayFormatter
    yield
    pd.options.display.float_format = orig_float_format
    pf.IntArrayFormatter = orig_int_format

with custom_formatting():
    print(youtube[['views','likes','dislikes','comment_count']].describe())


# **OBSERVATIONS**
# 
# (I) Half of the videos in our dataset have:
# <ul>
#     <li>Views greater than 205k.</li>
#     <li>Likes greater than ~1700.</li>
#     <li>Dislikes greater than 194.</li>
#     <li>Comments greater than 195.</li>
# </ul>
# 
# (II)<ul>
#     <li>Maximum Views = 125,432,237</li>
#     <li>Maximum Likes = 2,912,710</li>
#     <li>Maximum Dislikes = 1,545,017</li>
#     <li>Maximum Comments = 807,558</li>
# </ul>

# # Visualizations of Continous Features

# In[ ]:


# Plotting the Heatmap of the columns using correlation matrix

f,ax = plt.subplots(figsize=(20, 10))
sns.heatmap(youtube.corr(), annot=True, linewidths=0.5,linecolor="red",ax=ax)
plt.show()


# ### Number of videos by year

# In[ ]:


# Extracting the year from the 'trending date' and converting to a list
video_by_year = temp["trending_date"].apply(lambda x: '20' + x[:2]).value_counts().tolist()


# In[ ]:


# Plotting a pie chart for number of videos by year

labels = ['2017','2018']
values = [video_by_year[1],video_by_year[0]]
colors = ['turquoise', 'royalblue']

trace = go.Pie(labels=labels, values=values, textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# **Thus, in our data, ~76% of the videos are from 2018 and only 24% are from 2017.**

# ### Distribution of Log of Continous variables
# 
# **The reason for choosing log distribution is beacause it is very hard to get insights from normal number as they are very large so you will just a large peak. So, taking log will scale it to lower values to analyse it better!**

# In[ ]:


def plot_distributions(col, i, colors):

    column_name = col+'_log'
    youtube[column_name] = np.log(youtube[col] + 1)

    group_labels = [column_name]
    hist_data = [youtube[column_name]]
    
    colors = [colors]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels = group_labels, colors=colors,
                             bin_size=0.1, show_rug=False)

    # Add title
    title_dict = {1:'Views', 2:'Likes', 3:'Dislikes', 4:'Likes and Dislikes Ratio', 5:'Comment Count'}
    fig.update_layout(width=700, title_text= title_dict[i]+' Log Distribution')
    fig.show()


# In[ ]:


columns_list = ['views', 'likes', 'dislikes', 'likes_dislikes_ratio', 'comment_count']
colors = ['coral', 'darkmagenta', 'green', 'red', 'blue']

for i in range(0,5):
    plot_distributions(columns_list[i], i+1, colors[i])


# **OBSERVATIONS**
# 
# **All of the continous distributions are close are log-normal which is very common when studying internet based data like views, dwell time of a user etc. For more read visit this [link](https://www.wikiwand.com/en/Log-normal_distribution#Occurrence_and_applications).**

# ## Single Variate Statistics
# 
# **Let's look at how many percentage of trending videos got more than certain number of views, likes, dislikes etc.**

# In[ ]:


one_mil = youtube[youtube['views'] > 1000000]['views'].count() / youtube['views'].count() * 100

print(f"Only {round(one_mil, 2)}% videos have more than 1 Million views.")


# In[ ]:


hundered_k = youtube[youtube['likes'] > 100000]['likes'].count() / youtube['likes'].count() * 100

print(f"Only {round(hundered_k, 2)}% videos have more than 1OOK Likes.")


# In[ ]:


five_k = youtube[youtube['dislikes'] > 5000]['dislikes'].count() / youtube['dislikes'].count() * 100

print(f"Only {round(five_k, 2)}% videos have more than 5K Dislikes.")


# In[ ]:


five_k = youtube[youtube['comment_count'] > 5000]['comment_count'].count() / youtube['comment_count'].count() * 100

print(f"Only {round(five_k, 2)}% videos have more than 5K Comments.")


# ### Most Viewed, Liked, Disliked and Commented Videos

# In[ ]:


most_likes = youtube.loc[youtube[['views']].idxmax()]['title']
most_views = youtube.loc[youtube[['likes']].idxmax()]['title']
most_dislikes = youtube.loc[youtube[['dislikes']].idxmax()]['title']
most_comments = youtube.loc[youtube[['comment_count']].idxmax() ]['title']

print(f"Most Viewed Video : {most_likes.to_string(index=False)}\n")
print(f"Most Liked Video : {most_views.to_string(index=False)}\n")
print(f"Most Disliked Video : {most_dislikes.to_string(index=False)}\n")
print(f"Video with most comments : {most_comments.to_string(index=False)}")


# **Looks like youtube won in their own game! XD**

# In[ ]:


most_likes_ratio = youtube.loc[youtube[['likes_dislikes_ratio']].idxmax() ]['title']

print(f"Video with highest likes ratio : {most_likes_ratio.to_string(index=False)}\n")


# **Notice that we didn't get Youtube Rewind as the answer in case of maximum likes and dislike ratio. So, it was indeed a useful feature and we did a great job adding it!**

# ### What are the categories with largest number of trending videos?

# In[ ]:


# category had the largest number of trending videos

Category = youtube.category.value_counts().index
Count = youtube.category.value_counts().values

fig = px.bar(youtube, x=Category, y=Count, labels={'x':'Category', 'y' : 'Number of Videos'})

fig.update_traces(marker_color='mistyrose', marker_line_color='darkmagenta',
                  marker_line_width=1.5)

fig.update_layout(title_text='Video Per Category')
fig.show()


# **Entertainment category contains the largest number of trending videos among other categories with around ~7600 videos, followed by News & Politics with 2505 videos, followed by People&Blogs category with around 1232 videos, and so on.**

# ### Category

# In[ ]:


# Plotting a pie chart for top 10 channels with most trending videos

x = youtube.channel_title.value_counts().head(10).index
y = youtube.channel_title.value_counts().head(10).values

trace = go.Pie(labels=x, values=y, textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# **That is a very close math between all the channels! This wasn't very insignful so let's move on to doing a top 10 of all the continous features - Views, Likes, Dislikes etc. Believe me it will be a lot of fun!**

# ## Top 10 for each Continous Feature

# ### Views

# In[ ]:


sort_by_views = youtube.sort_values(by="views" , ascending = False)

Title = sort_by_views['title'].head(10)
Views = sort_by_views['views'].head(10)

fig = px.bar(youtube, x=Title, y=Views, labels={'x':'Title', 'y' : 'Number of views'})

fig.update_traces(marker_color='gold', marker_line_color='darkmagenta',
                  marker_line_width=1.5)

fig.update_layout(title_text='Top 10 Most Watched Videos')
fig.show()


# ### Likes

# In[ ]:


sort_by_likes = youtube.sort_values(by ="likes" , ascending = False)

Title = sort_by_likes['title'].head(10)
Likes = sort_by_likes['likes'].head(10)

fig = px.bar(youtube, x=Title, y=Likes, labels={'x':'Title', 'y' : 'Number of Likes'})

fig.update_traces(marker_color='dodgerblue', marker_line_color='olive',
                  marker_line_width=2.5)

fig.update_layout(title_text='Top 10 Most Liked Videos')
fig.show()


# ### Dislikes

# In[ ]:


sort_by_dislikes = youtube.sort_values(by = "dislikes" , ascending = False)

Title = sort_by_dislikes['title'].head(10)
Dislikes = sort_by_dislikes['dislikes'].head(10)

fig = px.bar(youtube, x=Title, y=Dislikes, labels={'x':'Title', 'y' : 'Number of Dislikes'})

fig.update_traces(marker_color='tomato', marker_line_color='#000000',
                  marker_line_width=1.5)

fig.update_layout(title_text='Top 10 Most Disliked Videos',width=1200,
    height=800)
fig.show()


# ### Comments

# In[ ]:


sort_by_comments = youtube.sort_values(by = "comment_count" , ascending = False)

Title = sort_by_comments['title'].head(10)
Comments = sort_by_comments['comment_count'].head(10)

fig = px.bar(youtube, x=Title, y=Comments, labels={'x':'Title', 'y' : 'Number of Comments'})

fig.update_traces(marker_color='papayawhip', marker_line_color='darkblue',
                  marker_line_width=2.5)

fig.update_layout(title_text='Top 10 Videos with Most Comments',width=950,
    height=700)
fig.show()


# ### Like and Dislike Ratio

# In[ ]:


sort_by_ldr = youtube.sort_values(by = "likes_dislikes_ratio" , ascending = False)

Title = sort_by_ldr['title'].head(10)
Comments = sort_by_ldr['likes_dislikes_ratio'].head(10)

fig = px.bar(youtube, x=Title, y=Comments, labels={'x':'Title', 'y' : 'Like/Dislike'})

fig.update_traces(marker_color='cyan', marker_line_color='darkred',
                  marker_line_width=2.5)

fig.update_layout(title_text='Top 10 Videos with Most Like to Dislike Ratio',width=1100, height = 800)
fig.show()


# **Thanks to Plotly's Awesome Interactive Visualization, it is very easy to drive insights from above drawn plots.**

# # Visualization of Text Features
# 
# **Let's create a WordCloud for the text features of (video title, description and tags), which is a way to visualize most common words in the titles; the more common the word is, the bigger its font size is.**

# ### Video Title

# In[ ]:


# Utility Function for creating word cloud

def createwordcloud(data, bgcolor, title):
    plt.figure(figsize=(15,10))
    wc = WordCloud(width=1200, height=500, 
                         collocations=False, background_color=bgcolor, 
                         colormap="tab20b").generate(" ".join(data))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')


# In[ ]:


# WordCloud for Title Column

title = youtube['title']
createwordcloud(title , 'black' , 'Commonly used words in Titles')


# **OBSERVATION**
# 
# **Episodes is the most popular word along with full, song and new. These all are related to Entertainment which makes sense as it is the category with most number of trending videos.**

# ### Description

# In[ ]:


# WordCloud for Description Column

description = youtube['description'].astype('str')
createwordcloud(description , 'black' , 'Commonly used words in Description')


# **OBSERVATION**
# 
# **The results were predictable. Most of the youtube videos have profile links in their description section to increase reachibility among viewers which explains the abundance of words like twitter, facebook etc. Also, https is in the link of these links itself so it is ultimately the most used word.**

# ### Tags

# In[ ]:


# WordCloud for Tags Column


tags = youtube['tags'].astype('str')
createwordcloud(tags , 'black' , 'Commonly used words in Tags')


# **OBSERVATION**
# 
# **Most used tags are video, new, latest, song, show, movie and serial which are related to Entertainment category so this means that tags used by channels are pretty relatable to the title and aren't just anything.**

# **Most of the work is done so now we are left with a time feature and three binary categorical features. So, let's start with the time feature which is very interesting!**

# ### Time and Date Column Analysis
# 
# **To analyse this time series feature we will find the difference between the time a video was trending and the video was published and plot it against the number of views. It will tell us how the views vary day by day after a video is published.**

# In[ ]:


youtube['publish_date'] = pd.to_datetime(youtube['publish_time'])
youtube['difference'] = (youtube['trending_date'] - youtube['publish_date']).dt.days


# In[ ]:


fig = px.bar(youtube, x=youtube['trending_date'], y=youtube['views'], labels={'x':'Trending Date', 'y' : 'Number of Views'})

fig.update_traces(marker_color='darkred', marker_line_color='#000000',
                  marker_line_width=0.5)

fig.update_layout(title_text='Trending Date VS Number of Views')
fig.show()


# **OBSERVATION**
# 
# **Looks like in between December 2017 and January 2018 there was a significant increase in the number of views which is might be because a lot of youtubers publish their summary of the year videos which views like to watch a lot as it summarises the whole past year.**
# 
# **Also, there is 3 days gap in Jan and 7 Days gap in April.**

# ## Trending Videos having Errors
# 
# **Let's see how many trending videos got removed or had some error, we can use video_error_or_removed column in the dataset.**

# In[ ]:


error_or_removed = youtube["video_error_or_removed"].value_counts().tolist()

labels = ['No','Yes']
values = [error_or_removed[0],error_or_removed[1]]
colors = ['orange', 'yellow']

trace = go.Pie(labels=labels, values=values, textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# ## Trending Videos with Comments Disabled

# In[ ]:


error_or_removed = youtube["comments_disabled"].value_counts().tolist()

labels = ['No','Yes']
values = [error_or_removed[0],error_or_removed[1]]
colors = ['pink', 'purple']

trace = go.Pie(labels=labels, values=values, textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# ## Trending Videos with Ratings Disabled

# In[ ]:


error_or_removed = youtube["ratings_disabled"].value_counts().tolist()

labels = ['No','Yes']
values = [error_or_removed[0],error_or_removed[1]]
colors = ['khaki', 'olive']

trace = go.Pie(labels=labels, values=values, textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# ### Trending videos with both Comments and Ratings Disabled

# In[ ]:


youtube[(youtube["comments_disabled"] == True) & (youtube["ratings_disabled"] == True)]['category'].value_counts()


# **Trending videos in category of Eduction, News & Politics, Entertainment and People & Blogs have the most number of such videos. I am not sure why they have disabled comments in Eductional videos because views might have some doubts!**

# > TASKS FOR YOU
# 
# (I) Provide suggestion, improvements and Criticism! (if any) :)
# 
# (II) Please do Upvote if you want to see more content like this!

# # THANKS FOR READING!
