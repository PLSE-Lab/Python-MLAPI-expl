#!/usr/bin/env python
# coding: utf-8

# #        Analysis of the 100 most followed personalities on Twitter in 2019
# Last updated on 26th Dec,2019
# 
# ![](https://cdn-images-1.medium.com/max/1200/1*rrNJ_a566_xi5zsMJlCNtA.png)
# 
# *Created with Python's [stylecloud](https://github.com/minimaxir/stylecloud) library*
# 
# Twitter, founded in 2006 is an online social networking and microblogging service that allows users to post text-based status updates and messages of up to 280 characters in length. These messages are known as tweets. As of the second quarter of 2019, Twitter had 139 million monetizable daily active users (mDAU) worldwide[[source](https://www.statista.com/statistics/273172/twitter-accounts-with-the-most-followers-worldwide/)]

# ## Dataset Preparation

# In[ ]:


# Import the necessary libraries

import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)

# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5

# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Reading the dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/100-mostfollowed-twitter-accounts-as-of-dec2019/Most_followed_Twitter_acounts_2019.csv')
df.head()


# In[ ]:


df.info()


# Since the numerical column have been assigned a string datatype, let's first convert them to `int`.Also, it is a good idea to get rid of the punctuations.

# In[ ]:


df['Followers'] = df['Followers'].str.replace(',', '')
df['Following'] = df['Following'].str.replace(',', '')
df['Tweets'] = df['Tweets'].str.replace(',', '')

df[['Followers','Following','Tweets']] = df[['Followers','Following','Tweets']].astype(int)


# Also, let's convert all the names in the same case. This is for the sake of uniformity

# In[ ]:


# Convert all the names in the same case
df['Name'] = df['Name'].str.title()
df.Name.iloc[:3]


# # Exploratory Data Analysis
# 
# ## 1. Most followed personalities in 2019
# 
# 

# In[ ]:


x1 = df['Followers']/1000000
x1 = x1.round(2)

trace = go.Bar(x = df['Followers'][:15],
               y = df['Name'][:15],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black')),
               text=x1,
               textposition='auto',
               hovertemplate = "<br>Followers: %{x}")

layout = go.Layout(
                   title='Most followed accounts on Twitter worldwide as of December,2019',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of Followers in Millions'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()


# ### 1.1 Out of these, who follow the most people as of December 2019

# In[ ]:


df_following = df.sort_values('Following',ascending=False)
following = (df_following['Following']/1000).round(2)

trace = go.Bar(x = df_following['Following'][:15],
               y = df_following['Name'][:15],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black',width=0)),
               text=following,
               textposition='auto',
               hovertemplate = "<br>Following: %{x}")

layout = go.Layout(
                   #title='People who follow the most',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of people following in thousands'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()


# ### 1.2 Out of these, who follow the least number of people as of December 2019

# In[ ]:


df_following = df.sort_values('Following',ascending=False)
x = df_following['Following'][-15:]
trace = go.Bar(x = df_following['Following'][-15:],
               y = df_following['Name'][-15:],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black',width=0)),
               text=x,
               textposition='auto',
               hovertemplate = "<br>Following: %{x}")

layout = go.Layout(
                   #title='People who follow the least',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of people following in thousands'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()


# ## 2. Countries of most popular Twitter personalities

# In[ ]:


counts = df['Nationality/headquarters'].value_counts()
labels = counts.index
values = counts.values

pie = go.Pie(labels=labels, values=values,pull=[0.05, 0], marker=dict(line=dict(color='#000000', width=1)))
layout = go.Layout(title='Region wise Distribution')

fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)


# ## 3. Industries of most popular Twitter personalities

# In[ ]:


df['Industry'].replace({'music':'Music',
                       'news':'News',
                       'sports':'Sports'},inplace=True)




counts = df['Industry'].value_counts()
y= counts.values
trace = go.Bar(x= counts.index,
               y= counts.values,
               marker={'color': y,'colorscale':'Picnic'})

layout = go.Layout(
                   #title='Most followed accounts on Twitter worldwide as of December,2019',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Industry'),
                   yaxis=dict(title='Count'),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()


# ## 4. People who tweeted the most(out of the Top 100 most followed)
# 

# In[ ]:


df_tweets = df.sort_values('Tweets',ascending=False)
tweets = df_tweets['Tweets'][0:10]

trace = go.Bar(x = df_tweets['Tweets'][:10],
               y = df_tweets['Name'][:10],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black',width=0)),
               text=tweets,
               textposition='auto',
               hovertemplate = "<br>Tweets: %{x}")

layout = go.Layout(
                   #title='Most active',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of tweets'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()


# In[ ]:


# A generalised barplot and Treemap function
def barplot(data):

    x1 = data['Followers']/1000000
    x1 = x1.round(2)

    trace = go.Bar(x = data['Followers'][:15],
               y = data['Name'][:15],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black',width=0)),
               text=x1,
               textposition='auto',
               hovertemplate = "<br>Followers: %{x}")

    layout = go.Layout(
                   #title='Most followed Political accounts on Twitter worldwide as of December,2019',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of Followers in Millions'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

    fig = go.Figure(data = [trace], layout = layout)

    fig.update_layout(width=700,height=500)
    fig.show()


# In[ ]:


def treemap(data,title):
    import squarify 
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(16, 4.5)
    squarify.plot(sizes=data['Activity'].value_counts().values, label=data['Activity'].value_counts().index, 
              alpha=0.5,
              text_kwargs={'fontsize':10,'weight':'bold'},
              color=["red","green","blue", "grey","orange","pink","blue","cyan"])
    plt.axis('off')
    plt.title(title,fontsize=20)
    plt.show()


# ## 5. Most followed Musicians

# In[ ]:


music = df[df['Industry'] == 'Music']
music['Activity'].replace({'singer-songwriter':'Singer and Songwriter',
                            'Singer and songwriter':'Singer and Songwriter'},inplace=True)


barplot(music)
treemap(music,'Various Categories in Music')


# ### 5.1 Most popular Bands

# In[ ]:


band = df[df['Activity'] == 'Band']
barplot(band)


# ## 6. Most followed Politicians/Offcial Political accounts

# In[ ]:


politics = df[df['Industry'] == 'Politics']

barplot(politics)


# ## 7. Most followed Sportspersons/Sports Club

# In[ ]:


sport = df[df['Industry'] == 'Sports']
barplot(sport)


# In[ ]:


sport['Activity'].replace({'Football league':'Football League'},inplace=True)
treemap(sport,'Various Categories in Sports')


# ## 8. Most followed Films/Entertainment industry personalities

# In[ ]:


films= df[df['Industry'] == 'Films/Entertainment']
films['Activity'].replace({'Actor ':'Actor'},inplace=True)
barplot(films)
treemap(films,'Various Categories in Films/Entertainment')


# ## 9. Most followed Technology companies

# In[ ]:


tech = df[df['Industry'] =='Technology ']
barplot(tech)


# ## 10. Most Popular in U.S.A
# 

# In[ ]:


USA=df[df['Nationality/headquarters']=='U.S.A']
barplot(USA)


# ## 11. Most Popular in India

# In[ ]:


India=df[df['Nationality/headquarters']=='India']
barplot(India)


# ## 12. Most Popular in U.K

# In[ ]:


UK=df[df['Nationality/headquarters']=='U.K']
barplot(UK)


# Let's have a quick recap of our findings from the above analysis.
# 
# # People who ruled Twitter in 2019 : A Quick Recap
# 
# * ## Most followed people Globally
# 
# ![](https://cdn-images-1.medium.com/max/800/1*CeKkWNZc6ImdD5rXwsJPgg.png)
# 
# 
# * ## Most followed in USA
# 
# ![](https://cdn-images-1.medium.com/max/800/1*CeKkWNZc6ImdD5rXwsJPgg.png)
# 
# 
# * ## Most followed in India
# 
# ![](https://cdn-images-1.medium.com/max/800/1*u0m8NUQiipG2PQMnqXiajA.png)
# 
# 
# * ## Most followed in U.K
# 
# ![](https://cdn-images-1.medium.com/max/800/1*bVoJ4q0WnN2X0_5AkdQfxg.png)
# 
# 
# * ## Most followed Politicians
# 
# ![](https://cdn-images-1.medium.com/max/800/1*s_MNZxKaZ5X7qAPJYNASvA.png)
# 
# 
# * ## Most followed Sports Clubs/Associations
# 
# ![](https://cdn-images-1.medium.com/max/800/1*Vz90pW2ofw1DpFRWtffO0A.png)
# 
# 
# * ## Most followed Musicians
# 
# ![](https://cdn-images-1.medium.com/max/800/1*Kz7bbY2I9SgGX-J8JqA3lQ.png)
# 
# * ## Most followed Bands
# ![](https://cdn-images-1.medium.com/max/800/1*UUowA9X2fZRchXR7iR04eQ.png)
# 
# * ## Most followed People in Entertainment Industry
# ![](https://cdn-images-1.medium.com/max/800/1*20Qez05xEDw0BRC2ztoQBA.png)
# 
# * ## Most followed Sport Stars
# ![](https://cdn-images-1.medium.com/max/800/1*RuynKvS1bnvL7eutkgtELw.png)
# 
# 
# 
# 

# In[ ]:




