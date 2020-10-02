#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Exploring Hackers News Post
# 
# In this project, we will compare two types of post from [Hacker News](https://news.ycombinator.com/), a popular site where technology related stories or posts are voted and commented upon. The two types of posts we'll explore begin with either `Ask HN` or `Show HN`.
# 
# Users sumbit `Ask HN` posts to ask the Hacker News community a specific questions such as "How to improve my personal website?" Like wise, user sumbit `Show HN` posts to show the Hacker News community a project, product, or just generally something interesting.
# 
# We'll compare these two types of posts to determine the follwing:
# 
# - Do `Ask HN` or `Show HN` receive more coments on average?
# - Do posts created at a certain time receive more comments on average?
# 
# It should be noted that the [data set](https://www.kaggle.com/hacker-news/hacker-news-posts) we are working on has been reduced from almost 300,000 rows to approximately 20,000 rows by removing all submissions that did not receive any comments, and then randomly samling from the remaining submissions

# ## INTRODUCTION
# First, we will read the data and remove the header from the column.
# 

# In[ ]:


# Read the data

import csv

file = open('../input/hacker-news-posts/hacker_news.csv')
hn = list(csv.reader(file))
print(hn[:5])


# ## Deleting the header column from the list of lists

# In[ ]:


# Removing header

header = hn[0]
hn = hn[1:]
print(header)
print('\n', hn[:5])


# We see that the data set contains following headers:
# 
# | Header | Description |
# | :---: | :---: |
# | id | The unique identifier from Hacker News for the post |
# | title | The title of the post |
# | url | The URL that the posts links to, if it the post has a URL |
# | num_points | The number of points the post acquired, calculated as the total |
# | num_comments | The number of comments that were made on the post |
# | author | The username of the person who submitted the post |
# | created_at | The data and time at which the post was submitted |
# 
# Let's start by exploring number of comments on each type of posts.

# ## Extracting Ask Hn and Show HN Posts
# First, we'll identify the posts begining either with `Ask HN` or `Show HN` and separate the data for those types of posts into different lists. Separating the data will make it easier to analyse going further.

# In[ ]:


# Creating empty list

ask_posts = [] # For Ask HN posts
show_posts = [] # For Show HN posts
other_posts = [] # For other posts

# Creating a loop over the data to separate the posts

for post in hn:
    title = post[1] # Since title is second column
    title = title.lower() # Converting into lower-case
    if title.startswith('ask hn') is True:
        ask_posts.append(post)
    elif title.startswith('show hn') is True:
        show_posts.append(post)
    else:
        other_posts.append(post)
        
# Checking the number of posts

print('Number of Ask HN posts: ', len(ask_posts))
print('Number of Show HN posts: ', len(show_posts))
print('Number of other posts: ', len(other_posts))


# ## Calculating the Average Number of Comments for Ask HH and Show HN Posts
# We'll use the separated list of posts to calculate the average number of comments for each type of post receives.

# In[ ]:


# Calculating first for Ask HN posts

total_ask_comments = 0

for posts in ask_posts:
    num_comments = int(posts[4])
    total_ask_comments += num_comments
    
avg_ask_comments = total_ask_comments / len(ask_posts)

print('Average Number of Comments on Ask posts: ', avg_ask_comments)


# In[ ]:


# Calculating for Show HN posts

total_show_comments = 0

for posts in show_posts:
    num_comments = int(posts[4])
    total_show_comments += num_comments
    
avg_show_comments = total_show_comments / len(show_posts)

print('Average Number of Comments on Show posts: ', avg_show_comments)


# On **average**, **ask posts** in our sample receive **approximately 14 comments**, whereas **show posts** receive **approximately 10 comments**.
# 
# Since ask posts are more likely to receive comments, we'll focus our remaining analysis just on these posts.

# Next, we'll determine if we can maximise the amount of comments an ask posts created at a certain time by:
# 1. First we will find out the amount of ask post created in each hour of the day, along with the number of comments receieved. 
# 2. Later, we'll calculate average number of comments ask posts receive by hour created.
# 
# ## 1. Finding the Amount of Ask Posts and Comments by Hours Created

# In[ ]:


# Calculating the amount of ask posts created in each hour of the day, along with the number of comments received.

import datetime as dt

result_list = [] # To store the results

for post in ask_posts:
    created_at = post[6]
    num_comments = int(post[4])
    result_list.append([created_at, num_comments])
    
counts_by_hour = {}
comments_by_hour = {}

for each_row in result_list:
    date = each_row[0]
    comment = each_row[1]
    date = dt.datetime.strptime(date, "%m/%d/%Y %H:%M")
    time = date.strftime("%H")
    if time not in counts_by_hour:
        counts_by_hour[time] = 1
        comments_by_hour[time] = comment
    else:
        counts_by_hour[time] += 1
        comments_by_hour[time] += comment
        
comments_by_hour


# ## 2. Calculating the average number of Comments for Ask HN Posts by Hour

# In[ ]:


avg_by_hour = []

for hr in comments_by_hour:
    avg_by_hour.append([hr, round(comments_by_hour[hr] / counts_by_hour[hr],3)])

avg_by_hour


# ## Sorting and Printing Values from a List of Lists

# In[ ]:


swap_avg_by_hour = []

for hr in avg_by_hour:
    swap_avg_by_hour.append([hr[1],hr[0]])

swap_avg_by_hour


# In[ ]:


sorted_swap = sorted(swap_avg_by_hour, reverse=True)

print("Top 5 Hours for Ask Posts Comments")

for avg, hr in sorted_swap[:5]:
    print("{}: {:.2f} average comments per post".format(
        dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg)
         )


# The hour that receives the most comments per post on average is 15:00, with an average of 38.59 comments per post. There's about a 60% increase in the number of comments between the hours with the highest and second highest average number of comments.
# 
# According to the data set [documentation](https://www.kaggle.com/hacker-news/hacker-news-posts/home), the timezone used is Eastern Time in the US. So, we could also write 15:00 as 3:00 pm est.
# 
# # Conclusion
# In this project, we analyzed ask posts and show posts to determine which type of post and time receive the most comments on average. Based on our analysis, to maximize the amount of comments a post receives, we'd recommend the post be categorized as ask post and created between 15:00 and 16:00 (3:00 pm est - 4:00 pm est).
# 
# However, it should be noted that the data set we analyzed excluded posts without any comments. Given that, it's more accurate to say that of the posts that received comments, ask posts received more comments on average and ask posts created between 15:00 and 16:00 (3:00 pm est - 4:00 pm est) received the most comments on average.
# 
# **Credits**: Learnt from DataQuest.io website about this project.
# This is my practice project and I don't intend to mis-lead anyone saying that I copied someone else's project. I intend to share what I learnt and will work on more projects once I complete entire course via DataQuest platform.
