#!/usr/bin/env python
# coding: utf-8

# # Data Analysis on Hacker news posts
# 
# Our goal is to perform analysis on the data to find out at which time number of comments are most. So, that our post can be among the top posts.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# First, we will start with importing the data. We want list of lists as each row of the data without the header column.

# In[ ]:


from csv import reader 
data=open("../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv")
hn = reader(data)
hn = list(hn)
print(hn[0:5])


# In this step, we will create three empty lists called ask_posts, show_posts, and other_posts. And collect the data as their name specifies. This data resides in the title column of the data with string starting as 'ask hn' and 'show hn'.

# In[ ]:


ask_posts = []
show_posts = []
other_posts = []
for row in hn:
    title = row[1]
    if title.lower().startswith('ask hn'):
        ask_posts.append(row)
    elif title.lower().startswith('show hn'):
        show_posts.append(row)
    else:
        other_posts.append(title)
print('Number of ask posts: -',len(ask_posts))
print('Number of show posts: -',len(show_posts))
print('Number of other posts: -',len(other_posts))


# Next, let's determine if ask posts or show posts receive more comments on average.

# In[ ]:


total_ask_comments = 0
for row in ask_posts:
    total_ask_comments += int(row[4])
avg_ask_comments = total_ask_comments/len(ask_posts)
print('Average comments in ask posts: -',avg_ask_comments)

total_show_comments = 0
for row in show_posts:
    total_show_comments += int (row[4])
avg_show_comments = total_show_comments/len(show_posts)
print('Average comments in show posts: -',avg_show_comments)


# By above analysis, We decided to make ask posts as they are getting nearly 2.5x more comments.
# Now let's calculate the amount of ask posts created per hour, along with the total amount of comments.

# In[ ]:


import datetime as dt

result_list =[]
for row in ask_posts:
    result_list.append([row[6],int(row[4])])
                       
counts_by_hour = {}
comments_by_hour ={}

for row in result_list:
    date_created_at = dt.datetime.strptime(row[0],'%m/%d/%Y %H:%M')
    hour = dt.datetime.strftime(date_created_at,'%H')
    if hour not in counts_by_hour:
        counts_by_hour[hour] = 1
        comments_by_hour[hour] = row[1]
    else:
        counts_by_hour[hour] +=1
        comments_by_hour[hour] += row[1]


# In[ ]:


print(counts_by_hour)
print(comments_by_hour)


# Now we will calculate the average number of comments per post for posts created during each hour of the day.

# In[ ]:


avg_by_hour =[]
for hour in counts_by_hour:
    for comments in comments_by_hour:
        if hour==comments:
            avg_by_hour.append([hour,(comments_by_hour[hour])/counts_by_hour[hour]])
print(avg_by_hour)


# In[ ]:


swap_avg_by_hour = []
for row in avg_by_hour:
    swap_avg_by_hour.append([row[1],row[0]])
print(swap_avg_by_hour)


# In[ ]:


sorted_swap = sorted(swap_avg_by_hour,reverse=True)
print(sorted_swap[0:5])


# In[ ]:


for row in sorted_swap:
    hour = dt.datetime.strptime(row[1],'%H')
    hour = dt.datetime.strftime(hour,'%H:00')
    string = '{}: {:.2f} average comments per post'.format(hour,row[0])
    print(string)


# # Result
# 
# After this analysis, We concluded that posting ask post on hacker news at 15:00 (1:00 pm) will increase chances to maximum to make the post on the top.
