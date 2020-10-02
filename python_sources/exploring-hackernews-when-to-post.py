#!/usr/bin/env python
# coding: utf-8

# # Exploring HackerNews: a simple data analysis project.
# 
# ## 1. Introduction
# 
# Hacker News is a site started by the startup incubator Y Combinator, where user-submitted stories (known as "posts") are voted and commented upon, similar to reddit. Hacker News is extremely popular in technology and startup circles, and posts that make it to the top of Hacker News' listings can get hundreds of thousands of visitors as a result.
# 
# The original dataset can be found [here](https://www.kaggle.com/hacker-news/hacker-news-posts), but note that it has been reduced from almost 300,000 rows to approximately 20,000 rows by removing all submissions that did not receive any comments, and then randomly sampling from the remaining submissions. Below are descriptions of the columns:
# 
# * id: The unique identifier from Hacker News for the post
# * title: The title of the post
# * url: The URL that the posts links to, if it the post has a URL
# * num_points: The number of points the post acquired, calculated as the total number of upvotes minus the total number of downvotes
# * num_comments: The number of comments that were made on the post
# * author: The username of the person who submitted the post
# * created_at: The date and time at which the post was submitted

# In[ ]:


#opening the hacker news dataset
from csv import reader
opened_file = open('../input/hacker-news/hacker_news.csv')
read_file = reader(opened_file)
hn = list(read_file)
print(hn[:5])


# In[ ]:


#cleaning the data by removing the header row
headers = hn[0]
hn = hn[1:]
print("headers: ", headers, "\n")
print("First Five Rows: ", hn[:5])


# ## 2. Objectives:
# 
# For this analysis, We're specifically interested in posts whose titles begin with either **Ask HN** or **Show HN**. Users often submit Ask HN posts to ask the Hacker News community a specific question. Below are a couple examples:
# 
# `Ask HN: How to improve my personal website?
# Ask HN: Am I the only one outraged by Twitter shutting down share counts?
# Ask HN: Aby recent changes to CSS that broke mobile?`
# 
# Likewise, users submit Show HN posts to show the Hacker News community a project, product, or just generally something interesting. Below are a couple of examples:
# 
# `Show HN: Wio Link  ESP8266 Based Web of Things Hardware Development Platform'
# Show HN: Something pointless I made
# Show HN: Shanhu.io, a programming playground powered by e8vm`
# 
# We'll compare these two types of posts to determine the following:
# 
# * Do Ask HN or Show HN receive more comments on average?
# * Do posts created at a certain time receive more comments on average?

# In[ ]:


#categorizing posts based on Ask and Show HN
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
        other_posts.append(row)
        
print("number of posts asking HN = ", len(ask_posts), "\n",
      "number of posts showing HN = ", len(show_posts), "\n",
      "number of other posts = ", len(other_posts), "\n",
     )


# ## 3.Comparing the averages
# Above we separated the "ask posts" and the "show posts" into two list of lists named `ask_posts` and `show_posts`.
# Next, let's determine if ask posts or show posts receive more comments on average.

# In[ ]:


# Finding the total number of comments to 'ask posts'
total_ask_comments = 0
for post in ask_posts:
    num_comments = int(post[4])
    total_ask_comments += num_comments

#calculating the avergae number of comments on 'ask posts'
avg_ask_comments = total_ask_comments / len(ask_posts)
print("The average number of comments on ask posts is : ", avg_ask_comments)

# Finding the total number of comments to 'show posts'
total_show_comments = 0
for post in show_posts:
    num_comments = int(post[4])
    total_show_comments += num_comments

#calculating the avergae number of comments on 'show posts'
avg_show_comments = total_show_comments / len(show_posts)
print("The average number of comments on show posts is : ", avg_show_comments)


# ## 4. Digging deeper into 'Ask Posts'
# We've determined above that, on average, ask posts receive more comments than show posts. Since ask posts are more likely to receive comments, we'll focus our remaining analysis just on these posts.
# 
# Next, we'll determine if ask posts created at a certain time are more likely to attract comments. We'll use the following steps to perform this analysis:
# 
# 1. Calculate the amount of ask posts created in each hour of the day, along with the number of comments received.
# 2. Calculate the average number of comments ask posts receive by hour created.

# In[ ]:


import datetime as dt
result_list = []
for post in ask_posts:
    created_at = post[6]
    num_comments = int(post[4])
    mixed_list = [created_at, num_comments]
    result_list.append(mixed_list)
    
counts_by_hour = {}
comments_by_hour ={}
for row in result_list:
    date = row[0]
    date_dt = dt.datetime.strptime(date, "%m/%d/%Y %H:%M")
    hour = date_dt.strftime("%H")
    if hour not in counts_by_hour:
        counts_by_hour[hour] = 1
        comments_by_hour[hour] = row[1]
    else:
        counts_by_hour[hour] += 1
        comments_by_hour[hour] += row[1]
        
print("counts by hour : ", counts_by_hour, "\n", "\n",
     "comments by hour : ", comments_by_hour)
        
    


# Above, we created two dictionaries:
# 
# * `counts_by_hour`: contains the number of ask posts created during each hour of the day.
# * `comments_by_hour`: contains the corresponding number of comments ask posts created at each hour received.
# 
# Next, we'll use these two dictionaries to calculate the average number of comments for posts created during each hour of the day.

# In[ ]:


#calculating the average number of comments per post per hour
avg_by_hour = []
for hour in comments_by_hour:
    avg_by_hour.append([hour, (comments_by_hour[hour] / counts_by_hour[hour])])
print("The average number of comments per post for each hour of the day is: \n \n ", 
     avg_by_hour)    


# Although we now have the results we need, this format makes it hard to identify the hours with the highest values. Let's finish by sorting the list of lists and printing the five highest values in a format that's easier to read.

# In[ ]:


#swap the columns
swap_avg_by_hour = []
for hour in avg_by_hour:
    swap_avg_by_hour.append([hour[1], hour[0]])

print(swap_avg_by_hour)

#sort the results
sorted_swap = sorted(swap_avg_by_hour, reverse = True)

print('\n Top 5 Hours for Ask Posts Comments: \n')
for avg, hour in sorted_swap[:5]:
    hour_dt = hour
    hour_ob = dt.datetime.strptime(hour_dt, "%H")
    hour_sf = hour_ob.strftime("%H:%M")
    avg_comments = "{}: {:.2f} average comments per post."
    print(avg_comments.format(hour_sf, avg))


# ## Conclusion: So when should you create an 'ask hn' post on HN?
# 
# It is apparent from our analysis that, on average 3pm(Eastern Time in the US) is the ideal time to create an 'ask hn ' post in order to increase the likelihood of people commenting on it. 
# 
# Though it is difficult to acertain all the factors playing in to this time slot being the most effective at eliciting comments, we can perhaps speculate at some causes:
#  * The morning hours are proably bustling with activity at most workplaces and thus slow use net usage
#  * Similarly the lunch and after lunch period are thus more relaxed and present a more comfortable time to explore the site at ease.
#  
#  The dataset is ripe for more investigation and perhaps in future I will continue to explore it.
