#!/usr/bin/env python
# coding: utf-8

# Exploring Hacker News Posts: Popularity of Show vs. Ask Posts

# **Introduction:** Hacker News is a site started by the startup incubator Y Combinator, where user-submitted stories (known as "posts") are voted and commented upon, similar to Reddit. Hacker News is extremely popular in technology and startup circles, and posts that make it to the top of Hacker News' listings can get hundreds of thousands of visitors as a result. 
# 
# We are interested in posts whose titles begin with either "Ask HN" or "Show HN". Users submit Ask HN posts to the Hacker News Community to ask a specific question. Show HN posts include posts where the user displays a community project, product, or other interesting artifact.
# 
# In this data analysis, we will compare these two types of posts to determine the following:
# 
# 1. Do Ask HN or Show HN posts receive more comments on average.
# 2. Do posts created at a certain time receive more comments on average.
# 
# Please note the data set we're working with was reduced from almost 300,000 rows to approximately 20,000 rows by removing all submissions that did not receive any comments, and then randomly sampling from the remaining submissions.

# In[ ]:


#opening the Hacker News Dataset
import csv

file = open('../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv')
hn = list(csv.reader(file))
print(hn[:5])


# **Removing Headers from the Printed List

# In[ ]:


headers = hn[0]
hn = hn[1:]
print(headers)
print(hn[:5])


# The above data set contains the following:
#     1. Post ID
#     2. Title of posts
#     3. Post URL
#     4. Number of points on post
#     5. Number of comments on post
#     6. Author of post
#     7. The date the post was created. 
# Next we will explore the number of comments for each type of post.

# **Extracting ASK Hacker News and SHOW Hacker News Posts:**
# Next we will identify posts that behin with either ASK HN or SHOW HN and separate the data for these types of posts into different lists.

# In[ ]:


#Separating post data into different lists by Title
ask_posts = []
show_posts = []
other_posts = []

for post in hn:
    title = post[1]
    if title.lower().startswith('ask hn'):
        ask_posts.append(post)
    elif title.lower().startswith("show hn"):
        show_posts.append(post)
    else:
        other_posts.append(post)
        
print(len(ask_posts))
print(len(show_posts))
print(len(other_posts))


# **Calculating the Avg. Number of Comments on ASK Hacker News and SHOW Hacker News Posts**

# In[ ]:


#Calculating the average number of comments 'Ask HN' posts receive.
total_ask_comments = 0

for post in ask_posts:
    total_ask_comments += int(post[4])
    
avg_ask_comments = total_ask_comments / len(ask_posts)
print(avg_ask_comments)


# In[ ]:


#Calculating the average number of comments 'Show HN' posts receive.
total_show_comments = 0

for post in show_posts:
    total_show_comments += int(post[4])
    
avg_show_comments = total_show_comments / len(show_posts)
print(avg_show_comments)


# Based on conclusions drawns from analysis above - ask posts receive approximately 14 comments while show posts receive approximately 10.
# 
# Because Ask posts receive more comments, the remaining analysis will focus on these posts.

# **Analyzing the Amount of Comments on Ask Posts Based on Hour Created**
# 
# Below we will determine if ask posts can maximize the amount of comments received based on the time it was created. First, we will discover the amounts of ask posts created per hour of the day. Then we will calculate the average amount of comments the posts created at each hour receive.

# In[ ]:


#Calculating the amount of ask posts created during each hour of the day and the number of comments received.
import datetime as dt

result_list = []

for post in ask_posts:
    result_list.append(
        [post[6], int(post[4])]
    )
    
comments_by_hour = {}
counts_by_hour = {}
date_format = "%m/%d/%Y %H:%M"

for each_row in result_list:
    date = each_row[0]
    comment = each_row[1]
    time = dt.datetime.strptime(date, date_format).strftime("%H")
    if time in counts_by_hour:
        comments_by_hour[time] += comment
        counts_by_hour[time] += 1
    else:
        comments_by_hour[time] = comment
        counts_by_hour[time] = 1

comments_by_hour


# **Calculating the Acg. Number of Comments for ASK Hacker News Posts by Hour**

# In[ ]:


# Calculating the average amount of comments `Ask HN` posts created at each hour of the day receive.
avg_by_hour = []

for hr in comments_by_hour:
    avg_by_hour.append([hr, comments_by_hour[hr] / counts_by_hour[hr]])

avg_by_hour


# **Sorting and Printing Values from the List Above**

# In[ ]:


swap_avg_by_hour = []

for row in avg_by_hour:
    swap_avg_by_hour.append([row[1], row[0]])
    
print(swap_avg_by_hour)

sorted_swap = sorted(swap_avg_by_hour, reverse=True)

sorted_swap


# In[ ]:


# Sorting the values and printing the the 5 hours with the highest average comments.

print("Top 5 Hours for 'Ask HN' Comments")
for avg, hr in sorted_swap[:5]:
    print(
        "{}: {:.2f} average comments per post".format(
            dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg
        )
    )


# From above, we can see the hour that receives the most comments per post on average is 15:00 with an average of 38.59 comments per post. Based on documention included with dataset, we can conclude the most amount of comments on Ask Hacker News Posts are on posts created at 3:00pm est.

# **Conclusion**
# Through the analysis of Hacker News Posts, we were able to determine which type of posts based on time created receive the most amount of comments on average. Our analysis shows that in order to maximize the amount of comments a post receives, we'd recommend the post be categorized as an 'Ask Post' and created between 3 and 4pm est.
