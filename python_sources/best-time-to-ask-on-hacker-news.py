#!/usr/bin/env python
# coding: utf-8

# # Best Time to Ask on Hacker News
# 
# ### What is Hacker News?
# 
# "[Hacker News](https://news.ycombinator.com/) is a social news website focusing on computer science and entrepreneurship." - Wikipedia. You can find more information about HN [here](https://en.wikipedia.org/wiki/Hacker_News). 
# 
# ### What is project about?
# 
# There are few types of posts in HN. This project only focuses on Ask Hacker News and Show Hacker News, to see which one is more  popular type than the others; and what time users give comments the most.
# 
# * Ask HN: is where users submit a post to ask a specific question.
# * Show HN: is where users submit a post to show a project, product, or just something interesting.
# 
# ### What are my goals?
# 
# My goals are able to answer two questions:
# 
# 1. Which one is the most popular type?
# 2. What time is the best to submit a post?
# 
# ## I. Opening the Data:
# 
# I will use this [data set](https://www.kaggle.com/hacker-news/hacker-news-posts) to analyze. It includes approximately 270,000 posts year to Sep 26, 2016.
# 
# Let's open it:

# In[ ]:


from csv import reader

opened_file = open('../input/HN_posts_year_to_Sep_26_2016.csv')
read_file = reader(opened_file)
hn = list(read_file)
hn_header = hn[0]
hn_body = hn[1:]

print(hn_header)
print('\n')
print(hn_body[:5])


# ## II. Extracting Ask HN and Show HN Posts:
# 
# As I mentioned above, there are few types of posts in Hacker News. All I want is Ask HN and Show HN posts, so I will categorize them into three lists `ask_hn`, `show_hn` and `other_hn`. I can use string method `startswith` to extract titles which are begin with `Ask HN` or `Show HN`, but the method is case sensitive. So to speak, It will understand `Ask HN` and `ask hn` are two different things. That's why, I'll need to lower (or upper) all titles by using method `lower()` or `upper()`. 

# In[ ]:


ask_hn = []
show_hn = []
other_hn = []

for row in hn_body:
    title = row[1].lower()
    if title.startswith('ask hn'):
        ask_hn.append(row)
    elif title.startswith('show hn'):
        show_hn.append(row)
    else:
        other_hn.append(row)

print('Number of Ask HN posts: ', len(ask_hn))
print('Number of Show HN posts: ', len(show_hn))
print('Number of Other HN posts: ', len(other_hn))


# ## III. Calculating the Average Number of Comments for Ask HN and Show HN Posts:
# 
# We already seperated all posts into three lists. Show HN are greater than Ask HN posts. However, we cannot use those numbers to conclude Show HN is the most popular. To determine the popularity of something, we have to figure out how much people talk about it. Total number of comments would not enough to answer my first question because it couldn't tell me how much HN users discuss on a topic. That's why I need to find out the average number of comments of a post in each type.

# In[ ]:


total_ask_comment = 0
total_show_comment = 0

for row in ask_hn:
    num_comments = int(row[4])
    total_ask_comment += num_comments
    
for row in show_hn:
    num_comments = int(row[4])
    total_show_comment += num_comments
    
avg_ask_comment = total_ask_comment / len(ask_hn)
avg_show_comment = total_show_comment / len(show_hn)

print('Average number of comment in Ask HN: ', avg_ask_comment)
print('Average number of comment in Show HN: ', avg_show_comment)


# Now I can conclude that Ask HN is more popular, because users discuss twice times of Show HN on a post. Let's move on to second question: **What time is the best to submit a post**?
# 
# ## IV. Finding the Amount of Ask Posts and Comments by Hour Created:
# 
# When I said 'the best', it meant the golden time that users would answer my question the most. First of all, I need to find the total amount of comments and total amount of posts by hour in Ask HN.

# In[ ]:


import datetime as dt

comment_by_hour = {}
count_by_hour = {}

for row in ask_hn:
    created_at = row[6]
    num_comments = int(row[4])
    hour = dt.datetime.strptime(created_at, '%m/%d/%Y %H:%M').strftime('%H')
    if hour not in count_by_hour:
        comment_by_hour[hour] = num_comments
        count_by_hour[hour] = 1
    else:
        comment_by_hour[hour] += num_comments
        count_by_hour[hour] += 1

print('Total number of comments by hour:\n', comment_by_hour)


# ## V. Calculating the Average Number of Comments for Ask HN Posts by Hour:
# 
# Secondly, I'll calculate the average number of comments by hour:

# In[ ]:


avg_comment_hour = []

for key in comment_by_hour:
    avg_comment_hour.append([key, comment_by_hour[key] / count_by_hour[key]])
    
avg_display = []

for element in avg_comment_hour:
    hour = element[0]
    avg = element[1]
    avg_display.append([avg, hour])
    
avg_display = sorted(avg_display, reverse = True)

print('Top 5 Average Number of Comments by Hour:\n')
for element in avg_display[:5]:
    print('{}:00 : {:.2f} average comments per post'.format(element[1],element[0]))


# ## VI. Conclusion:
# 
# Ask HN is more popular than Show HN; and if you want to get as much as answers for your question, you should submit a post between 13:00 and 15:00 of a day.
# 
# It might a bit out of this project, but after done my analysis, I have two more questions for myself:
# 
# 1. Why do people like to answering questions than sharing stuff?
# 2. Why is time between 13:00 and 15:00 the best?
# 
# I did google about these questions, and I think there are few reasons behind:
# 
# - We're curious pieces. Questioning things around us, and finding an answer are our nature. That's make us different from other animal, gain us knowledge of this world.
# - A question is often short and straigh forward. It's true that nowadays most people don't like reading a long article. We prefer skimming through, find the info we need here and there.
# - It's kind to help other people.
# 
# For the last question, I found a similar question in [Stackoverflow blog](https://stackoverflow.blog/2009/01/06/the-best-time-to-ask-a-stack-overflow-question/). It happens that most popular time to post is between 15:00 and 22:00. In my opion, in the morning, people is busy at work, so they won't have time. Right after lunch, some people would like to take a nap, and they could still be sleepy at the begining of afternoon; or many tasks haven't done yet. But from 15:00, our day is nearly finish, it makes us feel more relax and we could do something else such as answer a question on Hacker News.

# *The purpose of this project is mainly to practice what I have learned from [dataquest.io](dataquest.io) - Python for Data Science: Intermediate course. Many techniques, contents in this project were guided by dataquest.io and the following [solution](https://github.com/dataquestio/solutions/blob/master/Mission356Solutions.ipynb).*
