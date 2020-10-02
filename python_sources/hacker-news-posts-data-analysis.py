#!/usr/bin/env python
# coding: utf-8

# # **Analysing the dataset of Hacker News technology site submissions**  

# This work includes:
# * Analysing the posts with title beginning with *ASK HN* and *SHOW HN*.
# * Compare them to find out which post receive more comments on average
# * Find out the time of posts created having more number of comments on average
# 
# *ASK HN* - Users submit ASK HN posts to ask the Hacker News Community a specific question.
# *SHOW HN* - Users submit SHOW HN posts to show the Hacker News Community a project, product, or just generally something interesting.

# In[ ]:


#Reading all the required libraries
from csv import reader
import datetime as dt


# In[ ]:


opened_file = open('hacker_news.csv')
read_file = reader(opened_file)
hn = list(read_file)
hn[0:5]


# In[ ]:


headers = hn[0]


# In[ ]:


hn = hn[1:]


# In[ ]:


headers


# In[ ]:


hn[0:5]


# In[ ]:


ask_posts = []
show_posts = []
other_posts = []


# In[ ]:


for row in hn:
    title = row[1]
    title = title.lower()
    if title.startswith('ask hn'):
        ask_posts.append(row)
    elif title.startswith('show hn'):
        show_posts.append(row)
    else:
        other_posts.append(row)


# In[ ]:


print(len(ask_posts))
print(len(show_posts))
print(len(other_posts))


# In[ ]:


ask_posts[0:5]


# In[ ]:


show_posts[0:5]


# In[ ]:


total_ask_comments = 0
for row in ask_posts:
    comments = int(row[4])
    total_ask_comments += comments
    
avg_ask_comments = round(total_ask_comments/len(ask_posts))
print(avg_ask_comments)

total_show_comments = 0
for row in show_posts:
    comments = int(row[4])
    total_show_comments += comments
    
avg_show_comments = round(total_show_comments/len(show_posts))
print(avg_show_comments)


# From above result we see that the ask posts receive more comments than the show posts.
# And since ask posts are more likely to receive comments, we will focus our remaining analysis just on these posts.

# In[ ]:


result_list = []

for post in ask_posts:
    result_list.append(
        [post[6], int(post[4])]
    )

print(result_list)


# In[ ]:


res


# In[ ]:


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

print(counts_by_hour)
print(comments_by_hour)
    


# In[ ]:


avg_by_hour = []

for row in comments_by_hour:
    avg_by_hour.append([row,comments_by_hour[row]/counts_by_hour[row]])
    
avg_by_hour


# In[ ]:


len(avg_by_hour)


# In[ ]:


swap_avg_by_hours = []

for row in avg_by_hour:
    swap_avg_by_hours.append([row[1],row[0]])
    
print(swap_avg_by_hours)


# In[ ]:


sorted_swap = sorted(swap_avg_by_hours,reverse=True)


# In[ ]:


sorted_swap


# **Top 5 hours for Ask posts Comments**

# In[ ]:


sorted_swap[:5]


# In[ ]:


for avg,hr in sorted_swap[:5]:
    print("{}: {:.2f} avearge comments per post".format(dt.datetime.strptime(hr,"%H").strftime("%H:%M"),avg))
    
    


# The hour that receives most comments per post is 15:00, with an average of 38.59 comments per post.   According to the data set documentation, the timezone used is Eastern Time in the US. So, we could also write 15:00 as 3:00 pm est.

# # Conclusion

# In this notebook, analysis of ask posts and show posts has been carried out to determine which type of post and time receive maximum number of comments on average. From the results, it is seen that, to maximise the number of comments for the post, the post has to be a ask post type and should be posted between 15:00 to 16:00 (3 pm to 4pm est).
# 
# Note: The dataset used for this analysis excluded the posts without comments, so - of the posts that received comments , ask posts received more comments on average and the same created between 15:00 and 16:00 received maximum number of comments.

# In[ ]:




