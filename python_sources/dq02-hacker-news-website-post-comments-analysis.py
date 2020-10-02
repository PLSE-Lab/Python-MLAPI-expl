#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Exploring posts from a website

# Dataset:
# * Target: Hacker News https://news.ycombinator.com/
# * Dataset source: https://www.kaggle.com/hacker-news/hacker-news-posts

# Dataset columns:
# 
# * `id`: The unique identifier from Hacker News for the post
# * `title`: The title of the post
# * `url`: The URL that the posts links to, if it the post has a URL
# * `num_points`: The number of points the post acquired, calculated as the total number of upvotes minus the total number of downvotes
# * `num_comments`: The number of comments that were made on the post
# * `author`: The username of the person who submitted the post
# * `created_at`: The date and time at which the post was submitted
# 

# Goals:
# 
# * Do Ask HN or Show HN receive more comments on average?
# * Do posts created at a certain time receive more comments on average?
# 

# In[ ]:


from csv import reader
file='/kaggle/input/sample-hacker-news/hacker_news.csv'
hn=list(reader(open(file)))[1:]
headers=list(reader(open(file)))[0]
#hn[:5]


# In[ ]:


# Headers column
headers


# ## Slicing the dataset into categories

# In[ ]:


ask_posts=[]
show_posts=[]
other_posts=[]

for i in hn:
    title=i[1].lower()
    if title.startswith('ask hn'):
        ask_posts.append(i)
    elif title.startswith('show hn'):
        show_posts.append(i)
    else:
        other_posts.append(i)


# In[ ]:


print ('Ask HN posts count: \t',len(ask_posts))
print ('Show HN posts count: \t',len(show_posts))
print ('Other posts count: \t',len(other_posts))
print("_"*30)
print ('Total posts count: \t',len(hn))


# ## Comments by category
# Counting number of comments for each category

# In[ ]:


total_number_of_comments=0
for i in hn:
    comments_for_a_post =int(i[4])
    total_number_of_comments+=comments_for_a_post
    

def comments_counter(list):
    comments_total_for_category=0
    for i in list:
        comments_for_a_post =int(i[4])
        comments_total_for_category+=comments_for_a_post
    comments_percentage=round(comments_total_for_category/total_number_of_comments*100)
    avg_for_a_post=round(comments_total_for_category/len(list))
    return comments_total_for_category,comments_percentage,avg_for_a_post

ask_cnt=comments_counter(ask_posts)
show_cnt=comments_counter(show_posts)
other_cnt=comments_counter(other_posts)


# In[ ]:


print('Comments for Ask posts: {total}({percent}%), with average {post} for a post.'.format(post=ask_cnt[2],total=ask_cnt[0],percent=ask_cnt[1]))
print('Comments for Show posts: {total}({percent}%), with average {post} for a post.'.format(post=show_cnt[2],total=show_cnt[0],percent=show_cnt[1]))
print('Comments for Other posts: {total}({percent}%), with average {post} for a post.'.format(post=other_cnt[2],total=other_cnt[0],percent=other_cnt[1]))


# As we see from the cell above, Ask posts are significantly more popular than Show posts, but indeed Other posts are the most commented (2 times more popular than Ask). Let's dive deeper for Ask comments analysis.

# ## Number of comments by time posted

# In[ ]:


import datetime as dt
z=ask_posts[5][6]
dt_template="%m/%d/%Y %H:%M"


# In[ ]:


dt.datetime.strptime(z,dt_template)


# In[ ]:


dt_template="%m/%d/%Y %H:%M"

n_comments_by_hour=[]

for i in ask_posts:
    date_row=i[6]
    date_conv=dt.datetime.strptime(date_row,dt_template)
    hour=date_conv.hour
    n_comments=int(i[4])
    n_comments_by_hour.append((hour,n_comments))
    
    
n_comments_by_hour[:7]


# In[ ]:


comments_by_hour={}
counts_by_hour={}

for i in n_comments_by_hour:
    hour=i[0]
    comments=i[1]
    if hour not in counts_by_hour:
        counts_by_hour[hour]=1
        comments_by_hour[hour]=comments
    else:
        counts_by_hour[hour]+=1
        comments_by_hour[hour]+=comments


# Calculating average number of comments by hour of post publication

# In[ ]:


avg_by_hour={}

for i in range(24):
    avg_by_hour[i]=comments_by_hour[i]/counts_by_hour[i]

import operator
sorted_avg_by_hour = sorted(avg_by_hour.items(), key=operator.itemgetter(1),reverse=True)


print ("Hours to publish a post with most number of comments (top five):")
for i in sorted_avg_by_hour[:5]:
    print ("Hour: {hour}:00, number of comments on average: {comments:.2f}".format(hour=i[0],comments=i[1]))

