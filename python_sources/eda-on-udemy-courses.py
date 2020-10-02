#!/usr/bin/env python
# coding: utf-8

# This is my first notebook submission on Kaggle. So, please feel free to put in some feedback in the comments section.

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import plotly 
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv('../input/udemy-courses/udemy_courses.csv')


# # Data cleaning

# In[ ]:


#Creating a new column which consists of year only
date=dataset.published_timestamp.copy()

dataset['Published_year']=0
dataset=dataset.drop(index=2066)
dataset=dataset.reset_index(drop=True)
dataset.Published_year=dataset.Published_year.astype(int)
for i in range(len(date)):
    date[i]=date[i].split('-')
    dataset['Published_year'][i]=date[i][0]


# In[ ]:


#Creating a new column for title length
dataset['title_length']=[len(i) for i in dataset.course_title]


# In[ ]:


dataset=dataset.drop(columns=['course_id','url'])


# In[ ]:


#Making the is_paid column binary
dataset.is_paid=dataset.is_paid.map({'True':1,'False':0})


# In[ ]:


#Seperating the hours from content duration
"""for i in range(len(dataset.content_duration)):
    dataset.content_duration[i]=dataset.content_duration[i].split(' ')[0]
dataset.content_duration=dataset.content_duration.astype(float)"""


# In[ ]:


#Filling the missing values in is_paid
dataset=dataset.replace("Free",0)
dataset.price=dataset.price.astype(int)
check=dataset.is_paid.isna()
for i in range(len(dataset)):
    if(check[i]==True and dataset.price[i]!=0):
        dataset.is_paid[i]=0
    elif(check[i]==True and dataset.price[i]==0):
        dataset.is_paid[i]=1 


# In[ ]:


dataset.head()


# # Questions to ask:
#     In which year was the most courses published-done
#     Which level courses were most present-done
#     Does duration and num of lectures decide number of subscribers
#     Does the number of reviews affect the the number of subscribers
#     Which subject did most courses cover-done
#     Does the price affect the decisions to take a course-done
#     Title length does it matter
#     

# Q1.In which year was the most courses published?

# In[ ]:


sns.countplot(data=dataset,x='Published_year')


# In[ ]:


dataset.Published_year=dataset.Published_year.astype(int)


# We see that 2016 was the time when there were a lot of new courses introduced.

# Q2. Which level courses were most present?

# In[ ]:


sns.countplot(data=dataset,x='subject')


# In[ ]:


dataset.subject.value_counts()


# It can be seen that Web Development and Business Finance are the most present courses, to be precise Web development and Business Finance differ by 1. So we look at another aspect which would be the number of audience present for each of these subject. Which leads us to the question:
# 
# Q2. Which level courses were most present?

# In[ ]:


#unique_subject=list(dataset.subject.unique())
pd.pivot_table(dataset,values='num_subscribers',index='subject',aggfunc=np.sum)


# In[ ]:


unique_subject=list(dataset.subject.unique())

val=[1870747,1063148,846689,7980572]
trace=go.Pie(labels=unique_subject,values=val,
            hoverinfo='label+value',textinfo='percent',
            textfont=dict(size=25),
            marker=dict(line=dict(width=1)),
            title="Subscribers Vs Subject")
iplot([trace])


# Web development is the subject which has the most number of subscribers and to add on it has the most number of courses. Now, we need to check which level has the highest availability and the subscriber count. Surprising to see the surplus of Business Finance courses with respect to the audience

# In[ ]:


y=dataset.level.value_counts().rename_axis('unique_values').reset_index(name='counts')
y.head()
fig = go.Figure(data=go.Scatter(x=y.unique_values, y=y.counts))
fig.show()


# Most available level is All levels. However the preceding level which is Beginner makes sense as people use platforms such as udemy to gain a basic gist of a subject connected to their interest or profession . It is no surprise that there is the least choice for expert level because you hardly see them.
# Now, we see the distribution of each course with respect to the levels.

# In[ ]:


unique_subject=list(dataset.subject.unique())
l1=[]
for i in unique_subject:
    l2=[]
    for j in dataset.level.unique():
        sum1=0
        for k in range(len(dataset.num_subscribers)):
            if(dataset.level[k]==j and dataset.subject[k]==i):
                sum1+=1
        l2.append(sum1)
    l1.append(l2)


# In[ ]:


fig, axes = plt.subplots(2, 2,figsize=(14,8))

ax = sns.barplot(x=dataset.level.unique(), y=l1[0],ax=axes[0, 0]).set_title(unique_subject[0])
ax = sns.barplot(x=dataset.level.unique(), y=l1[1],ax=axes[0, 1]).set_title(unique_subject[1])
ax = sns.barplot(x=dataset.level.unique(), y=l1[2],ax=axes[1, 0]).set_title(unique_subject[2])
ax = sns.barplot(x=dataset.level.unique(), y=l1[3],ax=axes[1, 1]).set_title(unique_subject[3])


# The trend is similar to the one we saw previously, except for Musical Instruments.
# Next we see whether the duration and num of lectures decide number of subscribers?We do this by checking the correlation between the two using a heatmap. If a high magnitude of correlation is there, we can say that they do affect the decision

# In[ ]:


temp_df=pd.concat([dataset.num_subscribers,dataset.num_reviews,dataset.num_lectures,dataset.content_duration],axis=1)


# In[ ]:


sns.heatmap(temp_df.corr(),cmap="Greens")


# The number of reviews has a significant impact while the duration doesnt on the number of subscribers. This is evident as Udemy 
# gives badges to courses with the highest rating in that subject. The correlation between number of lectures and duration is evident as well.

# In[ ]:


pay_df=pd.concat([dataset.is_paid,dataset.price,dataset.num_subscribers],axis=1)
sns.heatmap(pay_df.corr(),cmap="Blues")


# Quite surprising that is_paid has some correlation and price has very little correlation. It seems that
# price plays a little or no role in the purchasing decision. Let's check this out with the values of 
# number of subscribers for any given price.

# In[ ]:


d=pd.pivot_table(pay_df,values='num_subscribers',index='price',aggfunc=np.sum)
d=d.reset_index()
d.price=d.price.astype(int)
d=d.sort_values(by='price')


# In[ ]:


d['num_ranks']=d['num_subscribers'].rank()


# In[ ]:


sizes=[]
colours=[]
for i in d.num_ranks:
    sizes.append(i*2)
    colours.append(120+(2*i))


# In[ ]:


x=d.iloc[:, 0].values
y=d.iloc[:, 1].values


# In[ ]:


fig = go.Figure(data=[go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        color=colours,
        size=sizes,
        showscale=True
        )
    )])
fig.show()

As mentioned above, the price plays very litle role in the purchase decision, although free courses have the highest number of audience the next highest course in terms of audience is in fact the highest paid course.Probably, these courses are focused much more on the expert side or are highly rated. Lets check it out.
# In[ ]:


extension=dataset.loc[dataset['price']==200]
extension=extension.reset_index(drop=True)


# In[ ]:


subject_extension=list(extension.subject.unique())
l1=[]
for i in subject_extension:
    l2=[]
    for j in extension.level.unique():
        sum1=0
        for k in range(len(extension.num_subscribers)):
            if(extension.level[k]==j and extension.subject[k]==i):
                sum1+=1
        l2.append(sum1)
    l1.append(l2)


# In[ ]:





# In[ ]:


fig, axes = plt.subplots(2, 2,figsize=(14,8))

ax = sns.barplot(x=extension.level.unique(), y=l1[0],ax=axes[0, 0]).set_title(subject_extension[0])
ax = sns.barplot(x=extension.level.unique(), y=l1[1],ax=axes[0, 1]).set_title(subject_extension[1])
ax = sns.barplot(x=extension.level.unique(), y=l1[2],ax=axes[1, 0]).set_title(subject_extension[2])
ax = sns.barplot(x=extension.level.unique(), y=l1[3],ax=axes[1, 1]).set_title(subject_extension[3])


# Considering the price still there are no expert level courses, let's just see what is the median price for expert price and also lets look at the ratings of these $200 courses

# In[ ]:


expert=dataset.loc[dataset['level']=='Expert Level']
print(" Expert price median: " +str(expert.price.median()),"  Median number of reviews expert:" + str(expert.num_reviews.median()))
print("Overall dataset review number median: "+str(dataset.num_reviews.median()))
print("Expensive review number median: "+str(extension.num_reviews.median()))


# As it can be seen expensive courses tend to have a higher median when it comes to giving reiews. This
# probably be one of the reasons why people are ready to pay such amounts largely due to the number of 
# review pool. But one thing to consider is that these courses have a huge audience as well and that can 
# be an indicator as well

# Final question: Does the title length affect the number of subsribers?
# This may sound silly but probably a highly descriptive title may or may not bring about more subscribers.

# In[ ]:


title=dataset.iloc[:, [3,11]]
title=title.sort_values(by=['title_length'])
title
x_title=title.title_length.values
y_title=title.num_subscribers.values


# In[ ]:


fig = go.Figure(data=go.Scatter(x=x_title, y=y_title))
fig.show()


# We can say that longer titles do not yield more subscribers rather shorter titles yield more 
# subscribers
