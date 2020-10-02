#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
warnings.simplefilter(action='ignore')
sns.set(style="ticks", color_codes=True,palette='inferno_r')


# In[ ]:


udemy=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
udemy.head()


# In[ ]:


udemy.info()


# There are 12 features and 3678 obervations
# There are 6 quantitative features and 6 categorical feature
# There are no missing values

# In[ ]:


cat=udemy.select_dtypes(include=['object','bool']).columns
cont=udemy.select_dtypes(exclude=['object','bool']).columns
cont


# In[ ]:


udemy1=udemy.copy()
mask = udemy1.applymap(type) != bool
d = {True: 'TRUE', False: 'FALSE'}

udemy1 = udemy1.where(mask, udemy1.replace(d))


# 1.We're going to explore this dataset and gain new insights on the data 
# 
# 
# 2.We'll check if is_paid features is affects subscribers

# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(udemy.corr(),cmap='YlGnBu',annot=True,square=True)
plt.xticks(rotation=45)
plt.tight_layout()
#coolwarm YlGnBu 


# # Interpretation
# 1. Price and is_paid has some positive correleation
# 2. Number of subscribers and is_paid has some negative correleation
# 3. Price and number of lectures has positive correleation
# 4. Number of reviews and number of lectures has positive correleation
# 5. Content duration and number of reviews has positive ccorreleation
# 6. Number of lectures and content duration has very high positive correleation

# In[ ]:


g=sns.pairplot(udemy1,diag_kind='kde',hue='is_paid',palette='husl')


# # Interpretation
# 
# 1. In course_id  number of subscribers there are seems to be a little correleation. Course_ids with id aroung 5000 have more subscribers. 
# 
# 2. Most of the courses are free and most of the paid courses cost less than 100(units?) , and few of them cost above 200(units)
# 
# 3. Free courses have most number of subscribers
# 
# 4. Paid courses have more number of reviews
# 
# 5. Paid courses have more number of lectures 
# 
# 6. Paid courses have more content duration.  Paid courses which cost more than 200 has comparitively more content duration
# 
# 7. Most of the courses have around 20k subscribers there's one course that has around 250k subscribers [We'll explore them in other graphs]
# 
# 8. There are less number of reviews for most of the courses
# 
# 9. Most of the courses have around 125-150 lectures
# 
# 9. This is given. more the course duration, more the number of lectures

# In[ ]:


def col_types():
    print(cat)
    print(cont)
col_types()


# # Let's analyse categorical features

# In[ ]:


udemy.groupby('is_paid').is_paid.count()


# 1. There are very few free courses available in udemy.
# 2. There are 310 free courses and 3368 paid courses

# In[ ]:


udemy.groupby('level').is_paid.value_counts()


# In[ ]:


sns.countplot(x='level',hue='is_paid',data=udemy)
plt.xticks(rotation=8)
plt.show()


# 1. All levels means course packages that teaches from the beginning to expert level. 1807 out of 1929 of these courses are paid
# 2. There are very few free courses in Intermediate and Beginner levels [158/1270 for beginner level, 30/421 for intermediate level]
# 3. There are no free courses in expert level

# In[ ]:


udemy.groupby('subject').is_paid.value_counts()


# In[ ]:


sns.countplot(x='subject',hue='is_paid',data=udemy)
plt.xticks(rotation=8)
plt.show()


# 1. There are 1099 Business Finance courses and 1067 Web Development courses
# 2. 91% of Businnes finance courses are paid and 88% of the web development courses are paid
# 3. Though very less , there are comparitively more free courses in web development compared to other subjects

# In[ ]:


udemy.describe()


# # Level Feature

# In[ ]:


col_types()


# In[ ]:


sns.catplot(x='level',y='price',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)
#udemy[(udemy['price']==200) & (udemy['level']=='All Levels')]


# 1. The price ranges for all courses range from 20 to 200 with most of the courses having price range bw 25-100
# 2. Like we already explored, expert level doesn't have any free courses

# In[ ]:


sns.catplot(x='level',y='num_lectures',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)


# 1. There are more number of lectures with courses that has all levels 
# 2. Average duration for courses is 25
# 2. Some courses with All levels have around 400-600 lectures

# In[ ]:


udemy.loc[udemy['num_lectures']>400]['course_title']


# These are the courses that has more than 400 lectures, All of them are courses that cover from beginner to expert level except one course that is beginner level and it has around 800 lectures

# In[ ]:


udemy.loc[udemy['num_lectures']>750]


# The course Back to School Web Development and Programming has 779 lectures and is a paid course

# In[ ]:


sns.catplot(x='level',y='num_reviews',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)


# 1. Courses with all levels has more  number of reviews
# 2. Some courses of All levels has more than 10000 reviews
# 3. In beginner's level of difficulty, more reviews are for courses that are free

# In[ ]:


a=udemy.loc[udemy['num_reviews']>5000]
a.groupby('subject').is_paid.value_counts()


# 15 out of 16 courses that has more than 5000 reviews are from the subject web development and 5 of them are free

# In[ ]:


a.groupby('level').subject.value_counts()


# 13 out of the 15 web development courses with more than 5k reviews has all 3 levels of difficulty

# In[ ]:


col_types()


# In[ ]:


sns.catplot(x='level',y='num_subscribers',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)


# 1. The average number of subscribers is around 911
# 2. Courses with all levels of difficulty and beginner level has more number of subscribers
# 3. Most of the courses have less than 25k subscribers
# 4. Courses with expert level of difficulty has less number of subscribers compared to other levels
# 5. There's one course that has whooping 270000 subscribers. Let's explore that course
# 6. There's onec course with beginner level of difficulty and has more than 150000 subs, Let's explore that course too
# 7. Most number of subscribers for a beginner's level courses are all free

# In[ ]:


udemy.loc[udemy['num_subscribers']>250000]


# The course with most number of subscribers is Learn HTML From scratch with 268923 subs and it is a free course

# In[ ]:


udemy.loc[(udemy['num_subscribers']>150000) &(udemy['level']=='Beginner Level') ]


# 1. The course with beginner level of difficulty that has 161029 number of subs is Coding for Entrepreneurs Basic 
# 2. This is also a free course

# In[ ]:


col_types()


# In[ ]:


sns.catplot(x='level',y='content_duration',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)


# Average content duration fora  course is 2 hours

# In data.describe there's a course with 0 duration which is weird let's check that out

# In[ ]:


udemy.loc[udemy['content_duration']==0]


# Mutual Funds for Investors in Retirement Account is a paid course that has 0 subs, lectures and content duration. This can be considered as an outlier
# 

# In[ ]:


udemy1=udemy.copy()
cond=udemy['content_duration']==0
udemy1.drop(cond.index,axis=0,inplace=True)
udemy1.loc[udemy1['content_duration']==0]


# In[ ]:


cond=udemy['content_duration']==0
udemy.drop(udemy[cond].index,axis=0,inplace=True)


# In[ ]:


udemy[udemy['content_duration']==0]


# In[ ]:


col_types()


# In[ ]:


sns.catplot(x='subject',y='price',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)


# Price seems to evenly spread accross the various subjects

# In[ ]:


sns.catplot(x='subject',y='num_subscribers',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=10)


# Like we discussed , Learn HTML from scartch from in Web Development subject has 268000 subs

# In[ ]:


sns.catplot(x='subject',y='num_reviews',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)


# Most of the courses have less than 5000 reviews except some courses in web development

# In[ ]:


udemy.loc[udemy['num_reviews']>25000]


# Web developer Bootcamp with 27445 is the most reviewed course. It has 121584 and has 342 lectures

# In[ ]:


sns.catplot(x='subject',y='num_lectures',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)


# 1. Most of the courses have less than 100 lectures averaging around 25 lectures
# 2. Business Finance and Web Development tend to have more than 200 courses
# 

# In[ ]:


udemy.loc[udemy['num_lectures']>700]


# Back to school web development and programming course has 779 lectures and is the course with highest number of lectures

# In[ ]:


sns.catplot(x='subject',y='content_duration',kind='swarm',data=udemy,hue='is_paid')
plt.xticks(rotation=8)


# There are some handful number of courses which has more than 40hour content duration across all the subjects

# In[ ]:


udemy.loc[udemy['content_duration']>50][['course_title','subject']]


# These are the courses with long content duration

# In[ ]:


col_types()


# # Year of Publishing
# 

# In[ ]:


udemy0=udemy.copy()
udemy0['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
udemy0['published_date'] = udemy0['published_timestamp'].dt.date
udemy0['published_year'] = pd.DatetimeIndex(udemy0['published_date']).year


# In[ ]:


udemy0.groupby(['published_year']).count()


# In[ ]:


plt.figure(figsize = (9,4))
sns.countplot(data = udemy0, x = 'published_year')
plt.show()


# In[ ]:


udemy0.nlargest(5, 'published_timestamp')


# This dataset has courses published till 2017. There is a considerable increase in the number of courses introduced every year, due to the increase in popularity of online courses. The decrease in 2017 can be attributed to the fact the last recorded course was published on 6 July 2017, as visible in the above table. 32.8% of all courses were published in 2016.

# # Further Developments 
# 
# 1. Develop a model to predict if the course is paid or not using Binary Classifiers
#  

# In[ ]:




