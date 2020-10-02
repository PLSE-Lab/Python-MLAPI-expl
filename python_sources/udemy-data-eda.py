#!/usr/bin/env python
# coding: utf-8

# # **Importing Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Importing the data from CSV**

# In[ ]:


PATH = "../input/udemy-courses/"
dataset=pd.read_csv(PATH + 'udemy_courses.csv')


# # Data checking and cleaning
# **Checking the imported data**

# In[ ]:


dataset.head(10)


# **Converting to datetime format and creating a new column with only year**

# In[ ]:


dataset['published_timestamp'] = pd.to_datetime(dataset['published_timestamp'])
dataset['year']=pd.DatetimeIndex(dataset['published_timestamp']).year


# **Checking for non-zeros**

# In[ ]:


dataset.info()


# **Removing the columns course_id,url**

# In[ ]:


dataset=dataset.drop(columns=['course_id','url'])


# # Initial observations

# In[ ]:


dataset.describe().transpose()


# 
# * The price of the courses varied from 0 to 200 dollars
# * There are courses with zero subscribers
# * Very few courses have  subscribers over 2600
# * The max number of reviews are 10 times less than the max subscribers
# * There are courses with zero hours content
# * Max content duration is 78.5hours

# In[ ]:


subjects_list=dataset["subject"].unique()
subject_count=dataset['subject'].value_counts().reset_index()


# In[ ]:


fig11=px.bar(subject_count,x='index',y='subject',text='subject',color='subject',title='courses per subject',labels={'index':'Subject','subject':'No. of courses'})
fig11.update_layout(showlegend=False, width=600)
fig11.show()


# More courses are under the subjects Web Development and Business Finance
# Least courses are under the subject Graphic Design

# In[ ]:


courses_charge=dataset['is_paid'].value_counts().reset_index()
fig12=px.pie(courses_charge,values='is_paid',names='index',title='Free Vs Paid courses')
fig12.update_layout(showlegend=True, width=600)
fig12.show()


# Most of the offered courses are paid courses

# In[ ]:


courses_level=dataset['level'].value_counts().reset_index()
fig13=px.pie(courses_level,values='level',names='index',title='Offered courses Level')
fig13.update_layout(showlegend=True, width=600)
fig13.show()


# Most of the offered courses are for All levels followed by Beginner level and Intermediate level.
# Only 1.58% of the offered courses are advanced level.

# # Most engaging courses

# In[ ]:


dataset_sorted=dataset.sort_values(['num_subscribers'],ascending=[False])
fig2=px.bar(dataset_sorted[0:6],x='course_title',y='num_subscribers',text='num_subscribers',color='price',title='Top 5 courses by subcription count')
fig2.update_layout(showlegend=False, width=600)
fig2.show()


# The courses with highest subscription count are programming courses (5 out of 6). Out of the top 6 subscribed courses, 4 are free courses and 2 are paid courses(worth 200$).

# In[ ]:


dataset_sorted_reviews=dataset.sort_values(['num_reviews'],ascending=[False])
fig3=px.bar(dataset_sorted_reviews[0:6],x='course_title',y='num_reviews',text='num_reviews',color='price',title='Top 5 courses by subcription count')
fig3.update_layout(showlegend=False, width=600)
fig3.show()


# The courses with more reviews are programming courses. Out of the top 6 reviewed courses, 4 are paid courses (200$, 200$, 190$, 180$) and 2 are free courses. It shows that the paid courses have more enegagement from the subjects.

# # **Analyse individual subject**
# :Business Finance
# ,Graphic Design
# ,Musical Instruments
# ,Web Development

# In[ ]:


dataset_BF=dataset[dataset['subject']==subjects_list[0]]
dataset_GD=dataset[dataset['subject']==subjects_list[1]]
dataset_MI=dataset[dataset['subject']==subjects_list[2]]
dataset_WD=dataset[dataset['subject']==subjects_list[3]]

BF_stats=pd.crosstab(dataset_BF.year,dataset_BF.is_paid)
GD_stats=pd.crosstab(dataset_GD.year,dataset_GD.is_paid)
MI_stats=pd.crosstab(dataset_MI.year,dataset_MI.is_paid)
WD_stats=pd.crosstab(dataset_WD.year,dataset_WD.is_paid)


# **Free Vs Paid courses per subject**

# In[ ]:


BF_stats.plot.bar(stacked=True)
plt.legend(title='Business Finance')
plt.show()

GD_stats.plot.bar(stacked=True)
plt.legend(title='Graphic Design')
plt.show()


# In[ ]:


MI_stats.plot.bar(stacked=True)
plt.legend(title='Music Instruments')
plt.show()

WD_stats.plot.bar(stacked=True)
plt.legend(title='Web Development')
plt.show()


# The offered courses are mostly paid courses with less free courses among all the subjects.

# **Course Levels per subject**

# In[ ]:


courselevel_BF=dataset_BF['level'].value_counts().reset_index()
courselevel_GD=dataset_BF['level'].value_counts().reset_index()
courselevel_MI=dataset_BF['level'].value_counts().reset_index()
courselevel_WD=dataset_BF['level'].value_counts().reset_index()


# In[ ]:


fig41=px.pie(courselevel_BF,values='level',names='index',title='Business Finance-courses Level')
fig41.update_layout(showlegend=True, width=500)
fig41.show()

fig42=px.pie(courselevel_GD,values='level',names='index',title='Graphic Design-courses Level')
fig42.update_layout(showlegend=True, width=500)
fig42.show()


# In[ ]:


fig43=px.pie(courselevel_MI,values='level',names='index',title='Musical Instrtuments-courses Level')
fig43.update_layout(showlegend=True, width=500)
fig43.show()

fig44=px.pie(courselevel_WD,values='level',names='index',title='Web Development-courses Level')
fig44.update_layout(showlegend=True, width=500)
fig44.show()


# Among all the subjects, many courses are for all levels followed by begineer level, Intermediate level and expert level. 
# Very few expert level courses are offered in all the subjects

# **Top Subscribed courses per subject**

# In[ ]:


sorted_dataset_BF=dataset_BF.sort_values(['num_subscribers'],ascending=[False])
sorted_dataset_BF.head(5)


# In[ ]:


sorted_dataset_GD=dataset_GD.sort_values(['num_subscribers'],ascending=[False])
sorted_dataset_GD.head(5)


# In[ ]:


sorted_dataset_MI=dataset_MI.sort_values(['num_subscribers'],ascending=[False])
sorted_dataset_MI.head(5)


# In[ ]:


sorted_dataset_WD=dataset_WD.sort_values(['num_subscribers'],ascending=[False])
sorted_dataset_WD.head(5)


# By Subscription count, the top 5 courses are programming courses.

# **Price distribution of Paid courses per subject**

# In[ ]:


BF_price=pd.crosstab(dataset_BF.is_paid,dataset_BF.price).stack().reset_index(name='count')
fig51=px.bar(BF_price,x='price',y='count',text='count',color='price',title='Price distribution of the Paid courses - Business Finance')
fig51.update_layout(showlegend=False, width=600)
fig51.show()


# 96 courses are free and 299 courses are priced 20$ in Business Finance followed by 163 courses that are priced 50$.
# 128 courses are priced 200$(max. price).

# In[ ]:


GD_price=pd.crosstab(dataset_GD.is_paid,dataset_GD.price).stack().reset_index(name='count')
fig52=px.bar(GD_price,x='price',y='count',text='count',color='price',title='Price distribution of the Paid courses - Graphic Design')
fig52.update_layout(showlegend=False, width=600)
fig52.show()


# 35 free courses and 203 courses are priced 20$ in Graphic Design.
# 35 courses are priced 200$(max. price).

# In[ ]:


MI_price=pd.crosstab(dataset_MI.is_paid,dataset_MI.price).stack().reset_index(name='count')
fig53=px.bar(MI_price,x='price',y='count',text='count',color='price',title='Price distribution of the Paid courses - Musical Instrument')
fig53.update_layout(showlegend=False, width=600)
fig53.show()


# 46 courses are free and 143 courses are priced 50$, 141 courses are priced 20$ in Musical Instruments
# 19 courses are priced 200$ (max. price).

# In[ ]:


WD_price=pd.crosstab(dataset_WD.is_paid,dataset_WD.price).stack().reset_index(name='count')
fig54=px.bar(WD_price,x='price',y='count',text='count',color='price',title='Price distribution of the Paid courses - Musical Instrument')
fig54.update_layout(showlegend=False, width=600)
fig54.show()


# 133 courses are free and 187 courses are priced 20$ in web development.
# 113 courses are priced 200$ (max. price).

# **Paid Vs Free course duration per subject**

# In[ ]:


fig61=px.box(dataset_BF,x='content_duration',y='is_paid',orientation='h',color='is_paid',title='Duration Distribution - Business Finance')
fig61.update_xaxes(title='Content Duration')
fig61.update_yaxes(title='Business Finance courses')
fig61.update_layout(showlegend=False, width=600)
fig61.show()


# In[ ]:


fig62=px.box(dataset_GD,x='content_duration',y='is_paid',orientation='h',color='is_paid',title='Duration Distribution - Graphic Design')
fig62.update_xaxes(title='Content Duration')
fig62.update_yaxes(title='Graphic Design courses')
fig62.update_layout(showlegend=False, width=600)
fig62.show()


# In[ ]:


fig63=px.box(dataset_MI,x='content_duration',y='is_paid',orientation='h',color='is_paid',title='Duration Distribution - Musical Instrument')
fig63.update_xaxes(title='Content Duration')
fig63.update_yaxes(title='Musical Instrument courses')
fig63.update_layout(showlegend=False, width=600)
fig63.show()


# In[ ]:


fig64=px.box(dataset_WD,x='content_duration',y='is_paid',orientation='h',color='is_paid',title='Duration Distribution - Web Development')
fig64.update_xaxes(title='Content Duration')
fig64.update_yaxes(title='Web Development courses')
fig64.update_layout(showlegend=False, width=600)
fig64.show()


# In all the subject courses, paid courses provide more duration compared to free courses.

# **Checking for course keyword per subject**

# In[ ]:


stopwords = set(STOPWORDS)
wordcloud1 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(''.join(dataset_BF['course_title']))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud1)
plt.axis("off") 
#plt.tight_layout(pad = 0)  
plt.show() 


# It can be observed that most of courses in Business Finance are based on "Trading","Stock","Accounting","Financial" and "Forex"

# In[ ]:


wordcloud2 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(''.join(dataset_GD['course_title']))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud2)
plt.axis("off") 
#plt.tight_layout(pad = 0)  
plt.show() 


# It can be observed that most of courses in Graphic Design are based on Adobe tools "Illustrator","Photoshop" for designing "Logos","Images" for books/web design

# In[ ]:


wordcloud3 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(''.join(dataset_MI['course_title']))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud3)
plt.axis("off") 
#plt.tight_layout(pad = 0)  
plt.show()


# It can be observed that most of courses in Musical Instruments are for "Piano","Guitar" Logos" and "Harmonica"

# In[ ]:


wordcloud4 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(''.join(dataset_WD['course_title']))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud4)
plt.axis("off") 
#plt.tight_layout(pad = 0)  
plt.show()


# It can be observed that most of courses in Web development are based on "Javascript","PHP" and "Wordpress"

# # Final Conclusions per subject

# **Business Finance subject -Conclusions:**
# Based on the above analysis in Business Finance, of the total 1195 courses.
# * 96 courses are free with mean content duration of 2.15 hours and 252 reviews
# * 1099 courses are paid courses with mean price of 74.5$, mean content duration of 3.7 hours and 47 reviews
# * Based on the level, 58% of the courses are for all levels, 28% are beginners level, 11% are intermediate level and only 3% are expert level
# * 40 courses were never subscribed
# * Top subscribed courses are free courses and are the following
#              "Bitcoin or How I Learned to Stop Worrying and Love Crypto"  
#              "Accounting in 60 Minutes - A Brief Introduction" 
#              "Stock Market Investing for Beginners" 
#              "Introduction to Financial Modeling"             
# * Top subcribed paid courses are "The Complete Financial Analyst Course 2017" and "Beginner to Pro in Excel: Financial Modeling and Valuation"
# * It was also observed that content duration is high for paid courses when compared to free courses.

# **Graphic Design subject -Conclusions:**
# Based on the above analysis in Graphic Design, of the total 603 courses.
# * 35 courses are free with mean content duration of 2 hours and 302 reviews
# * 568 courses are paid courses with mean price of 62$, mean content duration of 3.7 hours and 46 reviews
# * Based on the level, 49% of the courses are for all levels, 40% are beginners level, 9% are intermediate level and only 1% are expert level
# * 19 courses were never subscribed
# * Top subscribed courses are free courses and are the following
#             "Photoshop in-depth: Master all of Photoshop's Tools easily"  
#             "Professional Logo Design in Adobe Illustrator"        
# * Top subcribed paid courses are "Photoshop for Entreprenuers-Design 11 Practical Projects" and "Logo Design Essentials"
# * It was also observed that content duration is high for paid courses when compared to free courses.

# **Musical Instruments subject -Conclusions:**
# Based on the above analysis in Musical Instruments, of the total 680 courses.
# * 46 courses are free with mean content duration of 1.6 hours and 149 reviews
# * 634 courses are paid courses with mean price of 53$, mean content duration of 2.95 hours and 39 reviews
# * Based on the level, 41% of the courses are for all levels, 44% are beginners level, 15% are intermediate level and only 1% are expert level
# * 11 courses were never subscribed
# * Top subscribed courses are free courses and are the following
#              "Free Beginner Electric Guitar Lessons"  
#              "Getting Started with Playing Guitar"        
# * Top subcribed paid courses are "Pianoforall - Incredible New Way to Learn Piano and Keyboard" and "Complete Guitar System - Beginner to Advanced" and "Learn Guitar in 21 days"
# * It was also observed that content duration is high for paid courses when compared to free courses.
# * Unlikes Business Finance, Graphic Design and Web Development, most subscribed courses in Musical Instruments subject are paid courses.

# **Web Development subject -Conclusions:**
# Based on the above analysis in Musical Instruments, of the total 1200 courses.
# * 133 courses are free with mean content duration of 2.56 hours and 680 reviews
# * 1067 courses are paid courses with mean price of 87$, mean content duration of 5.97 hours and 318 reviews
# * Based on the level, 55% of the courses are for all levels, 33% are beginners level, 11% are intermediate level and only 1% are expert level
# * No courses with no subscribers. Min.19 subscribers for a course
# * Top subscribed courses are free courses and are the following
#             "Learn HTML5 Programming From Scratch"  
#             "Coding for Entrepreneurs Basic", "Build Your First Website in 1 week with HTML5 and CSS3"        
# * Top subcribed paid courses are "The Web Developer Bootcamp" and "The Complete Web Developer Course 2.0" and "Learn Javascript & JQuery From Scratch"
# * It was also observed that content duration is high for paid courses when compared to free courses.

# # Final general conclusions: 
# The most sought out courses are programming courses. Some courses from subjects Business Finance, Graphic Design and Musical Instruments have zero subscribers but Web Development did not have any zero subscribed courses indicating the interest/demand for programming languages. The cost of top subcribed paid courses is 200$ from web development subject. It was also observed that paid courses have more engagement (in terms of reviews) when compared to free courses.
