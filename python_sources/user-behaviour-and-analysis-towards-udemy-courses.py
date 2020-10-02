#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import pandas as pd
import seaborn as sb
import numpy as np


# In[ ]:


df = pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")


# In[ ]:


df.head()


# In[ ]:


df.drop(['course_id'],axis=1,inplace = True)


# In[ ]:


df.price.value_counts()[:10]


# In[ ]:


df.groupby(['is_paid'])['content_duration'].sum().plot(kind="barh")


# In[ ]:


df.groupby(['price'])['content_duration'].sum()[:10].plot(kind="bar")


# In[ ]:


import matplotlib. pyplot as plt

sorted_counts = df['level'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False,autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2);
plt.axis('square')


# In[ ]:


plt.scatter(data = df, x = 'content_duration', y = 'level')


# In[ ]:


sb.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)


# In[ ]:


sb.regplot(data = df, x = 'num_subscribers', y = 'num_reviews', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})


# In[ ]:


df['year'] = pd.DatetimeIndex(df['published_timestamp']).year


# In[ ]:


sb.distplot(df['year'])


# In[ ]:


base_color = sb.color_palette()[0]
sb.barplot(data = df, x = 'year', y = 'num_subscribers', color = base_color)


# In[ ]:


df.subject.value_counts()


# In[ ]:


sb.countplot(data = df, y = 'subject', hue = 'is_paid')


# In[ ]:


base_color = sb.color_palette()[0]
sb.violinplot(data = df, y = 'subject', x = 'year', color = base_color,
              inner = 'quartile')


# In[ ]:





# ## **Some Insights found: **

# ***1.Most people prefer courses that are worth upto 20 dollars***
# 
# ***2.Paid courses has a very high duration as compared to free courses. We can understand that free courses are short mini course while paid ones takes the concepts and course in depth***
# 
# ***3.We can see courses that cost around 50 dollars have highest duration followed by the courses worth 20 dollars. This could be because udemy keep offfering high courses to users at 20 dollars sometimes.***
# 
# ***4.Almost 50% of courses available are of equal level for all learners. Then 35% are begineers level  followed by 11% of intermediate level. Only 2% of entire course consist of expert level.***
# 
# ***5.There are outliers in beginner level data. We expert level to have lower content duration. All levels being the mostly time consuming courses.***
# 
# ***6.We can see that number number of subscribers is closely related with number of reviews. Thus we can see more people taking courses and hence they subscribe to those courses and automatically the course get high subscribers.***
# 
# ***7.We have number of subscribers negetively correlated with paid courses. So the free courses has many subscribers.***
# 
# ***8.The highest number of courses were published in the year 2016***
# 
# ***9.There is a trend of decrease in number of subscribers every year***
# 
# ***10.Most number of courses available belongs to web development and lowest to graphic design***
# 
# ***11.We have lowest number of free courses in graphic design and highest paid course in business finance***
# 
# ***12.In the initial time web develoment courses were sold. And that could be the reason why our dataset has most web devlopment courses.***
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# 
