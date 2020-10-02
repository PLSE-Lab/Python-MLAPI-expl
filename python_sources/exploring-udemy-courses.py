#!/usr/bin/env python
# coding: utf-8

# <h1>Exploring Udemy Courses Dataset</h1>
# <img src='data:image/png;base64,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'>
# <p>In this kernel, We want to explore udemy courses dataset in general. </p>

# <h1>Loading Dataset and Getting Some informations about it</h1>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
import re
plt.style.use('ggplot')
sns.set(style='darkgrid', context='notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


udemy_courses=pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv', parse_dates=['published_timestamp'])
udemy_courses.head(3)


# In[ ]:


udemy_courses.info()


# <ul>
#     <li>convert price column to numeric type</li>
#     <li>convert content_duration column to float type column</li>
#     <li>convert published_timestamp to datetime type column</li>
# </ul>

# <h1>Data Cleaning and Prepration for EDA</h1>

# In[ ]:


udemy_courses['price']=udemy_courses['price'].str.replace('Free', '0').str.replace('TRUE', '0')
udemy_courses['price']=udemy_courses['price'].astype('float')
udemy_courses['number_of_contents']=udemy_courses['content_duration'].str.extract('([\d\.]+)\s[\w]+').astype('float')
udemy_courses['content_duration_type']=udemy_courses['content_duration'].str.extract('[\d\.]+\s([\w]+)')
udemy_courses.drop('content_duration', axis=1, inplace=True)


# In[ ]:


udemy_courses.dropna(inplace=True)


# In[ ]:


new_dates=[]
for i in udemy_courses['published_timestamp']:
    new_date=dt.datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
    new_dates.append(new_date)
udemy_courses['published_timestamp']=new_dates


# In[ ]:


udemy_courses['is_paid'].value_counts()


# In[ ]:


udemy_courses['is_paid']=udemy_courses['is_paid'].str.replace('TRUE', 'True')
udemy_courses['is_paid']=udemy_courses['is_paid'].str.replace('FALSE', 'False')


# In[ ]:


udemy_courses['content_duration_type']=udemy_courses['content_duration_type'].str.replace('hours', 'hour')


# <h1>EDA</h1>
# <h2>Basic Knowledge EDA</h2>
# <ul>
#     <li>How many Free Courses in the dataset, and Explore which Category has the most free courses</li>
#     <li>How many Paid Courses in the dataset, and Explore which Category has the most paid courses</li>
#     <li>Compare the Average number of subscribers between Paid and Free Courses</li>
#     <li>How many Free/Paid Courses in each Level</li>
#     <li>Explore the distribution of price column</li>
#     <li>Average Prices for each level (Paid Category)</li>
#     <li>Relation between number of lectures and price.</li>
#     <li>Explore content duration and content duration for each price category</li>
#     <li>Published Courses each year.</li>
# </ul>

# In[ ]:


udemy_courses['is_paid'].value_counts()


# In[ ]:


g=sns.catplot(x='subject',
            data=udemy_courses,
            kind='count',
            hue='is_paid')
g.fig.suptitle('free/paid categories comparison', y=1.03)
plt.xticks(rotation=90)
plt.show()


# <p>Web Development Category has the most free courses<br>
# Business Finance Category has the most free courses.</p>

# In[ ]:


pd.pivot_table(index='is_paid', values='num_subscribers', data=udemy_courses, aggfunc='mean')


# <p>Average number of subscribers in free courses is greater than the Average number of subscribers in paid courses<br>
#     and we can say that this result is expected.</p>

# In[ ]:


g=sns.catplot(x='level',
            data=udemy_courses,
            kind='count',
            hue='is_paid')
g.fig.suptitle('free/paid lavel comparison', y=1.03)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Box-plot, CDF, histogram
def cdf(lst):
    x=np.sort(lst)
    y=np.arange(1, len(x)+1)/len(x)
    return x, y
fig, ax=plt.subplots(1,3, figsize=(15,5))
x_price, y_price=cdf(udemy_courses['price'])
ax[0].plot(x_price, y_price)
ax[0].set_title('CDF of prices')
ax[1].hist(udemy_courses['price'])
ax[1].set_title('histogram distribution of prices')
ax[2].boxplot(udemy_courses['price'])
ax[2].set_title('boxplot of prices')
plt.show()
print('median: ',udemy_courses['price'].median())
print('mean: ',udemy_courses['price'].mean())


# <p>Right skewed Distribution, we have 1840 courses thier price equal to 45 (median value) or below 45.<br>
# no outliers, mean value of price column is 66</p>

# In[ ]:


udemy_courses[udemy_courses['is_paid']=='True'].groupby('level')['price'].mean().sort_values(ascending=False)


# In[ ]:


price_category=[]
for i in udemy_courses['price']:
    if i==0:
        price_category.append('Free')
    elif i>0 and i<=45:
        price_category.append('cheap')
    elif i>45 and i<=95:
        price_category.append('expensive')
    else:
        price_category.append('very expensive')
udemy_courses['price_category']=price_category


# In[ ]:


udemy_courses['price_category'].value_counts()


# In[ ]:


# relation between number of lectures and price
udemy_courses.groupby('price_category')['num_lectures'].mean().sort_values(ascending=False)


# <p>it is clear now, as the price increases, the number of lectures increase and vice versa.</p>

# In[ ]:


udemy_courses['content_duration_type'].value_counts()


# <p>here we can see that we have some courses thier content duration are minutes and questions</p> 

# In[ ]:


# let's see the courses that thier content duration type = questions.
udemy_courses[udemy_courses['content_duration_type']=='questions']


# In[ ]:



mins_courses=udemy_courses[udemy_courses['content_duration_type']=='mins']
g=sns.catplot(x='price_category',
            data=mins_courses,
            kind='count',
            hue='level')
g.fig.suptitle("price category (minutes courses) in each level.", y=1.03, x=0.4)
plt.show()


# In[ ]:


udemy_courses['published_timestamp'].dt.year.value_counts().plot.bar()
plt.xlabel('published year')
plt.ylabel('frequency')
plt.title('number of courses published each year(2011-2017)')
plt.show()


# In[ ]:


udemy_courses['published_year']=udemy_courses['published_timestamp'].dt.year
pd.pivot_table(index='published_year',
               columns='subject',
               values='course_id',
               data=udemy_courses,
               aggfunc='count',
               fill_value=0)

