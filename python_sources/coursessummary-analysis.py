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


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 1px  black solid !important;\n  color: black !important;\n}\n</style>')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#Load data sets in to dataframes
udemy_df = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
#remove duplicate rows
udemy_df.drop_duplicates(inplace=True)
udemy_df.head()
len(udemy_df.index) 


# In[ ]:


#Get unique course IDs
unique_courses=udemy_df.course_id.unique()
print(np.size(unique_courses))

#Get unique course names
unique_course_names=udemy_df.course_title.unique()
print(np.size(unique_course_names))

#Group by records with counts to find if duplicate occurs
courses_count=udemy_df.groupby(['course_id','course_title'])['course_title'].count().to_frame(name='count').reset_index()
#print(courses_count)

print(courses_count.loc[courses_count['count']>1])


# In[ ]:


#Remove Outliers from the is_paid column
#Get unique course names
unique_paid=udemy_df.is_paid.unique()
print(unique_paid)

#Since values are different, we will convert TRUE to True and FALSE to False
udemy_df.loc[udemy_df.is_paid == "TRUE", "is_paid"] = "True"
udemy_df.loc[udemy_df.is_paid == "FALSE", "is_paid"] = "False"

#Get free and Paid courses
free_Courses=udemy_df.loc[udemy_df['is_paid']=='False']
paid_Courses=udemy_df.loc[udemy_df['is_paid']=='True']
#free_Courses
#paid_Courses
#Get outlier value
outlier_Value=udemy_df.loc[(udemy_df['is_paid'] != 'True') & (udemy_df['is_paid'] != 'False')]
#Remove outlier value
final_df=pd.merge(udemy_df,outlier_Value, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
final_df


# In[ ]:


# Let's analyze free courses
free_Courses.head()
free_subject_count=free_Courses.groupby(['subject'])['subject'].count().to_frame(name='count').reset_index()
free_subject_count

#plot this on pie chart
plt.axis("equal")
plt.pie(free_subject_count['count'],labels=free_subject_count['subject'],radius=1.1,autopct='%.2f',explode=[0,0.1,0.1,0])
plt.title("Free Courses offered by Udemy",bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# In[ ]:


# Let's analyze paid courses
paid_Courses.head()
paid_subject_count=paid_Courses.groupby(['subject'])['subject'].count().to_frame(name='count').reset_index()
paid_subject_count

#plot this on pie chart
plt.axis("equal")
plt.pie(paid_subject_count['count'],labels=free_subject_count['subject'],radius=1.1,autopct='%.2f',explode=[0,0.1,0.1,0])
plt.title("Paid Courses offered by Udemy",bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# In[ ]:


#Clean data for content duration

final_df['content_duration']=final_df['content_duration'].str.replace('hours','').str.strip()
final_df['content_duration']=final_df['content_duration'].str.replace('hours','').str.strip()
final_df.head()


# In[ ]:


#Get most subscribers for free courses
#free_Courses.head()
free_subscribers=free_Courses.groupby(['subject'])['num_subscribers'].sum().to_frame(name='total_Subscribers').reset_index()
free_subscribers

#plot this on pie chart
plt.axis("equal")
plt.pie(free_subscribers['total_Subscribers'],labels=free_subscribers['subject'],radius=.8,autopct='%.2f',explode=[0,0.1,0.1,0])
plt.title("Subcribers on Free Courses offered by Udemy",bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# In[ ]:


#Get most subscribers for paid courses
#free_Courses.head()
paid_subscribers=paid_Courses.groupby(['subject'])['num_subscribers'].sum().to_frame(name='total_Subscribers').reset_index()
paid_subscribers

#plot this on pie chart
plt.axis("equal")
plt.pie(paid_subscribers['total_Subscribers'],labels=paid_subscribers['subject'],radius=.8,autopct='%.2f',explode=[0,0.1,0.1,0])
plt.title("Subcribers on Paid Courses offered by Udemy",bbox={'facecolor':'0.8', 'pad':1},loc='right')
plt.show()


# In[ ]:


paid_Courses['price']=paid_Courses['price'].astype(int)
paid_Courses.dtypes
paid_Courses['price'].describe()

cheap_paid_Courses= paid_Courses.loc[paid_Courses['price']<25]
cheap_paid_Courses


# In[ ]:


#Get chep courses details
cheap_paid_Courses
#print(np.size(cheap_unique_courses))
#cheap_paid_Courses=cheap_paid_Courses.loc[cheap_paid_Courses['course_title'].duplicated()]
courses_count=cheap_paid_Courses.groupby(['subject'])['course_id'].count().to_frame(name='count').reset_index()
courses_count

#plot this on pie chart
plt.axis("equal")
plt.pie(courses_count['count'],labels=courses_count['subject'],radius=1,autopct='%.2f')
#plt.title("Most subscribed Courses offered by Udemy")
plt.title("Cheap Courses offered by Udemy",bbox={'facecolor':'.8', 'pad':1},loc='center')
plt.show()


# In[ ]:


paid_Courses.dtypes
paid_Courses.num_subscribers.describe()
most_subscribed_Courses= paid_Courses.loc[paid_Courses['num_subscribers']>75000]
most_subscribed_Courses=most_subscribed_Courses[['course_title','num_subscribers']]
most_subscribed_Courses

#plot this on pie chart
plt.axis("equal")
plt.pie(most_subscribed_Courses['num_subscribers'],labels=most_subscribed_Courses['course_title'],radius=1,autopct='%.2f')
#plt.title("Most subscribed Courses offered by Udemy")
plt.title("Most subscribed Courses offered by Udemy",bbox={'facecolor':'.8', 'pad':1},loc='center')
plt.show()

