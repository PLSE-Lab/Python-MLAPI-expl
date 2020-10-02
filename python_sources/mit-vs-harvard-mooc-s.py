#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mooc=pd.read_csv('../input/appendix.csv')


# In[ ]:


mooc.head(2)


# ### Checking Data Quality

# In[ ]:


mooc.isnull().sum()


# Just one null value in the Instructors column.
# 
# So lets do some Exploratory Analysis

# ### Mooc's Exploratory Analysis

# In[ ]:


print('The Different Course Subjects Are:',mooc['Course Subject'].unique())


# In[ ]:


print('The number of different Course Topics are: ',mooc['Course Title'].nunique())


# ### Total Courses By Each University

# In[ ]:


mooc['Institution'].value_counts().plot.bar()


# ### Distribution Of the Course Subjects

# In[ ]:


fig=plt.gcf()
subjects=mooc['Course Subject'].value_counts().index.tolist()
size=mooc['Course Subject'].value_counts().values.tolist()
plt.pie(size,labels=subjects,explode=(0,0,0,0.1),startangle=90,shadow=True,autopct='%1.1f%%')
plt.title('Course Subjects ')
fig.set_size_inches((6,6))
plt.show()


#  **Lets Dig in and find what distribution of courses do both the University has**

# In[ ]:


plt.subplot(211)
mit=mooc[mooc['Institution']=='MITx']
plt.pie(mit['Course Subject'].value_counts().values.tolist(),labels=mit['Course Subject'].value_counts().index.tolist(),startangle=90,explode=(0.1,0,0,0),autopct='%1.1f%%',shadow=True,colors=['Y', '#1f2ff3', '#0fff00', 'R'])
plt.title('MITx')
plt.subplot(212)
harvard=mooc[mooc['Institution']=='HarvardX']
plt.pie(harvard['Course Subject'].value_counts().values.tolist(),labels=harvard['Course Subject'].value_counts().index.tolist(),startangle=90,explode=(0.1,0,0,0),autopct='%1.1f%%',shadow=True,colors=['Y', '#1f2ff3', '#0fff00', 'R'])
plt.title('Harvardx')
fig=plt.gcf()
fig.set_size_inches((11,11))
plt.show()


# **Observations:**
# 
#  - **MIT** is providing majority of courses on subjects belonging to Science, Technology And Engineering.
#  - **Harvard** has focused a lot on subjects like Humanities, History i.e to more general topics.
#  - Both the universities are providing sufficient courses on Government and Health Services.
#  - Overall **MIT** has focused more on Engineering and Computer Sciences while **Harvard** has focused more on general topics.
# 
#  

# ### Yearly Courses By The Universities

# In[ ]:


mooc['Year']=pd.DatetimeIndex(mooc['Launch Date']).year  #taking only the year from the date
sns.countplot('Year',hue='Institution',data=mooc)


# We can clearly see that MIT has always provided more courses than Harvard except in the year 2014

# ### Course Participants each Year by University

# In[ ]:


a=mooc.groupby(['Institution','Year'])['Participants (Course Content Accessed)'].sum().reset_index()
a.pivot('Year','Institution','Participants (Course Content Accessed)').plot.bar()
plt.show()


# As we can see, there has a substantial increase in the number of participants from 2012-2015. Only in 2016 was there  a depreciation. 
# 
# There are more students in MIT in some year whereas in some year Harvard has more students . Thus there hasn't been a fixed trend.

# ### Lets check what type of subjects do Male and Female Students Prefer

# In[ ]:


abc=mooc[mooc['% Female'] >= mooc['% Male']]
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(" ".join(abc['Course Title']))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# As seen in the word cloud, we can say that Female Candidates mostly enroll for Subjects revolving around History, Politics and many such General Topics . Enrolment for courses like Computer Science is low

# In[ ]:


abc=mooc[mooc['% Female'] < mooc['% Male']]

wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(" ".join(abc['Course Title']))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# The above wordcloud shows that male candidates mostly enrol for subjects related to Computers, Sciences and Programming. Enrolment in generic courses like History is low.

# ### More to Come...Stay Tuned
# **Please Upvote** if u liked it!!

# In[ ]:




