#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


TedTalks=pd.read_csv('../input/ted_main.csv')


# In[ ]:


TedTalks.columns


# In[ ]:


TedTalks.shape


# In[ ]:


TedTalks.dtypes


# ****Collect the integer values and find out the correlation between them**

# In[ ]:


TedIntColumns=TedTalks.select_dtypes(include=['int64'])
TedIntColumns.head()


# In[ ]:


TedIntColumns.corr()


# In[ ]:


TedIntColumns.boxplot()


# In[ ]:


sns.heatmap(TedIntColumns.corr())


# Naturally we can see that there is good correlation between film date and publish date.After that there is some correlation between the views and comments.

# Lets us get the descriptive statistics of the different integer columns

# In[ ]:


TedIntColumns.describe()


# In[ ]:


TedIntColumns.boxplot('comments')


# Let us find out the most popular talks.The talks which have the most number of views

# In[ ]:


TedTalks.sort_values('views',ascending=False).head(10)


# In[ ]:


TedTalks[['title','main_speaker']][TedTalks.views==max(TedTalks.views)]


# The talk titled 'Do schools kill creativity' is the most popular talk.Let us find out other talks which are realted to schools

# In[ ]:


TedTalks[['title','main_speaker','views']][TedTalks.title.str.contains('school')].sort_values('views',ascending=False)


# Lets checkout which are the other talks form Ken Robinson

# In[ ]:


TedTalks.loc[TedTalks['main_speaker']=='Ken Robinson'].sort_values('views',ascending=False)


# Let us find out the other talks related to education

# In[ ]:


TedTalks[['title','main_speaker','views']][TedTalks['title'].str.contains('education')]


# In[ ]:


TedTalks['FirstName']=TedTalks['main_speaker'].apply(lambda x:x.split()[0])
TedTalks['FirstName'].head()


# In[ ]:


TedTalks.groupby('main_speaker').views.sum().nlargest(10).plot.bar()


# In[ ]:


TedTalks.groupby('main_speaker').views.mean().nlargest(10).plot.bar()


# In[ ]:


TedTalks.groupby('main_speaker').views.count().nlargest(10).plot.bar()


# In[ ]:


TedTalks.columns


# In[ ]:


TedTalks[['title','main_speaker','views','comments']].sort_values('comments',ascending=False).head(10)


# In[ ]:


TedTalks[['title','main_speaker','views','comments','duration']].sort_values('duration',ascending=False).head(10)


# In[ ]:


import datetime
TedTalks['film_date']=pd.to_datetime(TedTalks['film_date'],unit='s')
TedTalks['published_date']=pd.to_datetime(TedTalks['published_date'],unit='s')


# In[ ]:


TedTalks.groupby(TedTalks.published_date.dt.year).title.count().plot.bar()


# In[ ]:


TedTalks['year']=TedTalks.published_date.dt.year
TedTalks['month']=TedTalks.published_date.dt.month
TedTalks.groupby(['year','month']).title.count().plot.line()


# In[ ]:


Ted_month=TedTalks.groupby(['year','month']).title.count().reset_index(name='talks')
Ted_month.head()
#Ted_month.fillna(0,inplace=True)
Ted_month=Ted_month.pivot('year','month','talks')
Ted_month.fillna(0,inplace=True)
Ted_month


# In[ ]:


sns.heatmap(Ted_month)


# In[ ]:


import ast
TedTalks['ratings'] = TedTalks['ratings'].apply(lambda x: ast.literal_eval(x))
TedTalks['funny'] = TedTalks['ratings'].apply(lambda x: x[0]['count'])
TedTalks['jawdrop'] = TedTalks['ratings'].apply(lambda x: x[-3]['count'])
TedTalks['beautiful'] = TedTalks['ratings'].apply(lambda x: x[3]['count'])
TedTalks['confusing'] = TedTalks['ratings'].apply(lambda x: x[2]['count'])
TedTalks.head()



    


# In[ ]:


TedTalks[['title','main_speaker','funny']].sort_values('funny',ascending=False).head(10).plot.bar(x='title')


# In[ ]:


TedTalks[['title','main_speaker','jawdrop']].sort_values('jawdrop',ascending=False).head(10).plot.bar(x='title')


# In[ ]:


TedTalks[['title','main_speaker','beautiful']].sort_values('beautiful',ascending=False).head(10).plot.bar(x='title')


# In[ ]:


TedTalks[['title','main_speaker','confusing']].sort_values('confusing',ascending=False).head(10).plot.bar(x='title')


# In[ ]:


TedTalks[TedTalks['tags'].str.contains('children')]


# In[ ]:




