#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd 
import re
import os
from datetime import datetime
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud ,STOPWORDS, ImageColorGenerator
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
import plotly.tools as tls
import plotly.plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
sns.set()


# In[ ]:


p = "YlGnBu"
p2 = "YlGn"
p3 = "Greys"


# In[ ]:


df = pd.read_csv("../input/nyc-jobs.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Data cleaning & Feature engineering

# In[ ]:


# helper functions
def plot_wordcloud(text):
    wordcloud = WordCloud(background_color='white',
                     width=1024, height=720).generate(text)
    plt.clf()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

def parse_categories(x):
    l = x.replace('&', ',').split(',')
    l = [x.strip().rstrip(',') for x in l]
    key_categories.extend(l)

def parse_keywords(x, l):
    x = x.lower()
    tokens = nltk.word_tokenize(x)
    stop_words = set(stopwords.words('english'))
    token_l = [w for w in tokens if not w in stop_words and w.isalpha()]
    l.extend(token_l)
    
def preferred_skills(x):
    kwl = []
    df[df.job_category==x].preferred_skills.dropna().apply(parse_keywords, l=kwl)
    kwl = pd.Series(kwl)
    return kwl.value_counts()[:20]


# In[ ]:


df.drop(['Recruitment Contact', 'Post Until'], axis=1, inplace=True)
df.columns = ['id', 'agency', 'posting_type', 'number_of_positions', 'business_title', 'civil_service_title', 'title_code_number', 'level', 'job_category', 'full-time/part-time', 'salary_range_low', 'salary_range_high', 'salary_frequency', 'work_location', 'work_unit', 'job_description', 'minimum_requirements', 'preferred_skills', 'additional_info', 'to_apply', 'shift', 'work_location_1', 'residency_requirement', 'posting_date', 'posting_updated', 'process_date' ]


# In[ ]:


df.job_category.value_counts()


# > - There are many jobs demanding composite skillsets, lets break the composite categories down to single categories and generate a  countplot

# In[ ]:


key_categories = []
df.job_category.dropna().apply(parse_categories)
key_categories = pd.Series(key_categories)
key_categories = key_categories[key_categories!='']
popular_categories = key_categories.value_counts().iloc[:25]


# ## Full time or Part time?

# In[ ]:


sns.countplot(x='full-time/part-time', data=df, palette=p)


# ## Salary frequency

# In[ ]:


sns.countplot(x='salary_frequency', data=df, palette=p2)


# ## Most in-demand job categories

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y=key_categories, order=popular_categories.index, palette=p)


# In[ ]:


plt.figure(figsize=(10,8))
plot_wordcloud(' '.join(popular_categories.index.tolist()))


# ## Most popular job titles

# In[ ]:


plt.figure(figsize=(10,10))
titles_freq = df.civil_service_title.value_counts()
popular_titles = titles_freq.iloc[:25]
sns.countplot(y="civil_service_title", data=df, order=popular_titles.index, palette=p)


# In[ ]:


plt.figure(figsize=(10,8))
plot_wordcloud(' '.join(popular_titles.index.tolist()))


# ## Least popular job titles

# In[ ]:


least_popular = titles_freq.iloc[-10:]
sns.countplot(y="civil_service_title", data=df, order=least_popular.index, palette=p3)


# ## Jobs with highest low salary range (annual)

# In[ ]:


salary_table = df[['civil_service_title', 'salary_range_low', 'salary_range_high']]
jobs_highest_low_range = pd.DataFrame(salary_table.groupby(['civil_service_title'])['salary_range_low'].mean().nlargest(10)).reset_index()
plt.figure(figsize=(8,6))
sns.barplot(y='civil_service_title', x='salary_range_low', data=jobs_highest_low_range, palette=p)


# ## Jobs with highest high salary range (annual)

# In[ ]:


jobs_highest_high_range = pd.DataFrame(salary_table.groupby(['civil_service_title'])['salary_range_high'].mean().nlargest(10)).reset_index()
plt.figure(figsize=(8,6))
sns.barplot(y='civil_service_title', x='salary_range_high', data=jobs_highest_high_range, palette=p)


# ## Highest paying jobs on an hourly basis

# In[ ]:


hourly_jobs = df[df.salary_frequency == 'Hourly']
jobs_highest_high_range_hourly = pd.DataFrame(hourly_jobs.groupby(['civil_service_title'])['salary_range_high'].mean().nlargest(10)).reset_index()
plt.figure(figsize=(8,6))
sns.barplot(y='civil_service_title', x='salary_range_high', data=jobs_highest_high_range_hourly, palette=p2)


# In[ ]:


plt.figure(figsize=(10,8))
plot_wordcloud(' '.join(jobs_highest_high_range_hourly['civil_service_title'].tolist()))


# ## Hourly jobs salary distribution

# In[ ]:


sns.distplot(hourly_jobs.salary_range_high)


# ## Popular work units

# In[ ]:


popular_divisions = df.work_unit.value_counts().iloc[:10]
sns.countplot(y='work_unit', data=df, order=popular_divisions.index, palette='PuBuGn')


# ## Most common keywords in job descriptions

# In[ ]:


job_description_keywords = []
df.job_description.apply(parse_keywords, l=job_description_keywords)
plt.figure(figsize=(10, 8))
counter = Counter(job_description_keywords)
common = [x[0] for x in counter.most_common(40)]
plot_wordcloud(' '.join(common))


# ## Most popular preferred skills per job category

# In[ ]:


popular_categories = df.job_category.value_counts()[:5]
popular_categories


# In[ ]:


preferred_skills(popular_categories.index[0]).iplot(title='engineering', kind='bar', color='khaki')


# In[ ]:


preferred_skills(popular_categories.index[1]).iplot(title='technology', kind='bar', color='deepskyblue')


# In[ ]:


preferred_skills(popular_categories.index[2]).iplot(title='public safety', kind='bar', color='green')


# In[ ]:


preferred_skills(popular_categories.index[3]).iplot(title='health', kind='bar', color='powderblue')


# In[ ]:


preferred_skills(popular_categories.index[4]).iplot(title='legal affairs', kind='bar', color='darkolivegreen')


# ## Minimum Qualification Wordcloud

# In[ ]:


qualification_keywords = []
df.minimum_requirements.dropna().apply(parse_keywords, l=qualification_keywords)
plt.figure(figsize=(10, 8))
counter = Counter(qualification_keywords)
common = [x[0] for x in counter.most_common(40)]
plot_wordcloud(' '.join(common))

