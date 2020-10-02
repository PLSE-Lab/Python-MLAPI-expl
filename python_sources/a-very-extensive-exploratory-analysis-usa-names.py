#!/usr/bin/env python
# coding: utf-8

# ![US Name](https://www.citi.io/wp-content/uploads/2016/01/1607-00.png)

# **INDEX**
# * Loading Libraries
# * Loading and Converting Data into CSV
# * Peek into Data
# * EDA
#     1. Number of Females and Males
#     2. Word Cloud of Names
#     3. Top 5 States having most Applicants Ever
#     4. Number of Applicants per year
#     5. The most common names ever
#     6. The most common Male names ever
#     7. The most common Female names ever
#     8. Change in rate of Male Applicants over the years
#     9. Change in rate of Female Applicants over the years
#     10. Male/Female distribution over the years
# 

# **Loading Libraries**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
import os
# Any results you write to the current directory are saved as output.


# ** Loading and Converting Data into CSV**

# In[2]:


# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names.csv")


# **Peek into Data**

# In[3]:


agg_names.head()


# In[4]:


agg_names.shape


# In[5]:


pd.options.display.max_rows = 4000


# **1. Number of Females and Males **

# In[6]:


agg_names.groupby('gender')['gender'].count().plot.bar()


# In[7]:


temp = agg_names['gender'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')


# ** 2. Word Cloud of Names   **

# In[8]:


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                       width=600, height=300
                     ).generate(" ".join(agg_names['name'].sample(2000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Names", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# **  3. Top 5 States having most Applicants Ever**

# In[9]:


agg_names.groupby('number')['number'].count().head().plot.bar()


# In[10]:


temp = agg_names['number'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')


# ** 4. Number of Applicants per year**

# In[11]:


temp = agg_names.groupby('year')['year'].count()
ye = temp
temp


# In[12]:


q1=temp.plot.line(x=temp.index,y=temp.values,figsize=(12,3),lw=7)
q1.set_ylabel("Number of Applicants")


# **5. The most common names ever**

# In[13]:


agg_names['name'].value_counts().head()


# **6. The most common Male name ever**

# In[14]:


temp = agg_names.groupby('gender')['name'].value_counts()['M'].head()
temp


# **7. The most common Female names ever**

# In[15]:


temp = agg_names.groupby('gender')['name'].value_counts()['F'].head()
temp


# **8. Change in rate of Male Applicants over the years**

# In[16]:


temp = agg_names.groupby('year')['gender'].value_counts()
temp = temp.loc[1::2]
mval = ye.values - temp.values
mval


# In[17]:


q2=temp.plot.line(x=ye.index,y=mval,figsize=(12,3),lw=7,color="purple")
q2.set_ylabel("Number of Male Applicants")


# **9. Change in rate of Female Applicants over the years**

# In[18]:


temp = agg_names.groupby('year')['gender'].value_counts()
temp = temp.loc[1::2]
temp.values


# In[19]:


q2=temp.plot.line(x=ye.index,y=temp.values,figsize=(12,3),lw=7,color='orange')
q2.set_ylabel("Number of Female Applicants")


# **10. Male/Female distribution over the years**

# In[21]:


temp = agg_names.groupby('year')['gender'].value_counts()
temp


# **Stay Tuned and PLEASE, votes up! I will update this Kernel soon.**
