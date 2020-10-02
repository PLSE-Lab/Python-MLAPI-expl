#!/usr/bin/env python
# coding: utf-8

# ![](https://about.udemy.com/wp-content/uploads/2016/07/about-default.png)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['figure.figsize'] = 8, 5
pd.options.mode.chained_assignment = None 
pd.set_option('display.max_columns',None)
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
df = df.apply(lambda x: x.astype(str).str.lower())
df.head()


# In[ ]:


df.shape


# So we have 3683 rows and 12 columns in this dataset.

# In[ ]:


df.describe()


# In[ ]:


#missing data
df.isnull().sum().sort_values(ascending=False)


# We don't have any missing values in this dataset.

# **Paid Courses vs Free Courses**

# In[ ]:


course_fee = df.groupby('is_paid')['course_title'].count().reset_index().sort_values('course_title',ascending = False)
course_fee = course_fee[course_fee['is_paid'].isin(['true','false'])]
course_fee = course_fee.rename(columns = {'course_title':'count'})
fig = px.bar(course_fee, x='is_paid', y='count', color='count')
fig.show()


# Majority of the courses offered on Udemy are Paid Courses.

# **Difficulty Level of Courses**

# In[ ]:


difficulty_level = df.groupby('level')['course_title'].count().reset_index().sort_values('course_title',ascending = False)
difficulty_level = difficulty_level[difficulty_level['level'].isin(['all levels','beginner level','expert level','intermediate level'])]
difficulty_level = difficulty_level.rename(columns = {'course_title':'count'})
fig = px.bar(difficulty_level, x='level', y='count', color='count')
fig.show()


# As we that most of the courses are of all levels. As the difficulty level of the course increases, Number of courses offered are less.

# In[ ]:


subject = df.groupby('subject')['course_title'].count().reset_index().sort_values('course_title',ascending = False)
subject = subject.rename(columns = {'course_title':'count'})
fig = px.bar(subject, x='subject', y='count', color='count')
fig.show()


# **Most Popular Courses - Number Of Subscribers**

# In[ ]:


subject =df[['course_title','num_subscribers']]
subject =subject.sort_values('num_subscribers',ascending = False).head(10)
fig = px.bar(subject, x='course_title', y='num_subscribers', color='num_subscribers')
fig.show()


# **Bivariate Analysis**

# In[ ]:


df_price = df[~df['price'].isin(['true','free'])]
sns.distplot(df_price['price']);


# We can see that the price column has a positive tail.

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_price['price'].skew())
print("Kurtosis: %f" % df_price['price'].kurt())


# Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point. Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers.
