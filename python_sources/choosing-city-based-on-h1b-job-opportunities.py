#!/usr/bin/env python
# coding: utf-8

# **Background**
# 
# For new immigrants, especially newly married couples, selecting where to settle down can be a difficult decision. Assuming the two are in different majors, chances are their ideal city with the most affluent career opportunities also differ: 
# 
# A Mechanical Engineer who wants to work for an automaker may consider the greater Detroit area (or silicon valley in recent years) to be ideal, while a Statistician aspiring to work for investment banks on Wall Street may not agree.
# 
# **Questions Expected to be Answered**
# 
# 1. If the couple are still in the stage of choosing a major, what kind of major would yield the best return?
# 2. What are the leading companies in those fields?
# 3. Where are they located?
# 4. Given choices made by the two/degrees already completed, what cities would be the best to ensure both will be happy to reach their respective career goals?
# 
# **In Scope**
# 
# The only goal is to optimize the chance of finding satisfying career opportunities based on past statistical data.
# 
# **Out of Scope**
# 
# Factors such as housing market, safety, school district, etc. are not part of the consideration.
# 
# **Assumption**
# 
# 1. Trends shown by the past years' data shall continue for the next 5-10 years;
# 2. The prevalent wage as published is reflective of the true salary of the position;
# 3. All positions listed can be considered a profession that demand is greater than supply.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datashader as ds
from datashader import transfer_functions as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/h1b_kaggle.csv')
df.head()


# We now have a understanding of the content and format of the data. I'll mainly try to figure out what profession yields the best return, the leading companies in those fields, and their locations. For couples, the locations ideally should have high paying jobs for both parties' interests and career goals. It seems like the info contained in this dataset is sufficient for us to make the estimations.

# In[ ]:


df.describe()


# Now that we know the format of the data, I'd like to get a quick overview as shown below:
# 1. There are 3M samples, thus not too big a dataset, but we'll still try some newer plotting tools;
# 2. The data covers 2011-2016 statistics, which is fairly up to date, and can provide good guidance for us in 2018;
# 3. The longitudinal coordinates are beyond the range of [-124.7844079, -66.9513812], which means the info covers states beyond the [continental US](https://en.wikipedia.org/wiki/List_of_extreme_points_of_the_United_States). We'll need to keep this in mind when we plot:
# ```
# x_range, y_range = ((-124.7844079, -66.9513812), (24.7433195, 49.3457868))
# ```
# 4. There is a job that pays a salary of 7e9 dollars? We'll need to find out more about that one later on for sure!
# 5. For this analysis, jobs paying less than $30,000 will not be considered. And we'll only consider those petitions with a Certified case status to improve credibility of the data:

# In[ ]:


df = df[(df['PREVAILING_WAGE']>30000) & (df['CASE_STATUS']=='CERTIFIED')]
df.describe()


# Sadly it seems the petition with the 7 billion dollar wage wasn't certified, but $300M still sounds good!
# 
# We will be using the subset of data as shown above for our analysis.

# In[ ]:


US = x_range, y_range = ((-124.7844079, -66.9513812), (24.7433195, 49.3457868))
plot_width  = int(900)
plot_height = int(plot_width//2)

cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)
agg = cvs.points(df, 'lon', 'lat', ds.mean('PREVAILING_WAGE'))
img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='eq_hist')
img


# At a quick glance, California, New York, and Great Lakes areas seem to be concentrated with high paying opportunities.
# 
# Now let's get down to better understanding of the professions.

# In[ ]:


pop_profession = df['SOC_NAME'].describe()
pop_profession


# The most popular profession is Computer Systems Analyst, which is around 10% of all H1B applications from 2011 to 2016!
# 
# If the number of petitions is a direct reflection of the demand, or the potential quantity of opportunities in the job market, let's get an overview of the top 20 among the 1883 professions listed.

# In[ ]:


df['SOC_NAME'].value_counts().head(20).plot(kind='bar')


# Programmers or software developers dominate!
# 
# Now let's take a quick look at the two ends of the data based on wage:

# In[ ]:


df_sort_wage = df.sort_values(by=['PREVAILING_WAGE'])
df_sort_wage.head(10)


# In[ ]:


df_sort_wage.tail(10)


# We just solved the mystery of the job with the highest pay : )
# 
# At the same time, notice that there are several SDEs in the Top 10 as well!
# 
# So let's get an overview of the professionals that earned Top 50 salaries among all:

# In[ ]:


df_sort_wage.tail(50).describe()


# In[ ]:


df_sort_wage['SOC_NAME'].tail(50).value_counts().plot(kind='bar')


# I was excited to see multiple elementary school and preschool teachers, as well as TAs, made the list. Educators shape the future, and need to be better compensated to attract and retain talents!
# 
# To be continued...

# In[ ]:




