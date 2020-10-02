#!/usr/bin/env python
# coding: utf-8

# This notebook contains data analysis on the Socialblade's data of top 5000 Youtube Channels. I have mainly relied only on Exploratory Data Analysis in this project.
# The main motives of doing this project:
# * I am quite comfortable in doing predictive and mathematical analysis on various datasets, but felt the need to practice the art of  Exploratory Data Analysis. Hence I have relied only on it in this project.
# * The current hype surrounding Youtube also caused curiosity within me as to how the various features affect the number of subscribers of a channel and in general, how these features interact with each other.
# * I have tried including various statistical concepts in the project.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/data.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# The describe() method just gives one column, the 'Video Uploads' and 'Subscribers' are have 'object' datatype. Let's convert them to numeric value for better analysis. For that let's observe these series indiviually.

# In[ ]:


df['Subscribers'].value_counts()


# The value with highest count is obviously a missing values.Let's skip the rows with these missing values as just filling them with the mean would lead to less variance in the data.

# In[ ]:


missing_val = df['Subscribers'].value_counts().index[0]
df_1 = df[df['Subscribers'] != missing_val]
df_1.info()


# In[ ]:


df_1['Subscribers'] = pd.to_numeric(df_1['Subscribers'])
df_1.describe()


# Now let's convert 'Video Uploads' to numeric values.

# In[ ]:


vid_upload_no_miss = pd.to_numeric(df_1['Video Uploads'], errors = 'coerce')
vid_upload_no_miss


# In[ ]:


vid_upload_no_miss_fill = vid_upload_no_miss.fillna(value = -1)
vid_upload_no_miss_fill.describe()


# In[ ]:


df_2 = df_1.drop('Video Uploads', axis = 1)
df_3 = pd.concat([df_2, vid_upload_no_miss_fill], axis = 1)
df_3 = df_3[df_3['Video Uploads'] != -1]
df_3.describe()


# In[ ]:


from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize = (20, 20))
df_3.hist(bins = 50, ax = ax)


# Thus the distribution is quite skewed, many channels (>2000) in the dataset have < 1.25e6 subscribers, majority of the channels have video uploads much less than 100000, and also <3000 channels have views much less than 1e10.

# In[ ]:


df_3.head()


# The top 50 subscribed channels on Youtube are:

# In[ ]:


import seaborn as sns
df_subs_and_channels_top_50 = df_3[['Channel name', 'Subscribers']].sort_values('Subscribers', ascending = False)[:50]
fig, ax = plt.subplots(figsize = (12.5, 12.5))
sns.barplot(y = 'Channel name', x = 'Subscribers', data = df_subs_and_channels_top_50, orient = 'h',ax = ax)


# Top 50 channels with most views are:

# In[ ]:


df_views_and_channels_top_50 = df_3[['Channel name', 'Video views']].sort_values('Video views', ascending = False)[:50]
fig, ax = plt.subplots(figsize = (12.5, 12.5))
sns.barplot(y = 'Channel name', x = 'Video views', data = df_views_and_channels_top_50, orient = 'h',ax = ax)


# Top 50 channels with most videos uploaded are:

# In[ ]:


df_views_and_channels_top_50 = df_3[['Channel name', 'Video Uploads']].sort_values('Video Uploads', ascending = False)[:50]
fig, ax = plt.subplots(figsize = (12.5, 12.5))
sns.barplot(y = 'Channel name', x = 'Video Uploads', data = df_views_and_channels_top_50, orient = 'h',ax = ax)


# The distribution of Grades is as follows,

# In[ ]:


import seaborn as sns
sns.countplot(df_3['Grade'])


# Thus more than half channels have B+ grade.

# Now, let's see the relationship between the rank and grade of a channel.

# In[ ]:


df_rank_grade = df_3[['Rank', 'Grade']]


# Let's observe the rank distribution of the  df_rank_grade dataframe. Let's convert the Rank series of 'df_rank_grade' to int so analysis becomes easier.

# Let's modify the 'Rank' Series.

# In[ ]:


l = list(df_rank_grade['Rank'])
rank_list = [i.split(',')[0] + i.split(',')[1] if i[1] == ',' else i[:-2] for i in l]
final_list = [i[:-2] if len(i) == 6 else i for i in rank_list]
int_rank_series = pd.DataFrame(final_list, columns = ['Rank'], dtype = int)
df_rank_grade = df_rank_grade.drop('Rank', axis = 1)
df_rank_grade_new = pd.concat([df_rank_grade, int_rank_series], axis = 1)
df_rank_grade_new.head()


# Let's see the relationship between the columns via a boxplot,

# In[ ]:


fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(x = 'Grade', y = 'Rank', data = df_rank_grade_new, ax = ax)


# It can be clearly seen that there is a clear demarcation between various grades of channels based on their ranks, thus, the ranks of channels are very good indicators of grade of a channel.

# From the boxplot, median Rank of a B+ grade channel is about 3600. That of A- channel is ~1700, that of A grade channel is ~700. For A+ channels it is about 70-100, while for A++ channel,

# In[ ]:


df_rank_grade_new[df_rank_grade_new['Grade'] == 'A++ ']


# Only top 10 channels have A++ grade.

# Let's see the relationship between Grade and Subscribers,

# In[ ]:


fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(x = 'Grade', y = 'Subscribers', data = df_3)   


# There is a substantial amount of overlap between the boxes of some of the grades, also there are a lot of  outliers, thus the number of subscribers is not a very good indicator of the Grade of the channel.

# Let's compare the number of subscribers with the Rank of the channels,

# In[ ]:


df_3['Subscribers'].plot()
plt.xlabel('Rank of the channel.')
plt.ylabel('Number of Subscribers.')


# It is evident from the plot that the channels are not ranked according to the number of subscribers.

# Let's compare the Video Uploads with the Rank.

# In[ ]:


df_3['Video Uploads'].plot()
plt.xlabel('Rank of the channel.')
plt.ylabel('Number of uploaded videos.')


# Thus, the ranking of the channels also does not depend upon video uploads.

# Let's compare Video Uploads with grade of the channel.

# In[ ]:


sns.boxplot(x = 'Grade', y = 'Video Uploads', data = df_3)


# Thus, the distribution of Video Uploads with respect to the Grade of the channels is not quite insightful. 

# Let's comapre Video views with the rank.

# In[ ]:


df_3['Video views'].plot()
plt.xlabel('Rank')
plt.ylabel('Total number of views')


# Video views seems to influence the rank of the channel (barring a few exceptions where there are spikes in the plot).

# Comparing Grade of the channel to the VIdeo views,

# In[ ]:


sns.boxplot(x = 'Grade', y = 'Video views', data = df_3)


# From the boxplot, it can be seen that higher the number of video views, better is the grade (roughly).

# Now, let's make comparisons among continous variables,

# In[ ]:


sns.scatterplot(x = 'Video views', y = 'Subscribers', data = df_3)


# There is a clear positive trend between the two variables, i.e more video views generally have positive correlation with number of subscribers. Also the relationship seems to be linear.

# Confirming it mathematically,

# In[ ]:


corr_coef = np.corrcoef(x = df_3['Video views'], y = df_3['Subscribers'])
corr_coef


# Hence, a very strong positive correlation is verified.

# Comparing Video Uploads with the subscribers,

# In[ ]:


sns.scatterplot(x = 'Video Uploads', y = 'Subscribers', data = df_3)


# This relationship is very different from the one above, generally subscribers tend to decrease for the channels having very large number of videos uploaded, while for very low number of videos uploaded, there is a large variance. There is no strong linear relationship between the two variables though.

# Finding out the correlation mathematically,

# In[ ]:


corr_coef = np.corrcoef(x = df_3['Video Uploads'], y = df_3['Subscribers'])
corr_coef


# Thus, weak correlation is confirmed mathematically,

# Comapring Video Uploads and Video Views with respect to subscribers,

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 10))
sns.scatterplot(x = 'Video Uploads', y = 'Video views', size = 'Subscribers', data = df_3, ax = ax)


# Again the relationship between the two variables is not strongly linear.

# Let's create a new feature "Views per Upload". This feature might help in better analysis.

# In[ ]:


df_3['Views per Upload'] = df_3['Video views'] / df_3['Video Uploads']
df_3['Views per Upload'].head()


# Let's compare the 'Views per Upload' with the number of subscribers,

# In[ ]:


sns.scatterplot(x = 'Views per Upload', y = 'Subscribers', data = df_3)


# Thus, this feature seems to have a more strong relationship with the number of Subscribers than Video Uploads.Let's verify mathematically,

# In[ ]:


corr_coef = np.corrcoef(x = df_3['Views per Upload'], y = df_3['Subscribers'])
corr_coef


# Thus the newly made feature has a better correlation with Subscribers than Video Uploads. Hence the new feature can be very useful to predict number of subscribers.

# Thus the **conclusions** from the above analysis are:
# 1. There is a clear demarcation between various grades of channels based on their ranks, thus, the ranks of channels are very good indicators of grade of a channel.
# 2. The number of subscribers is not a very good indicator of the Grade of the channel.
# 3. Video views seems to influence the rank of the channel (although there are a few exceptions).
# 4. Higher the number of video views, higher is the grade (roughly).
# 5. There is a clear positive trend between the two variables, i.e more video views generally have strong positive correlation with number of subscribers.
# 6.  Generally number of subscribers are low for the channels having very large number of videos uploaded, while for channels having very low number of videos uploaded, there is a large variance. There is no strong linear relationship between the two variables though.
# 7. Video Uploads per view feature has a more strong relationship with the number of Subscribers than Video Uploads.The same is suggested mathematically too. Hence is a better feature for predicting number of subscribers of a channel.
