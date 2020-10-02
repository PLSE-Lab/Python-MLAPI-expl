#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ks_data = pd.read_csv ("../input/kickstarter-campaigns/Kickstarter_projects_Feb19.csv")


# In[ ]:


#checking null values
ks_data.isnull().sum()


# In[ ]:


#leave only date in launched column
ks_data["launched_at"]= ks_data["launched_at"].str.split(" ", n = 1, expand = True)
ks_data["deadline"]= ks_data["deadline"].str.split(" ", n = 1, expand = True) 


# In[ ]:


#change dates format to date
ks_data['deadline'] = pd.to_datetime(ks_data['deadline'])
ks_data['launched_at'] = pd.to_datetime(ks_data['launched_at'])
ks_data['year'] = ks_data['launched_at'].dt.year


# In[ ]:


ks_data['duration'] = pd.to_numeric(ks_data['duration'], downcast='float')
ks_data['usd_pledged'] = pd.to_numeric(ks_data['usd_pledged'], downcast='float')
ks_data.info()


# In[ ]:


#round goal_usd and usd_pledged values to 3 decimals 
ks_data['goal_usd'] = (ks_data.goal_usd).round(3)
ks_data['usd_pledged'] = (ks_data.usd_pledged).round(3)


# In[ ]:


#checking duplicates
ks_data.duplicated().sum()


# In[ ]:


#let's delete duplicates 
ks_data.drop_duplicates(keep='first', inplace=True)


# In[ ]:


#to check that duplicates are deleted
ks_data.shape


# In[ ]:


most_pledged_status = ks_data[ks_data['status'] == 'successful']
most_pledged = most_pledged_status.groupby('main_category')['usd_pledged'].sum().to_frame().reset_index().sort_values('usd_pledged', ascending = False).head(30).set_index('main_category')
most_pledged.head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(most_pledged['usd_pledged'], most_pledged.index, palette='deep')
plt.title("Top 30 campaigns with the highest pledge by category", fontsize=18)
plt.xlabel("Amount of money pledged")
plt.ylabel("Categories of the campaign")
plt.xticks([10000000, 100000000, 250000000, 500000000, 750000000],['10m', '100m', '250m', '500m', '750m'])
plt.show()


# *As there are so many campaigns in the US, let's focus only at non-US countries, so we could draw the conclution about other countries. And exclude 2019, as our dataset contains only first 2 months on 2019.*

# In[ ]:


only_success_status = ks_data[ks_data['status'] == 'successful']
only_success_status_no_US = only_success_status[only_success_status['country'] != 'US']
most_pledged_cat_no_US = only_success_status_no_US.groupby('main_category')['usd_pledged'].sum().to_frame().reset_index().sort_values('usd_pledged', ascending = False).head(30).set_index('main_category')
most_pledged_cat_no_US.head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(most_pledged_cat_no_US['usd_pledged'], most_pledged_cat_no_US.index, palette='deep')
plt.title("Top 30 campaigns with the highest pledge by category without US", fontsize=18)
plt.xlabel("Amount of money pledged")
plt.ylabel("Categories of the campaign")
plt.xticks([1000000, 25000000, 50000000, 75000000, 100000000, 125000000],['1m', '25m', '50m', '75m', '100m', '125m'])
plt.show()


# *It looks like top 4 leading categories are common for the dataset with the US and other countries.*

# In[ ]:


kick_data_non_US = ks_data[(ks_data['year'] != 2019) & (ks_data.country != 'US')]
kick_data_non_US = kick_data_non_US.groupby(['main_category', 'status'])['name'].count().reset_index(name='count')
kick_data_non_US = kick_data_non_US.sort_values('count', ascending = False).set_index('main_category')
kick_data_non_US.head()


# In[ ]:


plt.figure(dpi=120)
sns.barplot(kick_data_non_US.index, kick_data_non_US['count'],
            hue= kick_data_non_US['status'], palette = "Dark2")
plt.xticks(rotation = 90)
plt.title("Successful and failed projects per category non-US", fontsize=12)
plt.xlabel("Categories of the projects")
plt.ylabel("Amount of projects per category")
plt.show()


# *Technology is one of the categories that has more failed projects than successful, also food, photography and jornalism. On the other hand, film & video, music and publishing have more successful projects.*

# In[ ]:


#Goal and amount pledged per category (for non-US)
kick_data_goal_pledge = ks_data[(ks_data['year'] != 2019) & (ks_data.country != 'US') & (ks_data.status == 'successful')]


# In[ ]:


plt.figure(dpi=120)
sns.lineplot(data=kick_data_goal_pledge, x= kick_data_goal_pledge.main_category, y=kick_data_goal_pledge.goal_usd)
sns.lineplot(data=kick_data_goal_pledge, x= kick_data_goal_pledge.main_category, y=kick_data_goal_pledge.usd_pledged)
plt.title("USD goal and pledged per category among successful projects (non-US)", fontsize=12)
plt.xlabel("Categories of the projects")
plt.ylabel("Amount of money (USD)")
plt.xticks(rotation = 90)
plt.legend(['goal', 'pledged'], loc='upper left')
plt.show()


# *When we look at only succesful projects per category, we can see that design, games and technology are pledging in average  much more money, that their goal and than other categories.*

# In[ ]:


#success rate vs duration
kick_data_no_US = ks_data[(ks_data['country'] != 'US') & (ks_data['year'] != 2019) & (ks_data['duration'] <= 61)]
project_duration = kick_data_no_US.groupby(['duration','status'])['duration'].count().reset_index(name='count')
project_duration["sum_counts"] = project_duration.groupby('duration')['count'].transform('sum')
project_duration["pct"] = project_duration['count']/project_duration["sum_counts"]
project_duration = project_duration[project_duration['status'] == 'successful']
project_duration.head()


# In[ ]:


plt.figure(dpi=120)
plt.style.use('dark_background')
sns.lmplot(data=project_duration, x="duration", y="pct")
plt.title("Success Rate vs duration of campaign", fontsize=12)
plt.xlabel("Duration of campaign (days)")
plt.ylabel("Success Rate (%)")
plt.show()


# *We can see the negative correlation between success rate and duration of campaign.
# But as a lot of projects are only asking for small amount of money, which is easy to get for short duration, let's exclude them from our data. And check the correlation again.*

# In[ ]:


#success rate vs duration for projects with the goal higher than 1000 USD
kick_data_no_US = ks_data[(ks_data['country'] != 'US') & (ks_data['year'] != 2019) & (ks_data['duration'] <= 61) & (ks_data['goal_usd'] >= 1000)]
project_duration = kick_data_no_US.groupby(['duration','status'])['duration'].count().reset_index(name='count')
project_duration["sum_counts"] = project_duration.groupby('duration')['count'].transform('sum')
project_duration["pct"] = project_duration['count']/project_duration["sum_counts"]
project_duration = project_duration[project_duration['status'] == 'successful']
project_duration.head()


# In[ ]:


plt.figure(dpi=120)
plt.style.use('dark_background')
sns.lmplot(data=project_duration, x="duration", y="pct")
plt.title("Success Rate vs duration of campaign", fontsize=12)
plt.xlabel("Duration of campaign (days)")
plt.ylabel("Success Rate (%)")
plt.show()


# In[ ]:


#adding day of the week and month
ks_data["launched_day"] = ks_data['launched_at'].dt.weekday_name
ks_data["launched_month"] = ks_data['launched_at'].dt.month_name()


# In[ ]:


kick_data_per_weekday_no_US = ks_data[(ks_data['year'] != 2019) & (ks_data.country != 'US')]
kick_data_per_weekday = kick_data_per_weekday_no_US.groupby(['status','launched_day'])['name'].count().reset_index(name='count')
kick_data_per_weekday_sorted = kick_data_per_weekday.sort_values('count', ascending = False).set_index('launched_day')


# In[ ]:


plt.figure(dpi=120)
sns.barplot(kick_data_per_weekday_sorted.index, kick_data_per_weekday_sorted['count'],
            hue= kick_data_per_weekday_sorted['status'], palette = "gist_rainbow")
plt.xticks(rotation = 45)
plt.title("Successful and failed projects by day of launch", fontsize=12)
plt.xlabel("Launched at (weekday)")
plt.ylabel("Amount of projects")
plt.show()


# *Does this mean, that we should start our campaign on Tuesday?*

# In[ ]:


kick_data_per_month_no_US = ks_data[(ks_data['year'] != 2019) & (ks_data.country != 'US')]
kick_data_per_month = kick_data_per_month_no_US.groupby(['status','launched_month'])['name'].count().reset_index(name='count')
kick_data_per_month_sorted = kick_data_per_month.sort_values('count', ascending = False).set_index('launched_month')


# In[ ]:


plt.figure(dpi=120)
sns.barplot(kick_data_per_month_sorted.index, kick_data_per_month_sorted['count'],
            hue= kick_data_per_month_sorted['status'], palette = "CMRmap")
plt.xticks(rotation = 90)
plt.title("Successful and failed projects by month of launch", fontsize=12)
plt.xlabel("Launched at (month)")
plt.ylabel("Amount of projects")
plt.show()


# *Does this mean, that we should start our campaign on November? Or...*
# 
# *Let's look as a success rate per month*

# In[ ]:


#success rate vs month of launch
kick_data_no_US = ks_data[(ks_data.country != 'US') & (ks_data['year'] != 2019)]
project_launch_month = kick_data_no_US.groupby(['launched_month','status'])['name'].count().reset_index(name='count')
project_launch_month["sum_counts"] = project_launch_month.groupby('launched_month')['count'].transform('sum')
project_launch_month["pct"] = project_launch_month['count']/project_launch_month["sum_counts"]
project_launch_month = project_launch_month[project_launch_month['status'] == 'successful']
project_launch_month_sorted = project_launch_month.sort_values('pct', ascending = False).set_index('launched_month')
project_launch_month.head()


# In[ ]:


plt.figure(dpi=120)
sns.barplot(project_launch_month_sorted.index, project_launch_month_sorted['pct'], palette = "gnuplot")
plt.xticks(rotation = 90)
plt.title("Success rate of projects by month of launch", fontsize=12)
plt.xlabel("Launched at (month)")
plt.ylabel("Success Rate (%)")
plt.show()


# *So it looks like March has the highest successful rate, but in general the difference between months is too small for drawing conclusions.*
