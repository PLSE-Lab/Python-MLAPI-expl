#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# ### Hi there,
# this notebook is about the restaurant inspection data of the city of San Francisco.
# 
# I will show several visualizations about scores, risks and location and their dependencies.
# Furthermore and in the beginning, I will have a look at the time dependencies.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('List of datasets: ',os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# # Let's get started...<a id="1"></a>
# ---------------------------------------

# In[ ]:


df = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
df.head()


# In[ ]:


df.info()


# # Cleaning and Transformations <a id="2"></a>
# ---------------

# In[ ]:


# clean data, delete columns
df.drop(columns=['business_city', 'business_state', 'business_postal_code', 'business_location', 'business_address', 'business_phone_number'],inplace=True)


# In[ ]:


df['inspection_type'].unique()

# note there are many different types of inspections.
# For some reason, the "Foodborne Illness Investigration" might be of high interest ...


# In[ ]:


# convert time to datetime of pandas
df['inspection_date'] = pd.to_datetime(df['inspection_date'])
df['inspection_month_day'] = df['inspection_date'].map(lambda x: x.strftime('%m-%d'))
df['inspection_year'] = df['inspection_date'].apply(lambda x: x.year)
df['inspection_month'] = df['inspection_date'].apply(lambda x: x.month)
# df['inspection_year_month'] = pd.to_datetime(df['inspection_date'], format='%Y%m')
df['inspection_dayofweek'] = df['inspection_date'].apply(lambda x: x.weekday())


# # Analysis <a id="3"></a>
# -------------------

# ## What changed over time and why should you be more picky when it comes to choosing a restaurant!

# In[ ]:


# is there any day of the week where a surprising inspection is more likely to happen
sns.catplot(x="inspection_dayofweek", data=df[df['inspection_type']=='Routine - Unscheduled'], kind="count",  height=4, aspect=1.5);
plt.title('Number of Unscheduled Inspections for week days')
plt.xticks(range(7),['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',' Saturday',' Sunday']);


# Even at the weekend, some inspectors are active! Watch out! ;)
# 
# But let's face the truth, most of the time, the chart looks very much like a motivation of normal worker changing during the week:
# 
# **Monday:** The batteries are loaded but some still have a hangover or you are mentally still at the weekend.
# 
# **Thursday and Wednesday:** You can fully concentrate on your work. That is the peek of motivation.
# 
# **Tuesday and Friday:** The previous days were quite exhausiting...you start looking forward to the great social event coming on friday night and lose focus on what should really matter (if it would be up to your employer).
# 
# **Weekend:** No worries you restaurant owner. The only time you will see an inspector is when he is visiting your resturant as guest. 

# In[ ]:


sns.catplot(x="inspection_year", 
            data=df, 
            kind="count",  height=4, aspect=1.5,sharey=False);
plt.title('Overall number of inspections per year')

sns.catplot(x="inspection_year", 
            col='inspection_type',
            data=df[(df['inspection_type']=='Routine - Scheduled') | (df['inspection_type']=='Routine - Unscheduled')| (df['inspection_type']=='Foodborne Illness Investigation')], 
            kind="count",  height=4, aspect=1,sharey=False);


# Let's wrap this up:
# 
# In 2015, only very few inspection have been done. Most likely the data collection just started duirng the last weeks.
# 
# During the next years, the overall number of inspections did not change much but decreased a bit (probably it is just because the data of december is not complete, yet). Looking at specific types of inspection, a more drastic pattern is being visible.
# 
# The *schedules routine* inspections **strongly decreased** over the last years, while the unscheduled inspections decreased less strong. It seems that while the overall inspections decreased, the focus went more into the unscheduled inspections.
# 
# More impressive are the *Foodborne Illness Investigations*. The number of inspections is more than **4x higher** in 2018 compared to 2016. Very very worrying!
# 

# ## Inspection Activity
# The question arises to me, if there are specific month or times of the year, were inspectors are more active. Maybe before specific days like Thanks Giving or Christmas?

# In[ ]:


# is there any time of the year where there are more inspections?
sns.catplot(x="inspection_month",
            row='inspection_type', 
            col='inspection_year', 
            data=df[(df['inspection_type']=='Routine - Scheduled') | (df['inspection_type']=='Routine - Unscheduled')| (df['inspection_type']=='Foodborne Illness Investigation')], 
            kind="count",  height=4, aspect=1.5,sharey=False);


# There seems to be no trend or any pattern to the monthly activity. Maybe it is better visible when looking at the specific days. But stick to the *Unscheduled Routine Inspections* for the ease of simplicity.

# In[ ]:


df_tmp = df[(df['inspection_type']=='Routine - Unscheduled')].groupby('inspection_month_day').count()
fig, ax = plt.subplots(figsize=(30,4))
sns.barplot(x=df_tmp.index,
            y='inspection_year',
            data=df_tmp,
            ax=ax, 
            orient='v');
plt.setp(ax.get_xticklabels(), rotation=90);


# Ok, there is no special day with an increasing inspection activity. 
# 
# 

# In[ ]:


gb = df.groupby('business_id')


# ## Risk managment
# The question might arise: What makes a violation a *High Risk* or *Low Risk* one.
# 
# This question is answered with the following graphs. For every risk-type, the number of violations with the description is found. 
# 
# So e.g. if you don't store your food cool and clean, you are in danger to get a high risk state!
# 

# In[ ]:


sns.catplot(y="violation_description",
            row='risk_category', 
            data=df, kind="count",  
            height=12, aspect=1.5,
#             hue_order=['Low Risk', 'Moderate Risk', 'High Risk'],
            order = df['violation_description'].value_counts().index,
            orient='v');


# Alright, with that insight, we are now able to say by observation, which restaurant is risky to eat or not.
# 
# But if you may only have the restaurant overall score, you can only hope that it correlates with the risk category? Or can you be sure to choose wisely when only considering the score?

# In[ ]:


sns.boxplot(x = 'risk_category',
            y = 'inspection_score',
            data=df,
            order = ['Low Risk','Moderate Risk', 'High Risk'],
            );
plt.title('Score per risk class');


# There is a correlation. Lower scores means in average a higher risk and thereby a less healthy restaurant.
# 
# But there is obviously a high overlap between the risk classes. I would not count the score as a strong indicator for a reliable risk forecast. Let's go a bit more  into detail and look how the scores are disctributed for the different risk types.

# In[ ]:


sns.catplot(y="inspection_score",
            col='risk_category', 
            data=df, kind="count",  
            height=8, aspect=1,sharey=False, 
            orient='v',
           );


# In[ ]:


# nbr_businesses = df.groupby('business_name')['business_id'].count()
# nbr_businesses.values
# sns.barplot(y = nbr_businesses.index,x = 'business_id',data = nbr_businesses)
fig, ax = plt.subplots(figsize=(30,6))
sns.countplot(ax=ax, 
              y='business_name', 
              data=df, 
              order = df['business_name'].value_counts().iloc[:10].index,
              );
plt.title('Businesses with the most inspections');


# In[ ]:


# number of inspections to average score
df_unscheduled = df[(df['inspection_type']=='Routine - Unscheduled')]
gb_count = df_unscheduled.groupby('business_id')['business_name'].count()
gb_mean  = df_unscheduled.groupby('business_id')['inspection_score'].mean()

df_id = pd.DataFrame()
df_id['business_id'] = gb_count.index
df_id['inspection_score'] = gb_mean
df_id['inspection_count'] = gb_count

df_id.dropna(inplace=True)
sns.regplot(x='inspection_score', y='inspection_count', data=df_id, scatter_kws={'alpha':0.2});
plt.title('Inspection -  Score vs Count')


# Now, we see a correlation between the number of inspections and the corresponding average score. We are focusing on the *Unschedules Inspections*, but the results look very similar with all inspections together since the majority of inspections are unscheduled.

# In[ ]:


df_unscheduled = pd.concat([df_unscheduled, pd.get_dummies(df_unscheduled['risk_category'])], axis=1)


# In[ ]:


gb_risk = df_unscheduled.groupby('business_id')['Low Risk','Moderate Risk', 'High Risk'].sum()
gb_risk['most_severe_risk'] = gb_risk.apply(lambda x: np.argmax(x), axis=1)

df_concat = pd.concat([df_id,gb_risk['most_severe_risk']], axis=1)
df_concat.dropna(inplace=True)


# In[ ]:


fig, ax = plt.subplots(figsize=(5,7))
sns.scatterplot(x='inspection_score', 
                y='inspection_count', 
                data=df_concat, 
                alpha=0.7, 
                size='most_severe_risk',
                size_order = ['High Risk','Moderate Risk','Low Risk'],
                legend='full', 
                ax=ax);
plt.title('Inspection -  Score vs Count');


# This visualization looks a bit better then the one before. We can also see  for example the influence of the combination of risk and inspection_count on the inspection_score.
# 
# Even if the number of scores are quite low, the score can be low if the risk is high!

# ## You got mapped!
# 
# The map below shows you the restaurants with the highest average score (>98). If you really want to make sure to get out with a good feeling and a belly full of healthy food, pick one of these locations with a low risk of issues found during inspections (green).

# In[ ]:


df_formap = pd.concat([df_concat, df_unscheduled[['business_name','business_latitude','business_longitude']]], axis=1)
df_formap.dropna(inplace=True)


# In[ ]:


import folium

sf_coords = (37.76, -122.45)

#create empty map zoomed in on SF
sf_map = folium.Map(location=sf_coords, zoom_start=13)

risks = {
    'Low Risk'      : 'green', 
    'Moderate Risk' : 'orange', 
    'High Risk'     : 'red', 
    }

def plotAddress(df):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    #print("%s" %(risks[df.risk_category]))
    marker_color = risks[df['most_severe_risk']]
    folium.Marker(location=[df['business_latitude'], df['business_longitude']],
                        popup=df['business_name'],
                        icon=folium.Icon(color=marker_color, icon='circle',prefix='fa'),
                        
                       ).add_to(sf_map)
    
df_formap[df_formap['inspection_score']>98].apply(plotAddress, axis = 1)


display(sf_map)


# ## Thank you note
# Thanks for your time and patience to go through my notebook.
# 
# I hope you got some insights about the San Francisco Restaurant Scene and choose wisely before going out for dinner next time!
# 
# Feel free to upvote :)
# 

# In[ ]:





# In[ ]:




