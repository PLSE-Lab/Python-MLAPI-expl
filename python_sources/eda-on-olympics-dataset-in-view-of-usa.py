#!/usr/bin/env python
# coding: utf-8

# # USA in the Olympics

# ### 120 years of Olympic history: 
# - This Dataset is taken from Kaggle. It can be found on the below link:
#     - https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
#     
# This dataset contains basic bio data on athletes and medal results from Athens 1896 to Rio 2016

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


events = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
regions = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')


# In[ ]:


events.head()


# In[ ]:


events.columns


# In[ ]:


events.shape


# In[ ]:


events.isnull().sum()


# #### By the above values, We can find that Age, Height, Weight and Medals have lot of missing values. The medal column have 231333 missing values. This is fine because not all the participants win a medal. So we will replace this values with No - No Medal

# In[ ]:


events['Medal'].fillna('No Medal', inplace = True)


# In[ ]:


events.isnull().sum()


# In[ ]:


regions.shape


# In[ ]:


del regions['notes']
regions.rename(columns = {'region':'Country'}, inplace = True)
regions.head()


# In[ ]:


olympics = events.merge(regions, on = 'NOC', how = 'left')
olympics.head()


# In[ ]:


olympics.loc[olympics['Country'].isnull(),['Team','NOC']].drop_duplicates()


# **The Above NOC present in the olympic events dataset does not associate to a country in the regions dataset. But we can easily add them manually based on their TEAM Name**

# ### Filling all NA Values in the Country Column with the Team Names

# In[ ]:


olympics.Country.fillna(olympics.Team, inplace = True)


# In[ ]:


len(olympics.Country.unique().tolist())


# In[ ]:


olympics['Medal'].unique().tolist()


# In[ ]:


olympics1 = olympics.replace({'Medal':{'Gold': 1,'Silver': 1,'Bronze':1, 'No Medal': 0}})


# In[ ]:


olympics1.head()


# In[ ]:


summer_olympics = olympics1[olympics.Season == 'Summer']
winter_olympics = olympics1[olympics.Season == 'Winter']


# # Top 10 countries with highest number of medals.

# In[ ]:


summer_medal_count = summer_olympics[['Country','Medal']].groupby('Country', as_index = False).sum()
summer_medal_count = summer_medal_count[summer_medal_count.Medal>0].sort_values(by = ['Medal'], ascending = False).head(10)
winter_medal_count = winter_olympics[['Country','Medal']].groupby('Country', as_index = False).sum()
winter_medal_count = winter_medal_count[winter_medal_count.Medal>0].sort_values(by = 'Medal', ascending = False).head(10)

print('-'*5,'SUMMER OLYMPICS', '-'*5)
print(summer_medal_count)
print()
print('-'*5,'WINTER OLYMPICS', '-'*5)
print(winter_medal_count)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig,(ax1,ax2) = plt.subplots(1,2)
fig.tight_layout()
fig.set_figheight(12)
fig.set_figwidth(10)
ax1.pie(summer_medal_count.Medal, labels = summer_medal_count.Country, autopct='%1.1f%%')
ax1.set_title('Summer Olympics', fontsize = 16)
ax2.pie(winter_medal_count.Medal, labels = winter_medal_count.Country, autopct='%1.1f%%')
ax2.set_title('Winter Olympics', fontsize = 16)
# ax.axis('equal')

plt.show()


# # Report on Initial Data Analysis for USA on Olympics Dataset
# 
# ### Summer Olympics:
# - Among the Top 10 countries with the highest number of medals, we see that **USA ranks No 1** with the most number of medals won. It has achieved 24% of medals among the Top 10.
# 
# ### Winter Olympics:
# - Among the Top 10 countries, USA stands second place with 13.5% of medal won among the Top 10. In Winter Olympics, Russia stand on the Top with 16.1 % of medals won among the Top 10.
# 
# 
#     We will dig further in to study how USA has performed over the years.

# #  Data Analysis for USA

# In[ ]:


usa = olympics1[olympics1.Country == 'USA']
summer_usa = usa[usa.Season == 'Summer']
summer_usa.head()


# In[ ]:


winter_usa = usa[usa.Season == 'Winter']


# In[ ]:


winter_usa.head()


# In[ ]:


summer_usa_years = summer_usa[['Year', 'Medal']].groupby('Year', as_index = False).sum()
summer_event_count = summer_usa.groupby('Year')['Event'].nunique().reset_index()
winter_usa_years = winter_usa[['Year', 'Medal']].groupby('Year', as_index = False).sum()
winter_event_count = winter_usa.groupby('Year')['Event'].nunique().reset_index()


# In[ ]:


summer_usa_years.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig,ax = plt.subplots(figsize = (14,8))
plt.plot(summer_usa_years['Year'].values, summer_usa_years['Medal'].values, color = 'blue', marker = 'o', label = 'Medal Count')
plt.plot(summer_event_count['Year'].values, summer_event_count['Event'].values, color = 'green', marker = 'o', label = 'Events count')
plt.xlabel('Years',  fontweight = 'bold', fontsize = 14)
plt.ylabel('No of Medals', fontweight = 'bold',fontsize = 14)
plt.legend()
plt.title('Medals won by USA in Summer Event over years', fontweight = 'bold', fontsize = 18)
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize = (14,8))
plt.plot(winter_usa_years['Year'].values, winter_usa_years['Medal'].values, color = 'blue', marker = 'o', label = 'Medal count')
plt.plot(winter_event_count['Year'].values, winter_event_count['Event'].values, color = 'green', marker = 'o', label = 'Events count')
plt.xlabel('Years',  fontweight = 'bold', fontsize = 14)
plt.ylabel('No of Medals', fontweight = 'bold',fontsize = 14)
plt.legend()
plt.title('Medals won by USA in Winter Event over years', fontweight = 'bold', fontsize = 18)
plt.show()


# ## Number and type of medals won by USA per sport for Summer Olympics

# In[ ]:


from functools import reduce

usa_gold_summer = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Summer') & (olympics['Medal'] == 'Gold')]
usa_silver_summer = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Summer') & (olympics['Medal'] == 'Silver')]
usa_bronze_summer = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Summer') & (olympics['Medal'] == 'Bronze')]
usa_nm_summer = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Summer') & (olympics['Medal'] == 'No Medal')]


# In[ ]:


usa_gold = usa_gold_summer[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Gold'})
usa_silver = usa_silver_summer[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Silver'})
usa_bronze = usa_bronze_summer[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Bronze'})
events_per_sport = summer_usa.groupby('Sport')['Event'].nunique()                     .reset_index()                     .rename(columns = {'Event': 'Events per Sport'})
                            
dataframe = [events_per_sport,usa_gold, usa_silver, usa_bronze]
usa_medals_summer = reduce(lambda left,right: pd.merge(left, right, on = 'Sport', how = 'outer'),                         dataframe).fillna(0)

print('-------------------------Summer Olympics-------------------')
print()
print(usa_medals_summer.head())


# In[ ]:


fig,ax = plt.subplots(figsize = (15,10))
plt.barh(usa_medals_summer.Sport.values, usa_medals_summer.Gold.values, color = '#C49133', label = 'Gold')
plt.barh(usa_medals_summer.Sport.values, usa_medals_summer.Silver.values, color = '#828A95', label = 'Silver')
plt.barh(usa_medals_summer.Sport.values, usa_medals_summer.Bronze.values, color = '#914E24', label = 'Bronze')
plt.title('Medals won by USA for the Summer Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('No of Medals', fontsize = 18, fontweight = 'bold')
plt.ylabel('Sport', fontsize = 18, fontweight = 'bold')
plt.legend(fontsize = 20)
plt.tight_layout()
plt.show()


# ## Report on the sports and their medals achieved by USA - Summer Olympics
# 
# - By the grapth we can say that, USA has be dominating in **Swimming**, **Rowing**, **Basketball** and in most of the **Athletics** sports. 
# 
# 
# - ### No Medals won for the below Sports:
# 
#   - Trampolining
#   - Table Tennis
#   - Rugby Sevens
#   - Rhythmic Gymnastics
#   - Handball
#   - Badminton
# - ### No Gold Medals were won for the below Sports:
# 
#   - Modern Pentathlon
#   - Lacrosse
#   - Hockey
#   - Figure Skating      
#         

# ## Number and type of medals won by USA per sport for Winter Olympics

# In[ ]:


usa_gold_winter = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Winter') & (olympics['Medal'] == 'Gold')]
usa_silver_winter = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Winter') & (olympics['Medal'] == 'Silver')]
usa_bronze_winter = olympics[(olympics['Country'] == 'USA') & (olympics['Season'] == 'Winter') & (olympics['Medal'] == 'Bronze')]


# In[ ]:


usa_w_gold = usa_gold_winter[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Gold'})
usa_w_silver = usa_silver_winter[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Silver'})
usa_w_bronze = usa_bronze_winter[['Sport','Medal']].groupby('Sport', as_index = False).count()             .rename(columns = {'Medal':'Bronze'})
events_perw_sport = winter_usa.groupby('Sport')['Event'].nunique()                     .reset_index()                     .rename(columns = {'Event': 'Events per Sport'})
                            
dataframe = [events_perw_sport,usa_w_gold, usa_w_silver, usa_w_bronze]
usa_medals_winter = reduce(lambda left,right: pd.merge(left, right, on = 'Sport', how = 'outer'),                         dataframe).fillna(0)
print('-'*30,'Winter Olympics','-'*30)
print()
print(usa_medals_winter)


# In[ ]:


fig,ax = plt.subplots(figsize = (15,10))
plt.barh(usa_medals_winter.Sport.values, usa_medals_winter.Gold.values, color = '#C49133', label = 'Gold')
plt.barh(usa_medals_winter.Sport.values, usa_medals_winter.Silver.values, color = '#828A95', label = 'Silver')
plt.barh(usa_medals_winter.Sport.values, usa_medals_winter.Bronze.values, color = '#914E24', label = 'Bronze')
plt.title('Medals won by USA for the Winter Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('No of Medals', fontsize = 18, fontweight = 'bold')
plt.ylabel('Sport', fontsize = 18, fontweight = 'bold')
plt.legend(fontsize = 20)
plt.tight_layout()
plt.show()


# ## Report on the sports and their medals achieved by USA - Winter Olympics
# 
# - By the grapth we can say that, USA has be dominating in **Ice Hockey** and **Speed Skating**. 
# 
# 
# ### No Medals won for the below Sports:
#   - Biathlon
#  
# ### No Gold Medals were won for the below Sports:
#   - Ski Jumping
#   - Luge
#   - Curling
#   - Cross Country Skiing    
