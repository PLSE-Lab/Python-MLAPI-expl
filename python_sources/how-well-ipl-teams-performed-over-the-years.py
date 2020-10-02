#!/usr/bin/env python
# coding: utf-8

# # We will Visualize individual team's Performance that how they have performed over the years in this IPL tournament and with which teams they have great records and with which teams they have performed poorly.
# 
# ## The analysis of the teams will be done in this order :
# 
#  1. Deccan Chargers
#  2. Mumbai Indians
#  3. Royal Challengers Bangalore
#  4. Chennai Super Kings
#  5. Kings XI Punjab
#  6. Delhi Daredevils
#  7. Kolkata Knight Riders
#  8. Sunrisers Hyderabad
#  9. Rajasthan Royals

# # Contents :
# 
# ## This Kernel is divided into 4 Steps :
# * Step-1 : Data Loading
# * Step-2 : Data Preparing/Cleaning
# * Step-3 : Exploratary Data Analysis
# * Step-4 : Data Visualisation

# In[ ]:


# Importing Essential Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


# # Step-1 : Data Loading

# In[ ]:


df = pd.read_csv('/kaggle/input/ipl/matches.csv')
df.head()


# # Step-2 : Data Preparing/Cleaning

# ### For all the cities having NULL values, VENUE is Dubai International Cricket Stadium, so we will fill city as DUBAI

# In[ ]:


df.city.fillna('Dubai',inplace=True)


# ### There are total of 3 records in which we don't have any winner. May be the match would have been washed out due to Rain. So, let's fill up those values as -> Match Abandoned

# In[ ]:


df.winner.fillna('Match Abandoned',inplace=True)
df.player_of_match.fillna('Match Abandoned',inplace=True)
df.umpire1.fillna('Anonymous',inplace=True)
df.umpire2.fillna('Anonymous',inplace=True)
df.drop(columns='umpire3',inplace=True)


# ### Now, as we know Franchise of Pune IPL team changed their name from Rising Pune Supergiants to Rising Pune Supergiant(removed s from Supergiants) in Season 2017. So, we will consider both the teams as 1 team only

# In[ ]:


df['winner'][df['winner'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'
df['team1'][df['team1'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'
df['team2'][df['team2'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'
df['toss_winner'][df['toss_winner'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'


# # Step-3 : Exploratory Data Analysis

# In[ ]:


teams = pd.Series(df['team1'].unique())


# In[ ]:


years = pd.Series(df['season'].unique())
years=years.sort_values()


# # Now, let's create a DATAFRAME that will show which team has PLAYED how many matches in a particular Season

# In[ ]:


total_matches_played_dict = {}

for team in teams:
    matches_per_year = []
    for year in years:
        total_match_per_year = df[((df['team1']==team) | (df['team2']==team)) & (df['season']==year)].shape[0]
        matches_per_year.append(total_match_per_year)
    total_matches_played_dict[team] = matches_per_year
dframe_matches_played=pd.DataFrame(total_matches_played_dict,index=years)


# In[ ]:


dframe_matches_played


# # Creating a DATAFRAME that will show which team has WON how many matches in a particular Season

# In[ ]:


dict_win_num = {}

for team in teams:
    matches_won = []
    for year in years:
        matches_won_per_year = df[(df['winner']==team) & (df['season']==year)].shape[0]
        matches_won.append(matches_won_per_year)
    dict_win_num[team] = matches_won
dframe_matches_won = pd.DataFrame(dict_win_num,index=years)


# In[ ]:


dframe_matches_won


# # Creating a DATAFRAME that will show WIN PERCENTAGE of teams in each Season

# In[ ]:


dframe_match_won_percnt = (dframe_matches_won/dframe_matches_played)*100
dframe_match_won_percnt.fillna(0,inplace=True)


# In[ ]:


dframe_match_won_percnt


# In[ ]:


dframe_match_won_percnt['Year'] = dframe_match_won_percnt.index


# # Now, let's create a DATAFRAME that will show number of times TEAMS PLAYED AGAINST EACH OTHER

# In[ ]:


dict_oppnt_matches = {}

for team_1 in teams:
    matches_against_oppnt = []
    for team_2 in teams:
        match_plyd_against_this_oppnt = df[((df['team1']==team_1) & (df['team2']==team_2)) | ((df['team1']==team_2) & (df['team2']==team_1))].shape[0]
        matches_against_oppnt.append(match_plyd_against_this_oppnt)
    dict_oppnt_matches[team_1] = matches_against_oppnt
    
dframe_total_match_agnst_oppnt = pd.DataFrame(dict_oppnt_matches,index=teams)


# In[ ]:


dframe_total_match_agnst_oppnt


# # Creating a DATAFRAME that will show number of times TEAMS WON AGAINST EACH OTHER

# In[ ]:


dict_oppnt_matches_won = {}

for team_1 in teams:
    matches_won_against_oppnt = []
    for team_2 in teams:
        match_won_against_this_oppnt = df[(((df['team1']==team_1) & (df['team2']==team_2)) | ((df['team1']==team_2) & (df['team2']==team_1))) & (df['winner']==team_1)].shape[0]
        matches_won_against_oppnt.append(match_won_against_this_oppnt)
    dict_oppnt_matches_won[team_1] = matches_won_against_oppnt
    
dframe_total_match_won_agnst_oppnt = pd.DataFrame(dict_oppnt_matches_won,index=teams)


# In[ ]:


dframe_total_match_won_agnst_oppnt


# # Creating a DATAFRAME that will show WIN PERCENTAGE of TEAMS WON AGAINST EACH OTHER

# In[ ]:


dframe_oppnt_won_percnt = (dframe_total_match_won_agnst_oppnt/dframe_total_match_agnst_oppnt)*100
dframe_oppnt_won_percnt.fillna(0,inplace=True)


# In[ ]:


dframe_oppnt_won_percnt


# In[ ]:


teams_dict = {'Sunrisers Hyderabad' : 'SRH',
'Mumbai Indians' : 'MI',
'Gujarat Lions' : 'GL',
'Rising Pune Supergiant' : 'RPS',
'Royal Challengers Bangalore' : 'RCB',
'Kolkata Knight Riders' : 'KKR',
'Delhi Daredevils' : 'DD',
'Kings XI Punjab' : 'KXIP',
'Chennai Super Kings' : 'CSK',
'Rajasthan Royals' : 'RR',
'Deccan Chargers' : 'DC',
'Kochi Tuskers Kerala' : 'KTK',
'Pune Warriors' : 'PW'}


# In[ ]:


dframe_oppnt_won_percnt['Teams'] = dframe_oppnt_won_percnt.index.map(teams_dict)


# In[ ]:


dframe_oppnt_won_percnt


# # Step-4 : Data Visualisation

# In[ ]:


sns.set_style('whitegrid')


# ## Let's see IPL team's Performance over the years....

# # 1. DECCAN CHARGERS

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Deccan Chargers Graph over the Years')
plt.ylim(0,100)
plt.xlim(2008,2012)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Deccan Chargers',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Deccan Chargers performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Deccan Chargers')


# # Observations :
# 
# ## * Deccan Chargers played first 5 seasons of the IPL and after that the owner sold this Hyderabad Franchise. Later, the new owner of this Hyderabad Franchise changed its name to Sunrisers Hyderabad.
# ## * As we can see from graph, the only Best Season for the Deccan Chargers was 2009. This year the whole IPL Tournament was played in South Africa and they WON IPL Trophy this year under the leadership of Adam Gilchrist.
# ## * The performance of Deccan Chargers kept on digging after 2009 which forced their owner to sell this franchise after season 2012.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Deccan Chargers against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Deccan Chargers')


# # Observations :
# ## * Deccan Chargers never lost to KOCHI TUSKERS KERALA.
# ## * Deccan Chargers had a Good Record only against PUNE WARRIORS and ROYAL CHALLANGERS BANGALORE. Rest they have struggled against all the teams.

# # 2. MUMBAI INDIANS

# In[ ]:


plt.figure(figsize=(16,6))
plt.title('Mumbai Indians Graph over the years')
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
sns.lineplot(
    x='Year',
    y='Mumbai Indians',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Mumbai Indians performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Mumbai Indians')


# # Observations :
# 
# ## * The 2 worst Seasons went for MUMBAI INDIANS were - 2009 and 2014
# ## * Overall, Mumbai Indians had an amazing IPL journey so far with their most successful seasons as - 2017, 2013 and 2010

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Mumbai Indians against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Mumbai Indians')


# # Observations :
# 
# ## * As seen from the above visualisation, Mumbai Indians has been a Strong Competitor for almost all the teams except RISING PUNE SUPERGIANT against whom they have struggled a bit.
# ## * The 2 teams against whom Mumbai Indians have an exceptional record are - KOLKATA KNIGHT RIDERS and PUNE WARRIORS

# # 3. ROYAL CHALLENGERS BANGALORE

# In[ ]:


plt.figure(figsize=(16,6))
plt.title('Royal Challengers Bangalore Graph over the Years')
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Royal Challengers Bangalore',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Royal Challengers Bangalore performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Royal Challengers Bangalore')


# # Observations :
# 
# ## * From the graph, we can clearly see that Royal Challengers Bangalore didn't had a smooth journey so far
# ## * The most successful season of RCB so far is - 2011
# ## * The worst seasons for RCB were - 2008,2014 and 2017

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Royal Challengers Bangalore against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Royal Challengers Bangalore')


# # Observations :
# 
# ## * Royal Challengers Bangalore have 100% win rate against - KOCHI TUSKERS KERALA and PUNE WARRIORS.
# ## * RCB is also having good record against - GUJARAT LIONS and DELHI DAREDEVILS.
# ## * RCB have worst records against - CHENNAI SUPER KINGS and MUMBAI INDIANS

# # 4. CHENNAI SUPER KINGS

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Chennai Super Kings Graph over the Years')
plt.ylim(0,100)
plt.xlim(2008,2015)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Chennai Super Kings',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Chennai Super Kings performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Chennai Super Kings')


# # Observations :
# 
# ## * Above Graph clearly indicates that Chennai Super Kings had a most successful IPL journey so far as compared to other teams.
# ## * Chennai Super Kings was BANNED for 2 years due to allegations of Betting and Match Fixing,that is why no data is there for seasons - 2016 and 2017.
# ## * Most successful seasons of CSK were - 2011,2013 and 2014.
# ## * CSK didn't had any bad season so far.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Chennai Super Kings against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Chennai Super Kings')


# # Observations :
# 
# ## * Chennai Super Kings is having an exceptional record against all the teams.
# ## * The only team against whom they have struggled a bit is MUMBAI INDIANS

# # 5. KINGS XI PUNJAB

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Kings XI Punjab Graph over the Years')
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Kings XI Punjab',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Kings XI Punjab performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Kings XI Punjab')


# # Observations :
# 
# ## * The Best Seasons for KXIP were - 2008 and 2014.
# ## * The Worst Seasons for KXIP were - 2010, 2015 and 2016.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Kings XI Punjab against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Kings XI Punjab')


# # Observations :
# 
# ## * Kings XI Punjab has never lost to KOCHI TUSKERS KERALA.
# ## * Kings XI Punjab is having Great Record against teams - DECCAN CHARGERS, ROYAL CHALLANGERS BANGALORE and DELHI DAREDEVILS.
# ## * Kings XI Punjab is having Worst Record against teams - SUNRISERS HYDERABAD, KOLKATA KNIGHT RIDERS, RAJASTHAN ROYALS and CHENNAI SUPER KINGS.

# # 6. DELHI DAREDEVILS

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Delhi Daredevils Graph over the Years')
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Delhi Daredevils',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Delhi Daredevils performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Delhi Daredevils')


# # Observations :
# 
# ## * Delhi Daredevils had a disappointed season so far.
# ## * The 2 only successful Seasons for Delhi Daredevils were - 2009 and 2012.
# ## * The 3 worst Seasons for Delhi Daredevils were - 2014, 2013 and 2011.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Delhi Daredevils against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Delhi Daredevils')


# # Observations :
# 
# ## * Delhi Daredevils had a Poor Record among all the teams except 2.
# ## * Delhi Daredevils had a Good Record only against - GUJARAT LIONS and DECCAN CHARGERS.

# # 7. KOLKATA KNIGHT RIDERS

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Kolkata Knight Riders Graph over the Years')
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Kolkata Knight Riders',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Kolkata Knight Riders performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Kolkata Knight Riders')


# # Observations :
# 
# ## * The 2 Great Seasons for Kolkata Knight Riders were - 2012 and 2014.
# ## * The 3 Worst Seasons for Kolkata Knight Riders were - 2009, 2013 and 2008.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Kolkata Knight Riders against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Kolkata Knight Riders')


# # Observations :
# 
# ## * Kolkata Knight Riders have a Good Record among all the teams except these 4 against whom they have a very Poor Record - MUMBAI INDIANS, GUJARAT LIONS, CHENNAI SUPER KINGS and RAJASTHAN ROYALS.

# # 8. SUNRISERS HYDERABAD

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Sunrisers Hyderabad Graph over the Years')
plt.ylim(0,100)
plt.xlim(2013,2017)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Sunrisers Hyderabad',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Sunrisers Hyderabad performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Sunrisers Hyderabad')


# # Observations :
# 
# ## * Sunrisers Hyderabad came into existence from 2013 onwards. Earlier this Hyderabad Franchise was owned by someone else and it was named as Deccan Chargers at that time.
# ## * The journey of Sunrisers Hyderabad has been Great so far with their Best Season as 2016(They won IPL Trophy as well) and Worst Season as 2014.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Sunrisers Hyderabad against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Sunrisers Hyderabad')


# # Observations :
# 
# ## * Sunrisers Hyderabad have 100% Win Record against - GUJARAT LIONS and PUNE WARRIORS.
# ## * Sunrisers Hyderabad have Great Record against - KINGS XI PUNJAB, DELHI DAREDEVILS and ROYAL CHALLENGERS BANGALORE.
# ## * Sunrisers Hyderabad have Worst Record against - RISING PUNE SUPERGIANT, KOLKATA KNIGHT RIDERS and CHENNAI SUPER KINGS.

# # 9. RAJASTHAN ROYALS

# In[ ]:


plt.figure(figsize=(18,6))
plt.title('Rajasthan Royals Graph over the Years')
plt.ylim(0,100)
plt.xlim(2008,2015)
plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(
    x='Year',
    y='Rajasthan Royals',
    data=dframe_match_won_percnt)


# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red',linewidth='0.6')
plt.axhline(y=50,color='red')
plt.title('Rajasthan Royals performance over the years')

sns.barplot(data=dframe_match_won_percnt,x='Year',y='Rajasthan Royals')


# # Observations :
# 
# ## * Rajasthan Royals doesn't had a smooth IPL journey so far.
# ## * Rajasthan Royals along with Chennai Super Kings was also BANNED for 2 years due to allegations of Betting and Match Fixing,that is why no data is there for seasons - 2016 and 2017.
# ## * Rajasthan Royals had an Exceptional Season in 2008(They won IPL Trophy as well).
# ## * Except 2008, their journey has been below par only. The another good season for them was - 2013.

# In[ ]:


plt.figure(figsize=(16,6))
plt.ylim(0,100)
plt.axhline(y=50,color='red')
plt.title('Rajasthan Royals against its Opponents')

sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Rajasthan Royals')


# # Observations :
# 
# ## * Rajasthan Royals had a Great Record against - PUNE WARRIORS and DECCAN CHARGERS.
# ## * Rajasthan Royals have also performed well against teams - DELHI DAREDEVILS, KOLKATA KNIGHT RIDERS, KINGS XI PUNJAB and SUNRISERS HYDERABAD.
# ## * The teams against whom they have struggled a lot are - MUMBAI INDIANS, CHENNAI SUPER KINGS and ROYAL CHALLENGERS BANGALORE.

# ### Thanks for watching this Kernel :)
