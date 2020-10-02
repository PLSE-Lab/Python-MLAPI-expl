#!/usr/bin/env python
# coding: utf-8

# ## Intro

# [](http://)![understat.JPG](http://sergilehkyi.com/wp-content/uploads/2019/06/understat.jpg)
# 
# In this notebook I will describe the process of scraping data from web portal [understat.com](https://understat.com) that has a lot of statistical information about all games in top 5 European football leagues.
# 
# From [understat.com](https://understat.com) home page:
# 
# * Expected goals (xG) is the new revolutionary football metric, which allows you to evaluate team and player performance.
# 
# * In a low-scoring game such as football, final match score does not provide a clear picture of performance.
# 
# * This is why more and more sports analytics turn to the advanced models like xG, which is a statistical measure of the quality of chances created and conceded.
# 
# * Our goal was to create the most precise method for shot quality evaluation.
# 
# * For this case, we trained neural network prediction algorithms with the large dataset (>100,000 shots, over 10 parameters for each).
# 
# * On this site, you will find our detailed xG statistics for the top European leagues.
# 
# At this moment they have not only xG metric, but much more, that makes this site perfect for scraping statistical data about football games.
# 
# 
# 

# We start by importing libraries that will be used in this project:
# * numpy - fundamental package for scientific computing with Python
# * pandas - library providing high-performance, easy-to-use data structures and data analysis tools
# * requests - is the only Non-GMO HTTP library for Python, safe for human consumption. (love this line from official docs :D)
# * BeautifulSoup - a Python library for pulling data out of HTML and XML files.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from bs4 import BeautifulSoup


# ## Website research and structure of data

# On the home page we can notice that the site has data for 6 European Leagues:
# 
# ![leagues.jpg](http://sergilehkyi.com/wp-content/uploads/2019/06/leagues.jpg)
# 
# *   La Liga
# *   EPL
# *   BundesLiga
# *   Serie A
# *   Ligue 1
# *   RFPL
# 
# And we also see that the data collected is starting from season 2014/2015. Another notion we make is the structure of URL. It is '`https://understat.com/league'` + '`/name_of_the_league`' + '`/year_start_of_the_season`'
# 
# ![seasons.jpg](http://sergilehkyi.com/wp-content/uploads/2019/06/seasons.jpg)
# 
# So we create global variables with this data to be able to select any of those.

# In[ ]:


# create urls for all seasons of all leagues
base_url = 'https://understat.com/league'
leagues = ['La_liga', 'EPL', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL']
seasons = ['2014', '2015', '2016', '2017', '2018', '2019']


# Next step is to understand where the data is located on the web-page. For this we open Developer Tools in Chrome, go to tab "Network", find file with data (in this case 2018) and check the "Response" tab. This is what we will get after running *requests.get(URL)*
# 
# ![requests_response_1.jpg](http://sergilehkyi.com/wp-content/uploads/2019/06/requests_response_1.jpg)
# 
# After going through content of the web-page we find that the data is stored under "script" tag and it is JSON encoded. So we will need to find this tag, get JSON from it and convert it into Python readable data structure.
# 
# ![requests_response_2.jpg](http://sergilehkyi.com/wp-content/uploads/2019/06/requests_response_2.jpg)

# In[ ]:


# Starting with latest data for Spanish league, because I'm a Barcelona fan
url = base_url+'/'+leagues[0]+'/'+seasons[4]
res = requests.get(url)
soup = BeautifulSoup(res.content, "lxml")

# Based on the structure of the webpage, I found that data is in the JSON variable, under <script> tags
scripts = soup.find_all('script')

# Check our <script> tags
# for el in scripts:
#   print('*'*50)
#   print(el.text)


# ### Working with JSON

# We found that the data interesting us is stored in teamsData variable, after creating a soup of html tags it becomes just a string, so we find that text and extract JSON from it.

# In[ ]:


import json

string_with_json_obj = ''

# Find data for teams
for el in scripts:
    if 'teamsData' in el.text:
      string_with_json_obj = el.text.strip()
      
# print(string_with_json_obj)

# strip unnecessary symbols and get only JSON data
ind_start = string_with_json_obj.index("('")+2
ind_end = string_with_json_obj.index("')")
json_data = string_with_json_obj[ind_start:ind_end]

json_data = json_data.encode('utf8').decode('unicode_escape')


# Once we have gotten our JSON and cleaned it up we can convert it into Python dictionary and check how it looks (uncomment print statement to do that).

# ### Understanding data with Python

# In[ ]:


# convert JSON data into Python dictionary
data = json.loads(json_data)
print(data.keys())
print('='*50)
print(data['138'].keys())
print('='*50)
print(data['138']['id'])
print('='*50)
print(data['138']['title'])
print('='*50)
print(data['138']['history'][0])

# Print pretty JSON data to check out what we have there
# s = json.dumps(data, indent=4, sort_keys=True)
# print(s)


# If you want to check how the entire <code>data</code> looks, just uncomment respective lines. (it is commented for the sake of saving screen space and do not oversaturate the view of notebook).
# 
# When we start to research the <code>data</code> we understand that this is a dictionary of dictionaries of 3 keys: *`id`*, *`title`* and *`history`*. The first layer of dictionary uses ids as keys too.
# 
# Also from this we understand that *`history`* has data regarding every single match the team played in its own league (League Cup or Champions League games are not included).
# 
# We can gather teams names if go over the first layer dictionary.

# In[ ]:


# Get teams and their relevant ids and put them into separate dictionary
teams = {}
for id in data.keys():
  teams[id] = data[id]['title']


# The *`history`* is the array of dictionaries where keys are names of metrics (read column names) and values are values, despite how tautological is that :D.
# 
# We understand that column names repeat over and over again so we add them to separate list. Also checking how the sample values look like.

# In[ ]:


# EDA to get a feeling of how the JSON is structured
# Column names are all the same, so we just use first element
columns = []
# Check the sample of values per each column
values = []
for id in data.keys():
  columns = list(data[id]['history'][0].keys())
  values = list(data[id]['history'][0].values())
  break

print(columns)
print(values)


# Found that Sevilla has the id=138, so getting all the data for this team to be able to reproduce the same steps for all teams in the league.

# In[ ]:


sevilla_data = []
for row in data['138']['history']:
  sevilla_data.append(list(row.values()))
  
df = pd.DataFrame(sevilla_data, columns=columns)
df.head(2)


# Wualya! We have the data for all matches of Sevilla in season 2018-2019 within La Liga!
# 
# Now we want to do that for all Spanish teams. Let's loop through that bites baby!

# In[ ]:


# Getting data for all teams
dataframes = {}
for id, team in teams.items():
  teams_data = []
  for row in data[id]['history']:
    teams_data.append(list(row.values()))
    
  df = pd.DataFrame(teams_data, columns=columns)
  dataframes[team] = df
  print('Added data for {}.'.format(team))
  


# Now we have a dictionary of DataFrames where key is the name of the team and value is the DataFrame with all games of that team.

# In[ ]:


# Sample check of our newly created DataFrame
dataframes['Barcelona'].head(2)


# ### Manipulations to make data as in the original source

# We can notice that here such metrics as PPDA and OPPDA (ppda and ppda_allowed) are represented as total amounts of attacking/defensive actions, but in the original table it is shown as coefficient. Let's fix that!

# In[ ]:


for team, df in dataframes.items():
  dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
  dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
  
# And check how our new dataframes look based on Sevilla dataframe
dataframes['Sevilla'].head(2)


# Now we have all our numbers, but for every single game. What we need is the totals for the team. Let's find out the columns we have to sum up. For this we go back to original table at [understat.com](https://understat.com/league/La_liga/2018) and we find that all metrics shoud be summed up and only PPDA and OPPDA are means in the end.

# In[ ]:


cols_to_sum = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'xpts', 'wins', 'draws', 'loses', 'pts', 'npxGD']
cols_to_mean = ['ppda_coef', 'oppda_coef']


# We are ready to calculate our totals and means. For this we loop through dictionary of dataframes and call .sum() and .mean() DataFrame methods that return Series, that's why we add .transpose() to those calls. We put these new DataFrames into a list and after that concat them into a new DataFrame `full_stat`

# In[ ]:


frames = []
for team, df in dataframes.items():
  sum_data = pd.DataFrame(df[cols_to_sum].sum()).transpose()
  mean_data = pd.DataFrame(df[cols_to_mean].mean()).transpose()
  final_df = sum_data.join(mean_data)
  final_df['team'] = team
  final_df['matches'] = len(df)
  frames.append(final_df)
  
full_stat = pd.concat(frames)


# Next we reorder columns for better readability, sort rows based on points, reset index and add column 'position'.

# In[ ]:


full_stat = full_stat[['team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'npxG', 'xGA', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts']]
full_stat.sort_values('pts', ascending=False, inplace=True)
full_stat.reset_index(inplace=True, drop=True)
full_stat['position'] = range(1,len(full_stat)+1)


# Also in the original table we have values of differences between expected metrics and real. Let's add those too.

# In[ ]:


full_stat['xG_diff'] = full_stat['xG'] - full_stat['scored']
full_stat['xGA_diff'] = full_stat['xGA'] - full_stat['missed']
full_stat['xpts_diff'] = full_stat['xpts'] - full_stat['pts']


# Converting floats to integers where appropriate

# In[ ]:


cols_to_int = ['wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'deep', 'deep_allowed']
full_stat[cols_to_int] = full_stat[cols_to_int].astype(int)


# Prettifying output and final view of a DataFrame

# In[ ]:


col_order = ['position','team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'xG_diff', 'npxG', 'xGA', 'xGA_diff', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts', 'xpts_diff']
full_stat = full_stat[col_order]
pd.options.display.float_format = '{:,.2f}'.format
full_stat.head(10)


# Original table
# 
# ![full_table.JPG](http://sergilehkyi.com/wp-content/uploads/2019/06/full_table.jpg)

# ## Scraping data for all teams of all leagues of all seasons

# Testing the flow before going full into the process

# In[ ]:


season_data = dict()
season_data[seasons[4]] = full_stat
print(season_data)
full_data = dict()
full_data[leagues[0]] = season_data
print(full_data)


# Putting all the previous code into loops to get all data.

# In[ ]:


full_data = dict()
for league in leagues:
  
  season_data = dict()
  for season in seasons:    
    url = base_url+'/'+league+'/'+season
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "lxml")

    # Based on the structure of the webpage, I found that data is in the JSON variable, under <script> tags
    scripts = soup.find_all('script')
    
    string_with_json_obj = ''

    # Find data for teams
    for el in scripts:
        if 'teamsData' in el.text:
          string_with_json_obj = el.text.strip()

    # print(string_with_json_obj)

    # strip unnecessary symbols and get only JSON data
    ind_start = string_with_json_obj.index("('")+2
    ind_end = string_with_json_obj.index("')")
    json_data = string_with_json_obj[ind_start:ind_end]
    json_data = json_data.encode('utf8').decode('unicode_escape')
    
    
    # convert JSON data into Python dictionary
    data = json.loads(json_data)
    
    # Get teams and their relevant ids and put them into separate dictionary
    teams = {}
    for id in data.keys():
      teams[id] = data[id]['title']
      
    # EDA to get a feeling of how the JSON is structured
    # Column names are all the same, so we just use first element
    columns = []
    # Check the sample of values per each column
    values = []
    for id in data.keys():
      columns = list(data[id]['history'][0].keys())
      values = list(data[id]['history'][0].values())
      break
      
    # Getting data for all teams
    dataframes = {}
    for id, team in teams.items():
      teams_data = []
      for row in data[id]['history']:
        teams_data.append(list(row.values()))

      df = pd.DataFrame(teams_data, columns=columns)
      dataframes[team] = df
      # print('Added data for {}.'.format(team))
      
    
    for team, df in dataframes.items():
      dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
      dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
      
    cols_to_sum = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'xpts', 'wins', 'draws', 'loses', 'pts', 'npxGD']
    cols_to_mean = ['ppda_coef', 'oppda_coef']
    
    frames = []
    for team, df in dataframes.items():
      sum_data = pd.DataFrame(df[cols_to_sum].sum()).transpose()
      mean_data = pd.DataFrame(df[cols_to_mean].mean()).transpose()
      final_df = sum_data.join(mean_data)
      final_df['team'] = team
      final_df['matches'] = len(df)
      frames.append(final_df)

    full_stat = pd.concat(frames)
    
    full_stat = full_stat[['team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'npxG', 'xGA', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts']]
    full_stat.sort_values('pts', ascending=False, inplace=True)
    full_stat.reset_index(inplace=True, drop=True)
    full_stat['position'] = range(1,len(full_stat)+1)
    
    full_stat['xG_diff'] = full_stat['xG'] - full_stat['scored']
    full_stat['xGA_diff'] = full_stat['xGA'] - full_stat['missed']
    full_stat['xpts_diff'] = full_stat['xpts'] - full_stat['pts']
    
    cols_to_int = ['wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'deep', 'deep_allowed']
    full_stat[cols_to_int] = full_stat[cols_to_int].astype(int)
    
    col_order = ['position', 'team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'xG_diff', 'npxG', 'xGA', 'xGA_diff', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts', 'xpts_diff']
    full_stat = full_stat[col_order]
    full_stat = full_stat.set_index('position')
    # print(full_stat.head(20))
    
    season_data[season] = full_stat
  
  df_season = pd.concat(season_data)
  full_data[league] = df_season
  
data = pd.concat(full_data)
data.head()
  


# ## Exporting data to CSV file

# In[ ]:


data.to_csv('understat.com.csv')

