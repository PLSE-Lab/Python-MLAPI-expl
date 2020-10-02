#!/usr/bin/env python
# coding: utf-8

# # Web Scraping

# ##### 1) Web Scrape of Players' Ratings

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
get_ipython().run_line_magic('matplotlib', 'inline')


pd.options.display.max_columns = None #Display all columns


# In[ ]:


# Web Scrap the list of Teams

resp = requests.get('https://www.2kratings.com/nba-2k19-teams')
soup = BeautifulSoup(resp.text, 'html.parser')
# table_tag = soup.find('table')
rows = soup.findAll('a')
Teams = []
for i in rows[23:53]:
    Teams.append(i.text.lower().replace(' ','-'))
Teams


# In[ ]:


# Web scrape name and ratings according to each team

players =[]
ratings = []
height = []
shoe = []

for team in Teams:

    resp = requests.get('https://www.2kratings.com/nba2k19-team/' + team)
    if resp.status_code == 200:
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        table_tag = soup.find('table')
        rows = table_tag.findAll('tr')
        player_rows = rows
        for row in player_rows[1:]:
    # find name
            td_tags = row.findAll('a')
            for name in td_tags:
                players.append(name.text)

    #find rating
            td_tags = row.findAll('span')
            for rating in td_tags:
                ratings.append(rating.text)
                
                
        rows1 = table_tag.findAll('td') 
        for row in rows1[1:]:
    # find height & shoe
            td_tags = row.findAll('td')
            for line,value in enumerate(td_tags):
                if line == 0:
                    height.append(value.text)
                else:
                    shoe.append(value.text)
        
        rating_df = pd.DataFrame( {'Player': players , 'Ratings' : ratings, 'Shoes' : shoe})
    else:
        print('link error')


# In[ ]:


rating_df.head()


# In[ ]:


rating_df.describe()


# ##### 2) Web Scrape of Players' Salaries

# In[ ]:


# Get salary
resp = requests.get('https://hoopshype.com/salaries/players/')
soup = BeautifulSoup(resp.text, 'html.parser')
rows = soup.findAll('tr')
mm= []
new= []
for i in rows:
    mm.append(i.text.strip().replace('\t','').replace(',,',''))

for i in range(0,len(mm)):
    new.append(mm[i].replace('\n',',').split(',,'))

salary = pd.DataFrame(new[1:])
salary.columns=['1','2','3','4','5','6','7','8','9']
salary = salary.drop('1', axis = 1)
salary = salary.drop('2', axis = 1)
salary.columns = (new[0])
salary['2018/19'] = salary['2018/19'].str.replace(',','').str.replace('$','').astype(int)
def convert(x):
    return '${:,}'.format(x)
salary['2018/19'] = salary['2018/19'].apply(convert)
salary1 = pd.DataFrame()
salary1['Player'] = salary['Player'] 
salary1['2018/19'] = salary['2018/19']
salary1.head()


# In[ ]:


salary1.describe()


# ##### 3) Web Scrape of Players' Stats

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from selenium import webdriver
from pandas import *
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import *


# In[ ]:


browser = webdriver.Chrome(executable_path="INSERT FILE PATH/chromedriver.exe")


# In[ ]:


url = 'https://stats.nba.com/leaders/?Season=2018-19&SeasonType=Regular%20Season'
browser.get(url)


# In[ ]:


# select filters using xpath 
browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div/label/select/option[1]').click()
browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[2]/div/div/label/select/option[2]').click()
browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[3]/div/div/label/select/option[2]').click()
browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[4]/div/div/label/select/option[10]').click()
# Select 'ALL' in show pages
browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[3]/div/div/select/option[1]').click()


# In[ ]:


table = browser.find_element_by_class_name('nba-stat-table__overflow')


# In[ ]:


table.text.split('\n')


# In[ ]:


player_ids = []
player_names = []
player_stats = []

for line_id, lines in enumerate(table.text.split('\n')):
    if line_id == 0:
        column_names = lines.split(' ')[1:]
    else:
        if line_id % 3 == 1:
            player_ids.append(lines)
        if line_id % 3 == 2:
            player_names.append(lines)
        if line_id % 3 == 0:
            player_stats.append( [float(i) for i in lines.split(' ')] )


# In[ ]:


stats = pandas.DataFrame({'Player': player_names,
                       'gp': [i[0] for i in player_stats],
                       'min': [i[1] for i in player_stats],
                       'pts': [i[2] for i in player_stats],
                       'fgm': [i[3] for i in player_stats], 
                       'fga': [i[4] for i in player_stats],
                       'fg%': [i[5] for i in player_stats],
                       '3pm': [i[6] for i in player_stats],
                       '3pa': [i[7] for i in player_stats],
                       '3p%': [i[8] for i in player_stats],
                       'ftm': [i[9] for i in player_stats],
                       'fta': [i[10] for i in player_stats],
                       'ft%': [i[11] for i in player_stats],
                       'oreb': [i[12] for i in player_stats],
                       'dreb': [i[13] for i in player_stats],
                       'reb': [i[14] for i in player_stats],
                       'ast': [i[15] for i in player_stats],
                       'stl': [i[16] for i in player_stats],
                       'blk': [i[17] for i in player_stats],
                       'tov': [i[18] for i in player_stats],
                       'eff': [i[19] for i in player_stats]
                       }
                     )
                     

#One annoying thing is that all the column names are getting re-ordered in alphabetical order. So we're going to reorder this by the following line:
#```python
stats = stats[['Player', 
         'gp', 
         'min', 
         'pts', 
         'fgm', 
         'fga', 
         'fg%', 
         '3pm', 
         '3pa', 
         '3p%', 
         'ftm',
         'fta', 
         'ft%', 
         'oreb', 
         'dreb',
         'reb',
         'ast',
         'stl',
         'blk',
         'tov',
         'eff']
      ]


# ##### 4) Web Scrape of Players' Biography

# In[ ]:


browser1 = webdriver.Chrome(executable_path="INSERT FILE PATH/chromedriver.exe")


# In[ ]:


url1 = 'https://stats.nba.com/players/bio/?Season=2018-19&SeasonType=Regular%20Season'
browser1.get(url1)


# In[ ]:


# select filters using xpath 
browser1.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div/label/select/option[1]').click()
browser1.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[2]/div/div/label/select/option[2]').click()
browser1.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[3]/div/div/label/select/option[2]').click()
# browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[4]/div/div/label/select/option[1]').click()
# Select 'ALL' in show pages
browser1.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[3]/div/div/select/option[1]').click()


# In[ ]:


table1 = browser1.find_element_by_class_name('nba-stat-table__overflow')


# In[ ]:


table1.text.split('\n')


# In[ ]:


player_name = []
player_bio = []

for line_id, lines in enumerate(table1.text.split('\n')):
    if line_id == 0:
        column_names = lines.split(' ')[1:]
    else:
        if line_id % 2 == 0:
            player_bio.append( [i for i in lines.split(' ')] )
        else:
            player_name.append(lines)


# In[ ]:


bio = pandas.DataFrame({'Player': player_name,
                       'Team': [i[0] for i in player_bio],
                       'Age': [i[1] for i in player_bio],
                       'Height': [i[2] for i in player_bio],
                       'Weight': [i[3] for i in player_bio], 
                       'College': [i[4] for i in player_bio],
                       'Country': [i[5] for i in player_bio],
                       'Draft Year': [i[6] for i in player_bio],
                       'Draft Round': [i[7] for i in player_bio],
                       'Draft Number': [i[8] for i in player_bio],

                       }
                     )
                     

#One annoying thing is that all the column names are getting re-ordered in alphabetical order. So we're going to reorder this by the following line:
#```python
bio = bio[['Player', 
         'Team', 
         'Age', 
         'Height', 
         'Weight', 
         'College', 
         'Country', 
         'Draft Year', 
         'Draft Round', 
         'Draft Number',
          ]
      ]


# ### Combine data sets

# In[ ]:


df1 = pd.merge(rating_df, salary1, how='inner', on=['Player'])
df2= pd.merge(df1, stats, how='inner', on=['Player'])
df = pd.merge(df2, bio, how = 'inner', on = ['Player'])
df.head()


# In[ ]:


# Reordering

df = df[['Player', 
         'Ratings',
         'Team',
         'Age', 
         'Height', 
         'Weight', 
         'College', 
         'Country', 
         'Draft Year', 
         'Draft Round', 
         'Draft Number',
         'Shoes',
         '2018/19',
         'gp', 
         'min', 
         'pts', 
         'fgm', 
         'fga', 
         'fg%', 
         '3pm', 
         '3pa', 
         '3p%', 
         'ftm',
         'fta', 
         'ft%', 
         'oreb', 
         'dreb',
         'reb',
         'ast',
         'stl',
         'blk',
         'tov',
         'eff']
      ]


# ### Convert to csv

# In[ ]:


df.to_csv(r'NBA Stats.csv')


# In[ ]:


opens = pd.read_csv('NBA Stats.csv',index_col = 0)
opens

