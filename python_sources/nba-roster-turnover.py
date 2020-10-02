#!/usr/bin/env python
# coding: utf-8

# # NBA roster turnover
# The 2019 NBA season is over and the start of free agency has begun. Free agency marks the period in the NBA offseason when players who are no longer under contract with any team and decide which team they want to play for in the upcoming season. The summer of 2019 marks one of the most volatile free agency periods in memory, with superstar players like Kawhi Leonard, Kevin Durant, and Kyrie Irving all looking for new teams.
# 
# Team chemistry is often discussed as an "immeasurable" metric that is difficult to quanity. This notebook explores correlation between team win totals and total roster turnover from the previous year for that past couple decades. My hypothesis, *a priori*, was that there would be a negative correlation between win totals and roster turnover. The rationale for this was that teams would be less likely to develop team chemsitry if they have many new teammates.

# In[ ]:


get_ipython().run_cell_magic('html', '', '<div class="github-card" data-github="arvkevi/nba-roster-turnover" data-width="600" data-height="100" data-theme="default"></div>\n<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>')


# In[ ]:


get_ipython().system(' pip install --user basketball-reference-web-scraper sportsreference;')


# In[ ]:


import os
import requests

import colorlover as cl
import cufflinks as cf
import numpy as np
import pandas as pd
import plotly
import plotly_express as px
import ipywidgets as widgets

from basketball_reference_web_scraper import client
from bs4 import BeautifulSoup
from collections import Counter
from ipywidgets import interact
from plotly.graph_objs import Scatter
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams

cf.go_offline()
plotly.offline.init_notebook_mode() # run at the start of every notebook


# In[ ]:


def get_player_statistics(season_end_year):
    """Return a DataFrame of a season stats for all players 
    using basketball reference web scraper
    """
    results = client.players_season_totals(season_end_year=season_end_year)
    for result in results:
        result['team'] = result['team'].value
    return pd.DataFrame.from_dict(results)


def calculate_turnover(gb, year1=2000, year2=2001):
    """turnover is defined as sum of the 
    absolute difference between mintues played 
    for each player from the two seasons.
    
    :param gb: A pandas groupby dataframe
    """
    s1 = gb.loc[gb['year'] == year1]
    s1.set_index('slug', inplace=True)
    
    s2 = gb.loc[gb['year'] == year2]
    s2.set_index('slug', inplace=True)
    
    combined = s1.join(s2, how='outer', lsuffix='_year1', rsuffix='_year2').fillna(0)
    turnover = np.abs(combined['minutes_played_year2'] -  combined['minutes_played_year1']).sum()
    return turnover


# ## Scrape raw data and calcualate turnover
# I have precomputed the roster turnover and wins for each team and stored them as ouput.
# This makes the kernel quicker and easier to run.
# 

# In[ ]:


# keep precomputed as False if you want to scrape the basketball-reference site (much slower)
# change to True to run the kernel much faster
precomputed = True


# In[ ]:


if precomputed is True:
    player_minutes = pd.read_csv('../input/nba-roster-turnover/NBA_player_minutes.2004-2019.csv')
    roster_turnover = pd.read_csv('../input/nba-roster-turnover/NBA_roster_turnover_wins.2004-2019.csv')
    roster_turnover.set_index('team', inplace=True)

else:
    # This DataFrame will store minutes played information for every player
    player_minutes = pd.DataFrame()
    # This DataFrame will be used to store team name, year, win total, and roster turnover
    roster_turnover = pd.DataFrame()
# hold the correlation values between wins and turnover for a given year
wins_turnover_corr = {}


# In[ ]:


years = range(2004, 2020)
for year in years:
    
    if precomputed is True:
        # calculate the correlation between wins and roster turnover
        wins_turnover_corr[year] = roster_turnover.loc[roster_turnover['year'] == year].corr()['wins']['turnover']
    
    else:
        # scrape basketball reference
        # calculate total wins for each team and store them dictionary
        wins = {}
        teams = Teams(year=year)
        for team in teams:  
            sched = Schedule(team.abbreviation, year=year)
            wins[team.name.upper()] = sched.dataframe['wins'].max()
        wins_df = pd.DataFrame.from_dict(wins, orient='index', columns=['wins'])

        # scrape season stats for every NBA player for the previous year
        previous_season = get_player_statistics(year - 1)
        previous_season['year'] = year - 1

        # scrape season stats for every NBA player for the current year
        current_season = get_player_statistics(year)
        current_season['year'] = year

        # combine the season stats into one DataFrame
        combined = pd.concat([previous_season, current_season])
        # add minutes played to the larger player_minutes DataFrame
        player_minutes = player_minutes.append(combined[['team', 'name', 'slug', 'minutes_played', 'year']])

        # GroupBy the teams to calculate how much roster turnover there is from year to year.
        gb = combined.groupby('team')
        turnover_df = pd.DataFrame(gb.apply(calculate_turnover, year1=year-1, year2=year), columns=['turnover'])

        # join the calculated turnover with the scraped wins totals
        turnover_df = turnover_df.join(wins_df)
        turnover_df['year'] = year

        roster_turnover = roster_turnover.append(turnover_df)

        # calculate the correlation between wins and roster turnover
        wins_turnover_corr[year] = turnover_df.corr()['wins']['turnover']

# always write these to file, because the kernel self-references it's output
player_minutes = player_minutes.drop_duplicates()
player_minutes.to_csv('NBA_player_minutes.2004-2019.csv')
roster_turnover.to_csv('NBA_roster_turnover_wins.2004-2019.csv')


# ### Scrape RGB values for teams' primary colors

# In[ ]:


# scrape a team's primary colors for the graphs below.
raw_team_colors = pd.read_json('https://raw.githubusercontent.com/jimniels/teamcolors/master/src/teams.json')
team_colors = {}

for team in Teams(year=2019):
    team_rgb_strings = raw_team_colors.loc[raw_team_colors['name'] == team.name]['colors'].item()['rgb'][0].split(' ')
    team_colors[team.name.upper()] = tuple(int(c) for c in team_rgb_strings)

# add old teams, the SuperSonics, and New Jersey Nets
team_colors['SEATTLE SUPERSONICS'] = tuple((0, 101, 58))
team_colors['NEW JERSEY NETS'] = tuple((0, 42, 96))
team_colors['NEW ORLEANS HORNETS'] = tuple((0, 119, 139))
team_colors['NEW ORLEANS/OKLAHOMA CITY HORNETS'] = tuple((0, 119, 139))
team_colors['CHARLOTTE BOBCATS'] = tuple((249, 66, 58)) # <--guess


# ### Visualize Each Season

# In[ ]:


@interact
def scatter_plot(year=range(2004, 2020)):
    print(f'Correlation Coefficient for {year}: {wins_turnover_corr[year]}')
    teams = Teams(year=year)
    teams = sorted([team.name.upper() for team in teams])
    teams_colorscale = [f'rgb{team_colors[team]}' for team in teams]
    
    scatter_data = roster_turnover.reset_index()
    scatter_data.loc[scatter_data.reset_index()['year'] == year].dropna().iplot(kind='scatter', 
                                                                       x='turnover', 
                                                                       y='wins',
                                                                       xTitle='Roster Turnover (absolute difference minutes played by player)',
                                                                       yTitle='Wins',
                                                                       mode='markers', 
                                                                       categories='team',
                                                                       title=f'NBA Team Roster Turnover vs Wins in {year}',
                                                                       theme='ggplot', 
                                                                       colors=teams_colorscale,)


# ## Try to generate a static image of 2008 for output/ directory

# In[ ]:


year = 2008
teams = Teams(year=year)
team_names = sorted([team.name.upper() for team in teams])
teams_colorscale = [f'rgb{team_colors[team]}' for team in team_names]

scatter_data = roster_turnover.loc[roster_turnover['year'] == year]

data = []
for team, team_rgb in zip(team_names, teams_colorscale):
    trace = Scatter(
        x = [scatter_data.loc[team]['turnover']],
        y = [scatter_data.loc[team]['wins']],
        mode = 'markers',
        name = team,
        marker = dict(color = team_rgb,
                      size = 14)
    )
    data.append(trace)
    
layout = dict(
    title = 'NBA Team Roster Turnover vs Wins in 2008',
    template = 'ggplot2'
)

fig = dict(data=data, layout=layout)
# Any Kagglers know how to export png so it will show up in output?
#plotly.offline.iplot(fig, filename='NBA_roster_turnover_and_wins.png', image='png')


# ## Check Outlier Seasons

# In[ ]:


def roster_turnover_pivot(player_minutes, team='ATLANTA HAWKS', year=2004): 
    pm_subset = player_minutes.loc[((player_minutes['year'] == year) 
                                     | (player_minutes['year'] == year - 1))
                                    & (player_minutes['team'] == team)]
    return pd.pivot_table(pm_subset, values='minutes_played', index=['team', 'name'], columns=['year']).fillna(0)


# ### 2008 -- Boston's Big 3 -- Ray Allen and Kevin Garnett join Paul Pierce
# 66 Wins

# In[ ]:


roster_turnover.loc[roster_turnover['year'] == 2008].loc['BOSTON CELTICS']


# In[ ]:


boston_big3 = roster_turnover_pivot(player_minutes, team='BOSTON CELTICS', year=2008)


# In[ ]:


# note Ray Allen and Kevin Garnett
boston_big3


# ### 2011 -- Miami's Big 3 -- Lebron James and Chris Bosh join Dywayne Wade
# 58 Wins

# In[ ]:


roster_turnover.loc[roster_turnover['year'] == 2011].loc['MIAMI HEAT']


# In[ ]:


miami_big3 = roster_turnover_pivot(player_minutes, team='MIAMI HEAT', year=2011)


# In[ ]:


# note LeBron and Chris Bosh
miami_big3


# ### Was the hypothesis correct?
# Year over year has showed a negative correlation with team roster turnover and total wins.

# In[ ]:


yoy_corr = pd.DataFrame.from_dict(wins_turnover_corr, orient='index', columns=['correlation_coefficient']).reset_index()
yoy_corr.rename(columns={'index': 'year'}, inplace=True)


# In[ ]:


px.scatter(yoy_corr,
           x='year', 
           y='correlation_coefficient',
           title='Correlation Coefficient Between Win totals and Roster Turnover',
          )

