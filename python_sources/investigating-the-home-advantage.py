#!/usr/bin/env python
# coding: utf-8

# The Numbers behind the home advantage

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from numpy import random


# In[ ]:


def compute_points(home_diff):
    if home_diff>0:
        return 3
    elif home_diff == 0:
        return 1
    else:
        return 0


# In[ ]:


#Data loading
con = sqlite3.connect('../input/database.sqlite')
countries = pd.read_sql_query("SELECT * from Country", con)
matches = pd.read_sql_query("SELECT * from Match", con)
leagues = pd.read_sql_query("SELECT * from League", con)
teams = pd.read_sql_query("SELECT * from Team", con)


# In[ ]:


sub_matches = matches[['country_id','league_id','season','stage','date','match_api_id',
         'home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']]
tm_merge_x=pd.merge(teams,sub_matches, left_on="team_api_id",right_on="home_team_api_id",suffixes=('_home', ''))
tm_merge=pd.merge(teams,tm_merge_x, left_on="team_api_id",right_on="away_team_api_id",suffixes=('_away', ''))
tml_merge=pd.merge(leagues,tm_merge, left_on="id",right_on="league_id")
tmlc_merge=pd.merge(countries,tml_merge, left_on="id",right_on="country_id_x")
tmlc_merge['home_diff'] = tmlc_merge['home_team_goal']-tmlc_merge['away_team_goal']
tmlc_merge['away_diff'] = tmlc_merge['away_team_goal']-tmlc_merge['home_team_goal']
tmlc_merge.drop(['id_x','country_id_x','country_id_y','id_y','team_api_id',
                 'league_id','team_api_id_away','id_away',
                 'id','match_api_id','home_team_api_id',
                 'away_team_api_id','team_short_name_away','team_short_name'],axis=1,inplace=True)
tmlc_merge['num_home_points'] = tmlc_merge[['home_diff']].applymap(compute_points)
tmlc_merge['num_away_points'] = tmlc_merge[['away_diff']].applymap(compute_points)

tmlc_merge.rename(columns = {'name_y':'league'},inplace=True)

#####################
tml_home = tmlc_merge.groupby(['league','team_long_name','num_home_points'])['team_long_name'].count().unstack().reset_index()
tml_home.columns.name = None
tml_home.columns = ['league','team_name','loss','draw','win']

#####################
tml_away = tmlc_merge.groupby(['league','team_long_name_away','num_away_points'])['team_long_name_away'].count().unstack().reset_index()
tml_away.columns.name = None
tml_away.columns = ['league','team_name','loss','draw','win']

#####################
tml_hm_away = pd.merge(tml_home,tml_away,on='team_name',suffixes=('_home','_away'))
tml_hm_away.drop('league_away',inplace=True,axis=1)
tml_hm_away.rename(columns = {'league_home':'league'},inplace=True)
tml_hm_away['total_games']= tml_hm_away.sum(axis=1)
tml_hm_away['total_home']= tml_hm_away['win_home']+tml_hm_away['draw_home']+tml_hm_away['loss_home']
tml_hm_away['total_away']= tml_hm_away['win_away']+tml_hm_away['draw_away']+tml_hm_away['loss_away']

#####################
tml_hm_away.dropna(inplace=True)
tml_hm_away['lh_prop']= tml_hm_away['loss_home']/tml_hm_away['total_home']
tml_hm_away['dh_prop']= tml_hm_away['draw_home']/tml_hm_away['total_home']
tml_hm_away['wh_prop']= tml_hm_away['win_home']/tml_hm_away['total_home']
tml_hm_away['la_prop']= tml_hm_away['loss_away']/tml_hm_away['total_away']
tml_hm_away['da_prop']= tml_hm_away['draw_away']/tml_hm_away['total_away']
tml_hm_away['wa_prop']= tml_hm_away['win_away']/tml_hm_away['total_away']

tml_final = tml_hm_away[['league','team_name','lh_prop','dh_prop','wh_prop','la_prop','da_prop','wa_prop']]


# In[ ]:


tml_final.hist(bins=20,figsize=(10, 10))


# In[ ]:


tml_final.describe()


# In[ ]:


tml_final.groupby('league')['lh_prop','dh_prop','wh_prop','la_prop','da_prop','wa_prop'].mean()


# In[ ]:


tml_final.groupby('league')['lh_prop','dh_prop','wh_prop','la_prop','da_prop','wa_prop'].mean().plot(kind='bar',figsize=(10,4))


# ### Home advantage could be truth.
# Teams win an average of 41% of home games compared with 25% of away games. <br\>
# The home advantage is most pronounced in the Netherlands Eredivisie and Spain LIGA BBVA. <br\>
# Home and away draws are equal on the average at 26% and 25% respectively.<br/>
# Home wins and away losses are the most frequent outcomes for any given league.<br/>
# The Scottish league is very wierd. The home advantage seems less significant.

# In[ ]:


tml_points = tml_hm_away.copy(deep=True)
tml_points["home_points"] = tml_points["win_home"]*3 + tml_points["draw_home"]
tml_points["away_points"] = tml_points["win_away"]*3 + tml_points["draw_away"]
#######
teams_data = tml_points[["league","team_name","home_points","away_points","total_games",'total_home',"total_away"]].copy(deep=True)
teams_data["home_ppg"] = teams_data["home_points"]/teams_data["total_home"]
teams_data["away_ppg"] = teams_data["away_points"]/teams_data["total_away"]
teams_data["total_ppg"] = (teams_data["away_points"]+teams_data["home_points"])/teams_data["total_games"]


# In[ ]:


teams_data.mean()


# In[ ]:


ttest_ind(teams_data["away_ppg"],teams_data["total_ppg"])


# In[ ]:


ttest_ind(teams_data["home_ppg"],teams_data["total_ppg"])


# Teams average 1.48 points per game at home , 0.99 away and 1.23 in total<br/>
# With a pvalue of 4.17E-14 this is very unlikely to be due to chance. Playing at home is almost certainly an advantage.

# In[ ]:


tml_prb = tml_hm_away.groupby('league').sum().reset_index()
tml_prb['p_h_h'] = ((tml_prb['win_home']/tml_prb['total_games'])*((tml_prb['win_home']-1)/(tml_prb['total_games']-1)))/(tml_prb['win_home']/tml_prb['total_games'])
tml_prb['p_a_h'] = ((tml_prb['win_away']/tml_prb['total_games'])*(tml_prb['win_home']/(tml_prb['total_games']-1)))/(tml_prb['win_home']/tml_prb['total_games'])
tml_prb['p_d_h'] = (((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])*(tml_prb['win_home']/(tml_prb['total_games']-1)))/(tml_prb['win_home']/tml_prb['total_games'])
tml_prb['p_h_d'] = (((tml_prb['draw_away']+tml_prb['draw_home'])/(tml_prb['total_games']-1))*(tml_prb['win_home']/(tml_prb['total_games'])))/((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])
tml_prb['p_a_d'] = (((tml_prb['draw_away']+tml_prb['draw_home'])/(tml_prb['total_games']-1))*(tml_prb['win_away']/(tml_prb['total_games'])))/((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])
tml_prb['p_d_d'] = (((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])*(((tml_prb['draw_away']+tml_prb['draw_home'])-1)/(tml_prb['total_games']-1)))/((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])
tml_prb['p_h_a'] = ((tml_prb['win_home']/tml_prb['total_games'])*(tml_prb['win_away']/(tml_prb['total_games']-1)))/(tml_prb['win_away']/tml_prb['total_games'])
tml_prb['p_a_a'] = ((tml_prb['win_away']/tml_prb['total_games'])*((tml_prb['win_away']-1)/(tml_prb['total_games']-1)))/(tml_prb['win_away']/tml_prb['total_games'])
tml_prb['p_d_a'] = (((tml_prb['draw_away']+tml_prb['draw_home'])/tml_prb['total_games'])*(tml_prb['win_away']/(tml_prb['total_games']-1)))/(tml_prb['win_away']/tml_prb['total_games'])
tml_avg_prob_distributions = tml_prb[['league','p_h_h','p_a_h','p_d_h','p_h_d','p_a_d','p_d_d','p_h_a','p_a_a','p_d_a']]
tml_avg_prob_distributions.set_index(['league'],inplace=True)


# In[ ]:


tml_avg_prob_distributions['norm_phh'] = tml_avg_prob_distributions['p_h_h']/(tml_avg_prob_distributions['p_h_h']+tml_avg_prob_distributions['p_a_h']+tml_avg_prob_distributions['p_d_h'])
tml_avg_prob_distributions['norm_pah'] = tml_avg_prob_distributions['p_a_h']/(tml_avg_prob_distributions['p_a_h']+tml_avg_prob_distributions['p_a_h']+tml_avg_prob_distributions['p_d_h'])
tml_avg_prob_distributions['norm_pdh'] = tml_avg_prob_distributions['p_d_h']/(tml_avg_prob_distributions['p_d_h']+tml_avg_prob_distributions['p_a_h']+tml_avg_prob_distributions['p_d_h'])

tml_avg_prob_distributions['norm_phd'] = tml_avg_prob_distributions['p_h_d']/(tml_avg_prob_distributions['p_h_d']+tml_avg_prob_distributions['p_a_d']+tml_avg_prob_distributions['p_d_d'])
tml_avg_prob_distributions['norm_pad'] = tml_avg_prob_distributions['p_a_d']/(tml_avg_prob_distributions['p_h_d']+tml_avg_prob_distributions['p_a_d']+tml_avg_prob_distributions['p_d_d'])
tml_avg_prob_distributions['norm_phd'] = tml_avg_prob_distributions['p_d_d']/(tml_avg_prob_distributions['p_d_d']+tml_avg_prob_distributions['p_a_d']+tml_avg_prob_distributions['p_d_d'])

tml_avg_prob_distributions['norm_pha'] = tml_avg_prob_distributions['p_h_a']/(tml_avg_prob_distributions['p_h_a']+tml_avg_prob_distributions['p_a_a']+tml_avg_prob_distributions['p_d_a'])
tml_avg_prob_distributions['norm_paa'] = tml_avg_prob_distributions['p_a_a']/(tml_avg_prob_distributions['p_h_a']+tml_avg_prob_distributions['p_a_a']+tml_avg_prob_distributions['p_d_a'])
tml_avg_prob_distributions['norm_pda'] = tml_avg_prob_distributions['p_d_a']/(tml_avg_prob_distributions['p_h_a']+tml_avg_prob_distributions['p_a_a']+tml_avg_prob_distributions['p_d_a'])


# In[ ]:


tml_avg_prob_distributions[['norm_phh', 'norm_pah', 'norm_pdh', 'norm_phd', 'norm_pad',
       'norm_pha', 'norm_paa', 'norm_pda']].T


# ### Compare Conditional Probabilities across leagues

# In[ ]:


tml_avg_prob_distributions[['norm_phh', 'norm_pah', 'norm_pdh', 'norm_phd', 'norm_pad',
       'norm_pha', 'norm_paa', 'norm_pda']].T.plot(kind='bar',figsize=(10,4))


# ### Compare Conditional Probabilities pairs for each league

# In[ ]:


tml_avg_prob_distributions[['norm_phh', 'norm_pah', 'norm_pdh', 'norm_phd', 'norm_pad',
       'norm_pha', 'norm_paa', 'norm_pda']].plot(kind='bar',figsize=(10,4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




