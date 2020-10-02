#!/usr/bin/env python
# coding: utf-8

# I wait for long long time to open up new Competition!  
# I will conduct EDA per each csv file at first, then I merge them to analyze seeing the treasure in the data.  
# Without further do, Let' go!
# 
# ## 180223 Update: Section 2 & A team with a good defense wins.
# 
# 1. Section 1 File EDA  
#     1.1. Team  
#     1.2. Seasons  
#     1.3. NCAA Tourney Seeds  
#     1.4. Regular Season Compact Results  
#     1.5. NCAA Tourney Compact Results  
# 2. Section 2 File EDA  
#     2.1. Regular Season Detail Results  
#     2.2. NCAA Season Detail Results  
# ---
# Import Libary & helper function

# In[ ]:


def preprocess():
    compact_season.rename(columns = {'Season':'year'}, inplace = True)
    compact_season = pd.merge(compact_season, zero_day, on = 'year')
    compact_season['game_date'] = compact_season.apply(lambda x: x['dayZero'] + pd.DateOffset(days = x['DayNum']), axis =1)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from collections import *
from scipy.stats import ttest_1samp as ttest


from subprocess import check_output

def get_path(path):
    data_dir = '../input/'
    return data_dir + path

file_list = check_output(["ls", "../input"]).decode("utf8")

file_list = file_list.split()
file_dict = defaultdict(list)
key_file = ['Events','NCAAT','Players','Regular','Team']
for x in file_list:
    etc = True
    for key in key_file:
        if x.startswith(key):
            file_dict[key].append(x)
            ect = False
    if etc:
        file_dict['ETC'].append(x)

print('Event Files: ' ,file_dict['Events'])


# #### 1.1. Teams

# In[ ]:


path = 'Teams.csv'
team = pd.read_csv(get_path(path))
team.head(2)


# In[ ]:


with plt.style.context('Solarize_Light2'):
    f, ax = plt.subplots(1,2, figsize = (12,4))
    first_season = team['FirstD1Season'].copy()
    cut_season = [x for x in range(1980, 2025, 5)]
    label_season = [x for x in cut_season if x > 1980]
    first_season = pd.cut(first_season, bins = cut_season, labels = label_season)
    sns.countplot(first_season, ax = ax[0])
    ax[0].set_title('The CountPlot of FirstSeason', size = 12)

    bg_ed_season_size = team.groupby(['FirstD1Season', 'LastD1Season']).size().reset_index()
    ax[1].scatter(x = bg_ed_season_size['FirstD1Season'], y = bg_ed_season_size['LastD1Season'], s = bg_ed_season_size[0])
    ax[1].set_title('Team Season', size = 12)
    ax[1].set_xlabel('Starting Year', size = 10)
    ax[1].set_ylabel('Ending Year', size = 10)
    #print('Number of University Keep going the competition in the Leadgue ', (team['LastD1Season'] == 2018).sum())
    #print('Number of University living from 1985 to 2018 ', ((team['FirstD1Season']== 1985) & (team['LastD1Season'] == 2018)).sum())
    plt.subplots_adjust(0,0,1, 0.8,wspace = 0.3)
    plt.suptitle('Team Status', size = 15)
    plt.show()


# Find:  
# - Most of Team starts at the beginning 1985.
# - The number of the team where keeps their status from the beginning : 274
# - The current number of the team in the league : 351
# - Continuously some teams went out of the compeition, and some team that improve their skill joined in the league
# 
# ---
# 
# #### 1.2. Seasons

# In[ ]:


path = 'Seasons.csv'
season = pd.read_csv(get_path(path))
zero_day = pd.to_datetime(season['DayZero'])
zero_day = pd.DataFrame({'dayZero': zero_day, 'year': zero_day.dt.year})
season.head(2)


# In[ ]:


cut_season = [x for x in range(1980, 2025, 5)]
label_season = [x for x in cut_season if x > 1980]
season['catgorized_season'] = pd.cut(season['Season'], bins = cut_season, labels = label_season)
with plt.style.context('Solarize_Light2'):
    plt.figure(figsize = (12, 4))
    tmp_region = pd.concat((season.iloc[:,x] for x in range(2,6)))
    tmp_region_cnt = tmp_region.value_counts()
    tmp_region_cnt.plot(kind='bar')
    plt.title('Region Count on Semi Final')
    plt.show()


# Find:  
# - East, West, Midwest, South, Southeast are traditional power region.  
# 
# Guessing:  
# - Even though I don't know well of the region of USA, I know Chicago (16th val) in the West! I suspects when two team on the same region come up to the semi-final, one of two team was written by their region.
# 
# ---
# 
# #### 1.3. NCAATourneySeeds

# In[ ]:


path = 'NCAATourneySeeds.csv'
seed = pd.read_csv(get_path(path))
seed.head(2)


# In[ ]:


test_seed = seed.groupby('Season').size()
x = test_seed.index
height = test_seed.values
with plt.style.context('bmh'):
    plt.figure(figsize = (12,4))
    plt.bar(x = x, height = height)
    plt.ylim(60, 70)
    plt.axvline(2000.5, color = 'k', linewidth = 3)
    plt.axvline(2010.5, color = 'k', linewidth = 3)
    plt.title('The Number of Seed per Year')
    plt.show()


# Find:  
# - Just four region could have seed. As a result, "South" and "Southeast" get in or get out frequently.
# - The unique seed number is [64, 65, 68]
# - The interval of the seed number is '64 Seed : 1985-2000 / 65 Seed : 2001 - 2010 / 68 Seed : 2011 - 2017')
# - (For me, 'Solarize_Light2' is comfortable to see..)
# 
# ---
# 
# #### 1.4. RegularSeasonCompactResult

# In[ ]:


path = 'RegularSeasonCompactResults.csv'
compact_season = pd.read_csv(get_path(path))
compact_season.head(2)


# In[ ]:


with plt.style.context('Solarize_Light2'):
    f, ax = plt.subplots(1,2, figsize = (12,4))
    compact_season['score_diff'] = compact_season['WScore'] - compact_season['LScore']
    test = compact_season.groupby('DayNum')
    test_diff_time = test['score_diff'].mean()
    ax[0].scatter(x = test_diff_time.index, y = test_diff_time)
    ax[0].set_title('Score Difference & Day', size = 10)
    ax[0].set_xlabel('Day')
    ax[0].set_ylabel('Score Difference')

    test_NumOT = test['NumOT'].mean()
    ax[1].scatter(x = test_NumOT.index, y = test_NumOT)
    ax[1].set_title('Overtime periods & Day', size = 10)
    ax[1].set_xlabel('Day')
    ax[1].set_ylabel('Number of Overtime perios')
    
    plt.subplots_adjust(0,0,1,0.8, wspace = 0.2)
    plt.suptitle('Compact Season', size = 12)
    plt.show()


# Find:  
# - Over time, there is a big gap between the teams. Score gap was smaller with over time.
# - However, the overtime had not severe difference.
# - The resaon that the first few values is small is there is no game or special game.
# 
# ---
# 
# #### 1.5. NCAA Tournament Compact Result

# In[ ]:


path = 'NCAATourneyCompactResults.csv'
Ncompact_season = pd.read_csv(get_path(path))
Ncompact_season.head(2)


# In[ ]:


with plt.style.context('Solarize_Light2'):
    f, ax = plt.subplots(1,2, figsize = (12,4))
    Ncompact_season['score_diff'] = Ncompact_season['WScore'] - Ncompact_season['LScore']
    
    ax[0].scatter(x = Ncompact_season['DayNum'], y = Ncompact_season['score_diff'])
    ax[0].set_title('Score Difference & Day', size = 10)
    ax[0].set_xlabel('Day')
    ax[0].set_ylabel('Score Difference')
    
    test = Ncompact_season.groupby(['DayNum', 'NumOT']).size().reset_index()
    ax[1].scatter(x = test['DayNum'], y = test['NumOT'], s = test[0])
    ax[1].set_title('Overtime periods & Day', size = 10)
    ax[1].set_xlabel('Day')
    ax[1].set_ylabel('Number of Overtime perios')
    
    plt.subplots_adjust(0,0,1,0.8, wspace = 0.2)
    plt.suptitle('NCAA Compact Season', size = 14)
    plt.show()


# Find:  
# - The Score difference tends to decreases. However, the some of the result over 30 is shocking, since they are the most power team in America at least 64!
# - There are no overtime on the most of day.
# 
# ---
# 
# ### 2. Section 2 File
# 
# #### 2.1. Detail Results on the regular season

# In[ ]:


path = "RegularSeasonDetailedResults.csv"
detail_result = pd.read_csv(get_path(path))
detail_result_p = detail_result.drop(['WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT'], axis = 1)

def ratio_npt(detail_result_p, get_name, col1, tot_cnt): 
    detail_result_p[get_name] = detail_result_p[col1] / detail_result_p[tot_cnt]
    
def preprocess2(detail_result_p):
#2 point number
    detail_result_p['W_FGM2'] = detail_result_p['WFGM'] - detail_result_p['WFGM3']
    detail_result_p['L_FGM2'] = detail_result_p['LFGM'] - detail_result_p['LFGM3']
#total score
    detail_result_p['W_SCORE'] = 2 * detail_result_p['W_FGM2'] + 3 * detail_result_p['WFGM3'] + detail_result_p['WFTM']
    detail_result_p['L_SCORE'] = 2 * detail_result_p['L_FGM2'] + 3 * detail_result_p['LFGM3'] + detail_result_p['LFTM']

#ratio_npt (1,2,3)
    detail_result_p['W_NUM_Goal'] = detail_result_p[['WFTM', 'W_FGM2', 'LFGM3']].sum(axis = 1)
    detail_result_p['L_NUM_Goal'] = detail_result_p[['LFTM', 'L_FGM2', 'LFGM3']].sum(axis = 1)
    ratio_pt = ['W_ratio_1pt', 'W_ratio_2pt', 'W_ratio_3pt', 'L_ratio_1pt', 'L_ratio_2pt', 'L_ratio_3pt']
    pt_name = ['WFTM', 'W_FGM2', 'WFGM3', 'LFTM', 'L_FGM2', 'LFGM3']
    tot_cnt_name = ['W_NUM_Goal'] * 3 + ['L_NUM_Goal'] * 3
    for get_name, col1, tot_cnt in zip(ratio_pt, pt_name, tot_cnt_name): ratio_npt(detail_result_p, get_name, col1, tot_cnt)
    
    detail_result_p['W_FGA2'] = detail_result_p['WFGA'].subtract(detail_result_p['WFGA3'], axis = 0)
    detail_result_p['L_FGA2'] = detail_result_p['LFGA'].subtract(detail_result_p['LFGA3'], axis = 0)
    
    name_col = set(col[1:] for col in detail_result_p.columns[3:] if col[0] == 'W')
    for col in name_col: detail_result_p['D'+ col] = detail_result_p['W' + col].subtract(detail_result_p['L' + col], axis = 0)

preprocess2(detail_result_p)
detail_result_p.head(2)


# - T-test; To check the mean 0 or not regard of the difference Win_val - Lose_val 

# In[ ]:


Difference = list(set(col for col in detail_result_p.columns if col[0] == 'D'))
def ttest_apply(cols): return ttest(cols, 0)[1]
#t = day_detail.agg({diff : 'mean' for diff in Difference})
#tmp = t.apply(ttest_apply, axis = 0)
tmp = detail_result_p[Difference].apply(ttest_apply, axis = 0)
if tmp[tmp > 0.05].empty: print('Every Val was different according to the competition')
else: print(tmp.index[tmp > 0.05], ' was having the same distribution')


# In[ ]:


with plt.style.context('Solarize_Light2'):
    
    f, ax = plt.subplots(2,3,figsize = (12,8))
    day_detail = detail_result_p.groupby('DayNum')
    t = day_detail.agg({'WFGA' : 'mean', 'LFGA' : 'mean'})
    ax_part = ax[0,0]
    t.plot(kind = 'line', ax = ax_part, legend = True, use_index = False)
    ax_part.set_title('Number of Trial in the Game', size = 12)
    ax_part.set_ylabel('Count')
    
    ratio_pt = ['W_ratio_1pt', 'W_ratio_2pt', 'W_ratio_3pt', 'L_ratio_1pt', 'L_ratio_2pt', 'L_ratio_3pt']
    t = day_detail.agg({x: 'mean' for x in ratio_pt})
    t[ratio_pt[0::3]].plot(ax = ax[0,1], legend = True, use_index = False)
    ax[0,1].set_title('Free Draw Ratio', size = 10)
    t[ratio_pt[1::3]].plot(ax = ax[1,0], legend = True, use_index = False)
    ax[1,0].set_title('Field Goal Ratio', size = 10)
    t[ratio_pt[2::3]].plot(ax = ax[1,1], legend = True)
    ax[1,1].set_title('Three Goal Ratio', size = 10)
    
    t = day_detail.agg({'WFGA': 'sum', 'WFGM':'sum', 'LFGA': 'sum', 'LFGM' : 'sum'})
    t1 = t['WFGM'].divide(t['WFGA'])
    t2 = t['LFGM'].divide(t['LFGA'])
    ax_part = ax[0,2]
    t = pd.DataFrame(OrderedDict({'Win': t1.values, 'Los' :t2.values}))
    t.plot(ax = ax_part, color = ['red', 'grey'], legend = True)
    ax_part.set_title('Success Ratio of winner / loser', size = 12)
    ax_part.set_ylabel('Ratio')
    ax_part.legend(bbox_to_anchor = (1, 0.2))
    
    t = day_detail.agg({'WPF':'mean', 'LPF':'mean'})
    t.plot(ax = ax[1,2],legend = True, color = ['red', 'grey'])
    ax[1,2].set_title('Number of Foul', size = 10)
    
    plt.show()


# Find:  
# - T-Test: Nothing to be in same distribution
# - Pic (0,2): The success ratio of the shooting trail is higher in the Winner, then the loser.
# - Where the team likes defense won the games more like Murinho in EPL.
#     - Pic (0,0): Over the season, losing team tried to shoot more times than winner. 
#     - Pic (0,1): Winner has more chance to do free draw. 
#     - Pic (1,2): Winner tended not to take a foul.

# In[ ]:


#WOR(opensive rebound), WAst(assit), WTO(turnovers), WStl(steal) ~ W_FGM2
#WBlk, WFoul,WDR ~ Defense


# #### 2.2. NCAA Detail Result

# In[ ]:


path = "NCAATourneyDetailedResults.csv"
detail_result = pd.read_csv(get_path(path))
detail_result_p = detail_result.drop(['WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT'], axis = 1)
preprocess2(detail_result_p)
detail_result_p.head(2)


# - T-test

# In[ ]:


Difference = list(set(col for col in detail_result_p.columns if col[0] == 'D'))
def ttest_apply(cols): return ttest(cols, 0)[1]
#t = day_detail.agg({diff : 'mean' for diff in Difference})
#tmp = t.apply(ttest_apply, axis = 0)
tmp = detail_result_p[Difference].apply(ttest_apply, axis = 0)
if tmp[tmp > 0.05].empty: print('Every Val was different according to the competition')
else: print(tmp.index[tmp > 0.05], ' was having the same distribution')


# In[ ]:


with plt.style.context('Solarize_Light2'):
    
    f, ax = plt.subplots(2,3,figsize = (12,8))
    day_detail = detail_result_p.groupby('DayNum')
    t = day_detail.agg({'WFGA' : 'mean', 'LFGA' : 'mean'})
    ax_part = ax[0,0]
    t.plot(kind = 'line', ax = ax_part, legend = True, use_index = False, color = ['red', 'grey'])
    ax_part.set_title('Number of Trial in the Game', size = 12)
    ax_part.set_ylabel('Count')
    
    ratio_pt = ['W_ratio_1pt', 'W_ratio_2pt', 'W_ratio_3pt', 'L_ratio_1pt', 'L_ratio_2pt', 'L_ratio_3pt']
    t = day_detail.agg({x: 'mean' for x in ratio_pt})
    t[ratio_pt[0::3]].plot(ax = ax[0,1], legend = True, use_index = False, color = ['red', 'grey'])
    ax[0,1].set_title('Free Draw Ratio', size = 10)
    t[ratio_pt[2::3]].plot(ax = ax[1,1], legend = True, color = ['black', 'grey'])
    ax[1,1].set_title('Three Goal Ratio', size = 10)
    
    t = day_detail.agg({'W_FGA2' : 'mean', 'L_FGA2' : 'mean'})
    t.plot(ax = ax[1,0], legend = True, use_index = False, color = ['black', 'gray'])
    ax[1,0].set_title('Trial of Field Goal', size = 10)
    
    t = day_detail.agg({'WFGA': 'sum', 'WFGM':'sum', 'LFGA': 'sum', 'LFGM' : 'sum'})
    t1 = t['WFGM'].divide(t['WFGA'])
    t2 = t['LFGM'].divide(t['LFGA'])
    ax_part = ax[0,2]
    t = pd.DataFrame(OrderedDict({'Win': t1.values, 'Los' :t2.values}))
    t.plot(ax = ax_part, color = ['red', 'grey'], legend = True)
    ax_part.set_title('Success Ratio of winner / loser', size = 12)
    ax_part.set_ylabel('Ratio')
    ax_part.legend(bbox_to_anchor = (1, 0.2))
    
    t = day_detail.agg({'WPF':'mean', 'LPF':'mean'})
    t.plot(ax = ax[1,2],legend = True, color = ['red', 'grey'])
    ax[1,2].set_title('Number of Foul', size = 10)
    
    plt.show()


# Find:
# - T-test: "Trial to shoot field goal" and "the 3 point goal / total score" was almost identical.
# - The other rules are identical to the regular season.

# ## See you tomorrow! I will prove that the defensive team takes more opportunity to win and analyze to Section 3 File

# In[ ]:




