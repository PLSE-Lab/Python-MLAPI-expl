#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

print('Files:')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Context

# ## Objective: Predict how many yards a play will go for
# - The output isn't a single number, but a list of probabilities for the play resulting in a number of yards <= that many yards. (-99 to 99, making each output an array for 199 values for 0 to 1). Meaning `col_-99` will always be 0 and `col_99` will always be 1
# - This output is then fed through the CRPS algo for a true score

# ## Data Schema
# 
# ### train.csv
# * `GameId` - a unique game identifier
# * `PlayId` - a unique play identifier
# * `Team` - home or away
# * `X` - player position along the long axis of the field. See figure below.
# * `Y` - player position along the short axis of the field. See figure below.
# * `S` - speed in yards/second
# * `A` - acceleration in yards/second^2
# * `Dis` - distance traveled from prior time point, in yards
# * `Orientation` - orientation of player (deg)
# * `Dir` - angle of player motion (deg)
# * `NflId` - a unique identifier of the player
# * `DisplayName` - player's name
# * `JerseyNumber` - jersey number
# * `Season` - year of the season
# * `YardLine` - the yard line of the line of scrimmage
# * `Quarter` - game quarter (1-5, 5 == overtime)
# * `GameClock` - time on the game clock
# * `PossessionTeam` - team with possession
# * `Down` - the down (1-4)
# * `Distance` - yards needed for a first down
# * `FieldPosition` - which side of the field the play is happening on
# * `HomeScoreBeforePlay` - home team score before play started
# * `VisitorScoreBeforePlay` - visitor team score before play started
# * `NflIdRusher` - the NflId of the rushing player
# * `OffenseFormation` - offense formation
# * `OffensePersonnel` - offensive team positional grouping
# * `DefendersInTheBox` - number of defenders lined up near the line of scrimmage, spanning the width of the offensive line
# * `DefensePersonnel` - defensive team positional grouping
# * `PlayDirection` - direction the play is headed
# * `TimeHandoff` - UTC time of the handoff
# * `TimeSnap` - UTC time of the snap
# * `Yards` - the yardage gained on the play (you are predicting this)
# * `PlayerHeight` - player height (ft-in)
# * `PlayerWeight` - player weight (lbs)
# * `PlayerBirthDate` - birth date (mm/dd/yyyy)
# * `PlayerCollegeName` - where the player attended college
# * `HomeTeamAbbr` - home team abbreviation
# * `VisitorTeamAbbr` - visitor team abbreviation
# * `Week` - week into the season
# * `Stadium` - stadium where the game is being played
# * `Location` - city where the game is being player
# * `StadiumType` - description of the stadium environment
# * `Turf` - description of the field surface
# * `GameWeather` - description of the game weather
# * `Temperature` - temperature (deg F)
# * `Humidity` - humidity
# * `WindSpeed` - wind speed in miles/hour
# * `WindDirection` - wind direction

# In[ ]:


df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
df_train.head()


# > # Clean Data
# lifted from https://www.kaggle.com/zero92/best-lbgm-new-features
# and https://www.kaggle.com/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars

# In[ ]:


outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 
           'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',
                 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
dome_open     = ['Domed, Open', 'Domed, open']


# In[ ]:


df_train['StadiumType'] = df_train['StadiumType'].replace(outdoor,'outdoor')
df_train['StadiumType'] = df_train['StadiumType'].replace(indoor_closed,'indoor_closed')
df_train['StadiumType'] = df_train['StadiumType'].replace(indoor_open,'indoor_open')
df_train['StadiumType'] = df_train['StadiumType'].replace(dome_closed,'dome_closed')
df_train['StadiumType'] = df_train['StadiumType'].replace(dome_open,'dome_open')


# In[ ]:


rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']

overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',
            'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
            'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
            'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
            'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
            'Partly Cloudy', 'Cloudy']

clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']

snow  = ['Heavy lake effect snow', 'Snow']

none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']


# In[ ]:


df_train['GameWeather'] = df_train['GameWeather'].replace(rain,'rain')
df_train['GameWeather'] = df_train['GameWeather'].replace(overcast,'overcast')
df_train['GameWeather'] = df_train['GameWeather'].replace(clear,'clear')
df_train['GameWeather'] = df_train['GameWeather'].replace(snow,'snow')
df_train['GameWeather'] = df_train['GameWeather'].replace(none,'none')


# In[ ]:


nan = ['nan','E','SE','Calm','SSW']
def clean_wind_speed(windspeed):
    ws = str(windspeed)

    if 'mph' in ws.lower():
        return int(ws.lower().split('mph')[0])
    else :
        return ws
df_train['WindSpeed'] = df_train['WindSpeed'].apply(clean_wind_speed)
df_train['WindSpeed'] = df_train['WindSpeed'].replace(nan,np.nan)


# In[ ]:


north = ['N','From S','North']

south = ['S','From N','South','s']

west = ['W','From E','West']

east = ['E','From W','from W','EAST','East']



north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']
north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']
south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']
south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']
no_wind = ['clear','Calm']
nan = ['1','8','13']


# In[ ]:


df_train['WindDirection'] = df_train['WindDirection'].replace(north,'north')
df_train['WindDirection'] = df_train['WindDirection'].replace(south,'south')
df_train['WindDirection'] = df_train['WindDirection'].replace(west,'west')
df_train['WindDirection'] = df_train['WindDirection'].replace(east,'east')
df_train['WindDirection'] = df_train['WindDirection'].replace(north_east,'north_east')
df_train['WindDirection'] = df_train['WindDirection'].replace(north_west,'north_west')
df_train['WindDirection'] = df_train['WindDirection'].replace(south_east,'clear')
df_train['WindDirection'] = df_train['WindDirection'].replace(south_west,'south_west')
df_train['WindDirection'] = df_train['WindDirection'].replace(no_wind,'no_wind')
df_train['WindDirection'] = df_train['WindDirection'].replace(nan,np.nan)


# In[ ]:


natural_grass = ['natural grass','Naturall Grass','Natural Grass']
grass = ['Grass']

fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

artificial = ['Artificial','Artifical']


# In[ ]:


df_train['Turf'] = df_train['Turf'].replace(natural_grass,'natural_grass')
df_train['Turf'] = df_train['Turf'].replace(grass,'grass')
df_train['Turf'] = df_train['Turf'].replace(fieldturf,'fieldturf')
df_train['Turf'] = df_train['Turf'].replace(artificial,'artificial')


# In[ ]:


df_train.loc[df_train.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
df_train.loc[df_train.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"

df_train.loc[df_train.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr']= "BLT"
df_train.loc[df_train.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"

df_train.loc[df_train.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
df_train.loc[df_train.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"

df_train.loc[df_train.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
df_train.loc[df_train.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"


# In[ ]:


df_train['ToLeft'] = 0
df_train.loc[df_train.PlayDirection == "left",'ToLeft'] = 1

df_train['IsBallCarrier'] = df_train['NflId'] == df_train['NflIdRusher']

df_train['TeamOnOffense'] = 'away'
df_train.loc[df_train.PossessionTeam == df_train.HomeTeamAbbr,             'TeamOnOffense'] = 'home'


# In[ ]:


df_train['IsBallCarrier'] = df_train['NflId'] == df_train['NflIdRusher']
df_train['IsOnOffense'] = df_train['Team'] == df_train['TeamOnOffense']


# In[ ]:


df_train['YardsFromOwnGoal'] = 50 + (50-df_train['YardLine'])
df_train.loc[df_train.YardLine == 50,'YardsFromOwnGoal'] = 50
df_train.loc[df_train.FieldPosition == df_train.PossessionTeam,              'YardsFromOwnGoal'] = 50

df_train['X_std'] = df_train['X']
df_train['Y_std'] = df_train['Y']

df_train.loc[df_train.ToLeft == 1, 'X_std'] = 120 - df_train['X']
df_train.loc[df_train.ToLeft == 1, 'Y_std'] = 160/3-df_train['Y']


# In[ ]:


df_train['SecondsPassed'] = ((df_train['Quarter'].astype('int')-1)*15 +                              df_train['GameClock'].str.slice(stop=2)                                                   .astype('int'))*60 +                              df_train['GameClock'].str.slice(start=3, stop=5)                                                   .astype('int')


# # Play Plotter

# In[ ]:


# from https://teamcolorcodes.com/category/nfl-team-color-codes/
nfl_color_dict = {
    'ARZ': ['#97233F','#000000'],
    'ATL': ['#A71930','#000000'],
    'BLT': ['#241773','#000000'],
    'BUF': ['#00338D','#C60C30'],
    'CAR': ['#0085CA','#101820'],
    'CHI': ['#0B162A','#C83803'],
    'CIN': ['#FB4F14','#000000'],
    'CLV': ['#311D00','#FF3C00'],
    'DAL': ['#003594','#FFFFFF'],
    'DEN': ['#FB4F14','#002244'],
    'DET': ['#0076B6','#B0B7BC'],
    'GB' : ['#203731','#FFB612'],
    'HST': ['#03202F','#A71930'],
    'IND': ['#002C5F','#A2AAAD'],
    'JAX': ['#101820','#D7A22A'],
    'KC' : ['#E31837','#FFB81C'],
    'LAC': ['#002A5E','#FFC20E'],
    'LA':  ['#002244','#866D4B'],
    'MIA': ['#008E97','#FC4C02'],
    'MIN': ['#4F2683','#FFC62F'],
    'NE' : ['#002244','#C60C30'],
    'NO' : ['#D3BC8D','#101820'],
    'NYG': ['#0B2265','#A71930'],
    'NYJ': ['#125740','#000000'],
    'OAK': ['#000000','#A5ACAF'],
    'PHI': ['#004C54','#A5ACAF'],
    'PIT': ['#FFB612','#101820'],
    'SF' : ['#AA0000','#B3995D'],
    'SEA': ['#002244','#69BE28'],
    'TB' : ['#D50A0A','#FF7900'],
    'TEN': ['#0C2340','#418FDE'],
    'WAS': ['#773141','#FFB612']
}


# In[ ]:


df_train.loc['YardsToTD'] =  df_train['X_std']
df_train.loc[ df_train.ToLeft == 0, 'YardsToTD'] = 100 - df_train['X_std']
df_train.loc[ (df_train.ToLeft == 0) & (df_train.X_std > 100), 'YardsToTD'] = df_train['Yards']

df_train['team_abr'] = df_train['VisitorTeamAbbr']
df_train.loc[df_train.Team == 'home','team_abr'] = df_train['HomeTeamAbbr']


# In[ ]:


np.cos(10)


# In[ ]:


def plot_motion(row):
    x = row.X_std
    y = row.Y_std
    degrees = row.Dir
    toLeft = row.ToLeft
    speed = row.S
    
    if toLeft == 1:
        degrees = 360-degrees
    quad = int(np.floor(degrees / 90) + 1)
    degrees = int(degrees)


    # SOH CAH TOA
#     dx = los+runner.Yards.values[0]-run_x
    rads = degrees * np.pi/180
    # cos() = A/H
    if quad <= 2:
        if quad == 1:
            dx = speed*np.cos((np.pi/2)-rads)
            dy = speed*np.sin((np.pi/2)-rads)
        elif quad == 2:
            dx = speed*np.cos(np.pi-rads)
            dy = -speed*np.sin(np.pi-rads)
    elif quad == 3:
        dx = -speed*np.cos((3*np.pi/2)-rads)
        dy = -speed*np.sin((3*np.pi/2)-rads)
    else:
        dx = -speed*np.cos((2*np.pi)-rads)
        dy = speed*np.sin((2*np.pi)-rads)
        
    ax2.arrow(x=x,y=y, 
          dx=dx,dy=dy,
          head_width=0.2, head_length=0.2,zorder=3,ec='#FFFFF0')


# In[ ]:


# Plot 1 Play
import matplotlib.pyplot as plt

plays_ids = df_train['PlayId'].unique()
play_sample = plays_ids[np.random.randint(0,len(plays_ids))]

play_pd = df_train.loc[df_train.PlayId == play_sample]

runner = play_pd[play_pd.IsBallCarrier == True]
runner_loc = runner[['X_std','Y_std']].values[0]
player_c = [nfl_color_dict[i][0] for i in play_pd.team_abr.values]
player_o = [nfl_color_dict[i][1] for i in play_pd.team_abr.values]

los = round(np.mean(play_pd.loc[(play_pd.Position == 'T')|(play_pd.Position == 'G')|                                 (play_pd.Position == 'C')]['X_std']))

teams = play_pd[['HomeTeamAbbr','VisitorTeamAbbr']].iloc[1].values
try:
    left_team = play_pd.loc[(play_pd.ToLeft == 1), 'Team'].unique()[0]
except:
    left_team = play_pd['Team'].unique()[0]
if left_team == 'away':
    left_abr = teams[1]
    right_abr = teams[0]
else:
    left_abr = teams[0]
    right_abr = teams[1]
    


plt.figure(figsize=(22,14))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25,10), gridspec_kw={'width_ratios': [1, 5]})
plt.subplots_adjust(wspace=0.01, hspace=0)
ax2.set_facecolor('#6b9c58')

#Players
ax2.scatter(play_pd.X_std.values, play_pd.Y_std.values,             c=player_c, s=150, edgecolors=player_o, zorder=2)
ax2.scatter(runner_loc[0],runner_loc[1],c='white',marker='*', zorder=2)
ax2.set_xlim([-10,110])
ax2.set_ylim([0,53.3])

# Endzones
ax2.axvspan(-10, 0, alpha=0.9, color=nfl_color_dict[left_abr][0])
ax2.axvspan(100, 110, alpha=0.9, color=nfl_color_dict[right_abr][0])
ax2.axvline(0,c=nfl_color_dict[left_abr][1])
ax2.axvline(100,c=nfl_color_dict[right_abr][1])
ax2.text(-7, 53.3/2, left_abr, fontsize=15, color=nfl_color_dict[left_abr][1], fontweight='bold')
ax2.text(103, 53.3/2, right_abr, fontsize=15, color=nfl_color_dict[right_abr][1], fontweight='bold')

# Hash Marks + Midfield
ax2.axhline(53.3/2+18.5/2, linestyle='--',c='white',zorder=-2)
ax2.axhline(53.3/2-18.5/2, linestyle='--',c='white',zorder=-2)
ax2.axvline(50,c='white',zorder=-2)
ax2.scatter(50,53.3/2, c='white', s=15000, zorder=-2)
ax2.text(47.5,53.3/2-1.5, s='Big Data\n  Bowl\n  2019', fontsize=10, color='#6b9c58', 
         fontweight='bold', zorder=-1)
yards = 10
while yards <= 50:
    ax2.axvline(yards, linestyle='-',c='white',zorder=-1)
    ax2.axvline(100-yards, linestyle='-',c='white',zorder=-1)
    
    ax2.text(yards-2, 50, str(yards), fontsize=10, color='white', fontweight='bold')
    ax2.text(yards-2, 2, str(yards), fontsize=10, color='white', fontweight='bold')
    if yards == 50:
        break
    ax2.text(100-yards+1, 50, str(yards), fontsize=10, color='white', fontweight='bold')
    ax2.text(100-yards+1, 2, str(yards), fontsize=10, color='white', fontweight='bold')
    
    yards+=10

# Movement
play_pd.apply(plot_motion, axis=1)
    
ax2.arrow(x=runner.X_std.values[0],y=runner.Y_std.values[0], 
          dx=los+runner.Yards.values[0]-runner.X_std.values[0],dy=0,
          head_width=1.2, head_length=1.2,zorder=3, ec='#FFFFF0')

# Play YardLines
ax2.axvline(los,c='grey',zorder=-1)
first_down = los+play_pd.Distance.values[0]
# Set it to goal to go
if first_down >= 100:
    first_down = 100
ax2.axvline(first_down,c='yellow',zorder=3)

# Scoreboard
ax1.set_facecolor('#000000')
ax1.set_xlim([0,100])
ax1.set_ylim([0,100])
season = str(int(play_pd.Season.values[0]))
week   = str(int(play_pd.Week.values[0]))
home_team = play_pd.HomeTeamAbbr.values[0]
away_team = play_pd.VisitorTeamAbbr.values[0]
home_score = str(int(play_pd.HomeScoreBeforePlay.values[0]))
away_score = str(int(play_pd.VisitorScoreBeforePlay.values[0]))
quarter = str(int(play_pd.Quarter.values[0]))
game_clock = play_pd.GameClock.values[0][:-3]
down =  int(play_pd.Down.values[0])
if first_down >= 100:
    distance = 'Goal'
else:
    distance = str(int(play_pd.Distance.values[0]))
ax1.text(10,90,season+' Week '+ week, fontsize=20, color='white')
ax1.text(10,80,home_team +':', fontsize=40, color='white')
ax1.text(10,70,away_team +':', fontsize=40, color='white')
ax1.text(60,80,home_score, fontsize=40, color='white')
ax1.text(60,70,away_score, fontsize=40, color='white')
if play_pd.TeamOnOffense.values[0] == 'home':
    ax1.scatter(90,82.5,color='white', s=200)
else:
    ax1.scatter(90,72.5,color='white', s=200)
ax1.text(10,50, 'Q'+quarter, color='white', fontsize=30)
ax1.text(45,50, game_clock, color='white', fontsize=30)
if down == 1:
    down_str = 'st'
elif down == 2:
    down_str = 'nd'
elif down == 3:
    down_str = 'rd'
else:
    down_str = 'th'
ax1.text(10,40, str(down)+down_str+' and '+distance, color='white', fontsize=30)

# plt.gca().fill(x=15, y=10, x2=30, y2=20,c='black')  

# Turn off tick labels
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.grid(False)
ax2.grid(False)
plt.show()


# ## Ideas
# - Give player a certain amount of space depending on their location + velocity
# - Historical Features
#     - RB by Yards
#     - OL by Yards
#     - DVLA
#         - Function of DSkill + RB Skill
# - Worrying about the closest 2-3 defenders
#     - Blocked?
#         - There's a Blocker within N distance
#         - Look at speed/direction
#         - Are they moving slowly?
#     - Project where RB will be in 1 sec
#         - Are defenders moving in that direction?
#         - Is there a blocker in their way    
# - Count OFF men in box
# - Only look at "normal" instances
#     - Get rid of 4th quarter
#     - Get rid of redzone
#     - Get rid of 2min of 1st half
#     - Only look at reasonably close games
# 

# # Data Explore + Dummy Models

# In[ ]:


runners_pd = df_train[df_train.IsBallCarrier == True]

left_orientation = runners_pd[runners_pd.ToLeft == 1].Orientation.values
right_orientation = runners_pd[runners_pd.ToLeft == 0].Orientation.values

plt.hist(left_orientation, alpha=0.5, label='left facing')
plt.hist(right_orientation, alpha=0.5, label='right facing')
plt.legend()
plt.show()
# runners_pd.head()


# In[ ]:


runners_pd.shape[0] == len(df_train.PlayId.unique())


# In[ ]:


run_id = runners_pd.PlayId.unique()
play_id = df_train.PlayId.unique()
for i in run_id:
    if i not in play_id:
        print(i)


# In[ ]:



yards = runners_pd.Yards.values

plt.hist(yards, bins = 50)
plt.axvline(np.mean(yards),label='Mean:{}'.format(np.round(np.mean(yards))), color='orange')
plt.axvline(np.median(yards),label='Median:{}'.format(np.round(np.median(yards))), color='red')
plt.legend()
plt.show()


# In[ ]:


df_train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df_labels = df_train.copy()
for col in df_labels.columns:
    if df_labels[col].dtype == 'O':
        df_labels[col] = enc.fit_transform(df_labels[col].astype(str))


# In[ ]:


import seaborn as sns


sns.set(rc={'figure.figsize':(11.7,8.27)})

corr = df_labels.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


col_names = corr.columns
found_pairs = []
def parse_corr_matrix(corr_matrix):
    for row in col_names:
        for col in col_names:
            if row == col:
                continue
            if ([row, col] in found_pairs) or ([col, row] in found_pairs):
                continue
            corr_val = np.abs(corr_matrix[row][col])
            if corr_val > 0.6:
                print('{} by {}: {}'.format(row,col,np.round(corr_matrix[row][col],2)))
                found_pairs.append([row,col])

parse_corr_matrix(corr)


# In[ ]:


# YardsFromOwnGoal, X_std is important features
df_uncorr_train = runners_pd.copy()
df_uncorr_train['above_avg_yards'] = runners_pd.Yards > 3
labels = df_uncorr_train.above_avg_yards
df_uncorr_train = df_uncorr_train.drop(['GameId','PlayId','YardsToTD','Quarter',
                      'PlayDirection','Yards','above_avg_yards', 'YardsFromOwnGoal','X_std'],axis=1)
cols = df_uncorr_train.columns
cat_features = [i for i in range(len(cols)) if df_uncorr_train[cols[i]].dtype == 'O']
# Check for NaNs
print('NaNs present in...')
for col in cols:
    if df_uncorr_train[col].isnull().values.any():
        df_uncorr_train.loc[df_uncorr_train[col].isnull()] = 0
        print(col)


# In[ ]:


from collections import Counter
Counter(labels)


# In[ ]:


from catboost import CatBoostClassifier

clf = CatBoostClassifier(iterations=20)
clf.fit(df_uncorr_train.values, labels.astype('int').values, cat_features, verbose=0)


# In[ ]:


Counter(clf.predict(df_uncorr_train.values))


# In[ ]:


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(labels.astype('int').values, 
                                         clf.predict(df_uncorr_train.values), pos_label=1)

metrics.auc(fpr, tpr)


# In[ ]:


x_vals = df_uncorr_train.columns
y_vals = clf.feature_importances_

plt.bar(x_vals,y_vals)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X = runners_pd.SecondsPassed.values
X_ = X.reshape(len(X),1)
y = runners_pd.Yards.values
reg = LinearRegression().fit(X_, y)


r2 = reg.score(X_, y)
plt.scatter(runners_pd.SecondsPassed.values, runners_pd.Yards.values, alpha=0.3)
plt.title('Seconds Pass vs. Yards Gained on Runs\n Explanatory Power: {}%'.format(np.round(100*r2)) )
plt.ylabel('Yards Gained')
plt.xlabel('Seconds Passed')
plt.show()


# In[ ]:


runners_pd.columns


# In[ ]:


dib_mean = runners_pd.loc[runners_pd.DefendersInTheBox > 3]                      .groupby('DefendersInTheBox')['Yards'].mean()
dib_count = runners_pd.loc[runners_pd.DefendersInTheBox > 3]                     .groupby('DefendersInTheBox')['PlayId'].count()


# In[ ]:


legit = runners_pd.loc[runners_pd.DefendersInTheBox > 3]
X = legit.DefendersInTheBox.values
X_ = X.reshape(len(X),1)
y = legit.Yards.values
reg = LinearRegression().fit(X_, y)
r2 = reg.score(X_, y)

plt.title('Defenders in Box vs. Yards Gained on Runs\n Explanatory Power: {}%'.format(np.round(100*r2)) )
plt.bar(dib_mean.index,dib_mean.values,fill=None,edgecolor='orange')
plt.scatter(legit.DefendersInTheBox.values, legit.Yards.values)
plt.ylabel('Yards gained')
plt.xlabel('# of Defenders in the Box')
plt.show()


# In[ ]:


# Close vs. Blowouts per quarter

runners_pd.loc['score_diff'] = runners_pd.VisitorScoreBeforePlay - runners_pd.HomeScoreBeforePlay
runners_pd.loc[runners_pd.Team == 'home', 'score_diff'] = runners_pd.HomeScoreBeforePlay - runners_pd.VisitorScoreBeforePlay



X = runners_pd.score_diff.fillna(0).values
X_ = X.reshape(len(X),1)
y = runners_pd.Yards.fillna(0).values
reg = LinearRegression().fit(X_, y)
r2 = reg.score(X_, y)

# plt.title('Defenders in Box vs. Yards Gained on Runs\n Explanatory Power: {}%'.format(np.round(100*r2)) )
# plt.bar(dib_mean.index,dib_mean.values,fill=None,edgecolor='orange')
# plt.scatter(legit.DefendersInTheBox.values, legit.Yards.values)
# plt.ylabel('Yards gained')
# plt.xlabel('# of Defenders in the Box')
# plt.show()


# In[ ]:


score_diff = runners_pd.HomeScoreBeforePlay.values - runners_pd.VisitorScoreBeforePlay.values
runners_pd.score_diff = score_diff


# In[ ]:


plt.scatter(runners_pd.Dis,runners_pd.Yards, alpha=0.1)
plt.show()


# ## Checking for irregularity in game situation

# In[ ]:


# - Only look at "normal" instances
#     - Get rid of 4th quarter
#     - Get rid of redzone
#     - Get rid of 2min of 1st half
#     - Only look at reasonably close games


# In[ ]:


runners_pd = runners_pd.dropna(subset=['Yards'])


# In[ ]:


runners_pd.columns


# In[ ]:


from scipy.stats import ttest_ind

# Is Q4 so different than other times in game?
q4_yards = runners_pd.loc[runners_pd.Quarter == 4].Yards.values
not_q4_yards = runners_pd.loc[runners_pd.Quarter != 4].Yards.values

t, p = ttest_ind(q4_yards, not_q4_yards)
plt.boxplot([q4_yards,not_q4_yards], labels=['Q4 Plays\n{}\n{}'.format(round(np.mean(q4_yards),2),
                                                                       round(np.std(q4_yards),2)),
                                             '~Q4 Plays\n{}\n{}'.format(round(np.mean(not_q4_yards),2),
                                                                       round(np.std(not_q4_yards),2))])
plt.ylabel('Yards Gained')
plt.title('Is Q4 so different than other times in game? {}'.format(round(p,2)< 0.05))
plt.show()


# In[ ]:


# Is redzone so different than other times in game?
rz_yards = runners_pd.loc[runners_pd.YardsToTD <= 20].Yards.values
not_rz_yards = runners_pd.loc[runners_pd.YardsToTD > 20].Yards.values

t, p = ttest_ind(rz_yards, not_rz_yards)
plt.boxplot([rz_yards,not_rz_yards],  labels=['RZ Plays\n{}\n{}'.format(round(np.mean(rz_yards),2),
                                                                       round(np.std(rz_yards),2)),
                                             '~RZ Plays\n{}\n{}'.format(round(np.mean(not_rz_yards),2),
                                                                       round(np.std(not_rz_yards),2))])
plt.ylabel('Yards Gained')
plt.title('Is redzone so different than other times in game? {}'.format(round(p,2) < 0.05))
plt.show()


# In[ ]:


# Is 2min of Q2 different?
twomin_yards = runners_pd[(runners_pd.SecondsPassed >= (15+13)*60) & (runners_pd.SecondsPassed < (15+15)*60)].Yards.values
not_twomin_yards = runners_pd[(runners_pd.SecondsPassed < (15+13)*60) | (runners_pd.SecondsPassed >= (15+15)*60)].Yards.values

t, p = ttest_ind(rz_yards, not_rz_yards)
plt.boxplot([twomin_yards,not_twomin_yards], labels=['2min Plays\n{}\n{}'.format(round(np.mean(twomin_yards),2),
                                                                       round(np.std(twomin_yards),2)),
                                             '~2min Plays\n{}\n{}'.format(round(np.mean(not_twomin_yards),2),
                                                                       round(np.std(not_twomin_yards),2))])
plt.ylabel('Yards Gained')
plt.title('Is 2min OFF so different than other times in game? {}'.format(round(p,2)< 0.05))
plt.show()


# In[ ]:


# Are bowouts different
blowout_yards = runners_pd[abs(runners_pd.HomeScoreBeforePlay - runners_pd.VisitorScoreBeforePlay) >= 20].Yards.values
win_blowout_yards = runners_pd[runners_pd.score_diff >= 20].Yards.values
lose_blowout_yards = runners_pd[runners_pd.score_diff < -20].Yards.values
not_blowout_yards = runners_pd[abs(runners_pd.HomeScoreBeforePlay - runners_pd.VisitorScoreBeforePlay) < 20].Yards.values
t, p = ttest_ind(blowout_yards, not_blowout_yards)
plt.boxplot([blowout_yards,not_blowout_yards], labels=['All Blowout Plays\n{}\n{}'.format(round(np.mean(blowout_yards),2),
                                                                       round(np.std(blowout_yards),2)),
                                             '~Not Blowout Plays\n{}\n{}'.format(round(np.mean(not_blowout_yards),2),
                                                                       round(np.std(not_blowout_yards),2))])
plt.ylabel('Yards Gained')
plt.title('Blowout Runs so different than other times in game? {}'.format(round(p,2)< 0.05))
plt.show()


# In[ ]:


t, p = ttest_ind(win_blowout_yards, not_blowout_yards)
t, p1 = ttest_ind(lose_blowout_yards, not_blowout_yards)
plt.boxplot([win_blowout_yards,lose_blowout_yards, not_blowout_yards], labels=['Winning Blowout Plays','Losing Blowout Plays','~Blowout Plays'])
plt.ylabel('Yards Gained')
plt.title('Blowout Runs so different than other times in game? Winning:{} Losing: {}'.format(round(p,2)< 0.05,round(p1,2)< 0.05))
plt.show()


# ## FCNN Functions

# In[ ]:


# evaluation metric
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 


# In[ ]:


def generate_cpd(yards):
    yards = int(yards)
    return(np.append(np.zeros(99+yards),np.zeros(199-(99+yards))+1).reshape(199,1))

def generate_cpd2(yards):
    return(np.append(np.zeros(99+(int(np.floor(yards)))),np.array(yards-np.floor(yards)), np.zeros(199-(99+int(yards)-1))+1).reshape(199,1))


# In[ ]:


generate_cpd(4).shape


# In[ ]:


runners_answers = [generate_cpd(i) for i in runners_pd.Yards.values]
dummy_answers = [generate_cpd(4) for i in runners_pd.Yards.values]


# In[ ]:


dummy_results = [crps(runners_answers[i],dummy_answers[i]) for i in range(runners_pd.shape[0])]
print(
"Dummy Sum: {}\nDummy MAE: {}".format(sum(dummy_results),np.mean(dummy_results)*1000000)
)


# In[ ]:


labels.astype('int').values


# In[ ]:


from catboost import Pool, CatBoostRegressor


# Initialize CatBoostRegressor
# clf = CatBoostRegressor(iterations=500)
# clf.fit(df_uncorr_train.values, runners_pd.Yards.values, cat_features, verbose=0)


# In[ ]:


# preds = clf.predict(df_uncorr_train.values)
# preds


# In[ ]:





# # Spacial Net

# ## Spacial Netv1
# Making feature space the formation of the play, normalized around...
#  - Running back?
#  - Line of Scrimmage?
#  - Nothing, because they are already normalized around 0,0 of the actual field?

# In[ ]:


from collections import namedtuple, defaultdict
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler, SequentialSampler


# In[ ]:


# Create the dataset
df_train = df_train.loc[pd.notnull(df_train.Position)]


# In[ ]:


# create positional dataset
# Which positions are in every single play?
# So there is not set positional information to keep constant... but positon doesn't matter!
# What does matter is proximity to ball carrier, what team they are on, their speed, direction, height, weight
cnt = Counter(df_train.Position.values)
play_cnt = len(df_train.PlayId.unique())
print('Total Number of Plays:',play_cnt)
for k,v in cnt.items():
    print(k,v, '({}%)'.format(round(100*v/play_cnt,2)))


# In[ ]:


play_pd.head()


# In[ ]:


def motion_prepro(row):
    x = row.X_std
    y = row.Y_std
    degrees = row.Dir
    toLeft = row.ToLeft
    speed = row.S
    
    # This was checked-- it's bc speed == 0, so no angle of movement Dir
    if speed == 0:
        return(0,0)
    
    if toLeft == 1:
        degrees = 360-degrees
        
    quad = int(np.floor(degrees / 90) + 1)
    degrees = int(degrees)
   


    # SOH CAH TOA
    rads = degrees * np.pi/180
    # cos() = A/H
    if quad <= 2:
        if quad == 1:
            dx = speed*np.cos((np.pi/2)-rads)
            dy = speed*np.sin((np.pi/2)-rads)
        elif quad == 2:
            dx = speed*np.cos(np.pi-rads)
            dy = -speed*np.sin(np.pi-rads)
    elif quad == 3:
        dx = -speed*np.cos((3*np.pi/2)-rads)
        dy = -speed*np.sin((3*np.pi/2)-rads)
    else:
        dx = -speed*np.cos((2*np.pi)-rads)
        dy = speed*np.sin((2*np.pi)-rads)
        
    return(dx, dy)


# In[ ]:


dis_list = df_train.apply(motion_prepro, axis=1)

dxs = []
dys = []
for dx,dy in dis_list.items():
    dxs.append(dx); dys.append(dy)
df_train['dx'] = [i[0] for i in dis_list]
df_train['dy'] = [i[1] for i in dis_list]


# In[ ]:


df_train['dx'] = [i[0] for i in dis_list]
df_train['dy'] = [i[1] for i in dis_list]


# In[ ]:


position_dict = {
    'QB':0,
    'HB':1,
    'RB':1,
    'FB': 2,
    'WR': 3,
    'TE': 4,
    'T': 5,
    'OT':5,
    'G': 6,
    'OG':6,
    'C': 7,
    'DT':8,
    'DE':9,
    'DL':10,
    'ILB':11,
    'MLB':12,
    'OLB':13,
    'LB':14,
    'CB':15,
    'DB':16,
    'S':17,
    'FS':18,
    'SS':19,
    'SAF':20,
    'NT':21
}
Counter(df_train.loc[df_train.IsBallCarrier == 1].Position.values)
df_train['adam_encode_pos'] = [position_dict[i] for i in df_train.Position.values]


# In[ ]:


import re
rez = [re.split(r'-',i) for i in df_train.PlayerHeight]
df_train['Height'] = [int(i[0])*12+int(i[1]) for i in rez]


# In[ ]:


df_train.shape


# In[ ]:


def preprocess(df_train, train=True):
    
    print('Starting at {} features...'.format(df_train.shape[1]))
    df_train['ToLeft'] = 0
    df_train.loc[df_train.PlayDirection == "left",'ToLeft'] = 1

    df_train['IsBallCarrier'] = df_train['NflId'] == df_train['NflIdRusher']
    
    df_train['YardsFromOwnGoal'] = 50 + (50-df_train['YardLine'])
    df_train.loc[df_train.YardLine == 50,'YardsFromOwnGoal'] = 50
    df_train.loc[df_train.FieldPosition == df_train.PossessionTeam,              'YardsFromOwnGoal'] = 50

    df_train['X_std'] = df_train['X']
    df_train['Y_std'] = df_train['Y']

    df_train.loc[df_train.ToLeft == 1, 'X_std'] = 120 - df_train['X']
    df_train.loc[df_train.ToLeft == 1, 'Y_std'] = 160/3-df_train['Y']
    
    df_train['encode_pos'] = [position_dict[i] for i in df_train.Position.values]
    
    dis_list = df_train.apply(motion_prepro, axis=1)

    dxs = []
    dys = []
    for dx,dy in dis_list.items():
        dxs.append(dx); dys.append(dy)
    df_train['dx'] = [i[0] for i in dis_list]
    df_train['dy'] = [i[1] for i in dis_list]
    
    rez = [re.split(r'-',i) for i in df_train.PlayerHeight]
    df_train['Height'] = [int(i[0])*12+int(i[1]) for i in rez]

    plays = df_train.PlayId.unique()
    spacial_list = []
    play_list = []
    
    print('Now at {} features...'.format(df_train.shape[1]))
    for play in plays:
        play_pd = df_train.loc[df_train.PlayId == play]
        runner = play_pd.loc[play_pd.IsBallCarrier == 1]
        run_x = runner.X_std.values[0]
        run_y = runner.Y_std.values[0]
        run_dx = runner.dx.values[0]
        run_dy = runner.dy.values[0]
        run_height = runner.Height.values[0]
        run_weight = runner.PlayerWeight.values[0]
        run_pos = runner.encode_pos.values[0]

        run_np = np.array([run_x,run_y,run_dx,run_dy,run_height,run_weight,run_pos])

        players = play_pd.loc[play_pd.IsBallCarrier != 1]
        unrank_pd = pd.DataFrame()

        play_x = players.X_std.values - run_x
        play_y = players.X_std.values - run_y
        unrank_pd['play_x'] = play_x
        unrank_pd['play_y'] = play_y
        unrank_pd['play_dis'] = np.sqrt((run_x-play_x)**2+(run_y-play_y)**2) #euclidean
        unrank_pd['play_dx'] = players.dx.values
        unrank_pd['play_dy'] = players.dy.values
        unrank_pd['play_height'] = players.Height.values
        unrank_pd['play_weight'] = players.PlayerWeight.values
        unrank_pd['play_pos'] = players.encode_pos.values

        rank_pd = unrank_pd.sort_values(by='play_dis', axis=0, ascending=True, 
                              inplace=False, kind='quicksort', na_position='last')

        spacial_list.append(np.append(run_np, rank_pd.values.flatten()))
        play_list.append(play)
      
    torch_data = torch.from_numpy(np.array(spacial_list)).float()
    
    print('Returning {} features...'.format(torch_data.shape[1]))
    if train:
        y = [df_train.loc[df_train.PlayId == play].Yards.values[0] for play in plays]
        y = np.array([generate_cpd(i) for i in y])
        y = torch.from_numpy(y)
        return(torch_data, y, play_list)

    return(torch_data, play_list)


# In[ ]:


# x = df_train[df_train.GameId == 2017090700.0].copy()
# a, b = preprocess(df_train, train=False)


# In[ ]:


def make_pred(test_data, prediction_df, env, model):
    data, plays = preprocess(test_data,train=False)
    output = model(data)
    
    col_list = ['Yards{}'.format(i) for i in range(-99,100)]
    answer_pd = pd.DataFrame(output.numpy(), columns=col_list)
    answer_pd['PlayId'] = plays
    env.predict(answer_pd)
    env.write_submission_file()


# In[ ]:


import time

# Change everyone's x/y to be in relation to runner
    # Collect their team, dx/dy, height, weight
# Rank each play by distance to runner
# Populate column values based on proximity

plays = df_train.PlayId.unique()
cnt = 10
spacial_list = []
start = time.time()
for play in plays:
    play_pd = df_train.loc[df_train.PlayId == play]
    runner = play_pd.loc[play_pd.IsBallCarrier == 1]
    run_x = runner.X_std.values[0]
    run_y = runner.Y_std.values[0]
    run_dx = runner.dx.values[0]
    run_dy = runner.dy.values[0]
    run_height = runner.Height.values[0]
    run_weight = runner.PlayerWeight.values[0]
    run_pos = runner.adam_encode_pos.values[0]

    run_np = np.array([run_x,run_y,run_dx,run_dy,run_height,run_weight,run_pos])

    players = play_pd.loc[play_pd.IsBallCarrier != 1]
    unrank_pd = pd.DataFrame()

    play_x = players.X_std.values - run_x
    play_y = players.X_std.values - run_y
    unrank_pd['play_x'] = play_x
    unrank_pd['play_y'] = play_y
    unrank_pd['play_dis'] = np.sqrt((run_x-play_x)**2+(run_y-play_y)**2) #euclidean
    unrank_pd['play_dx'] = players.dx.values
    unrank_pd['play_dy'] = players.dy.values
    unrank_pd['play_height'] = players.Height.values
    unrank_pd['play_weight'] = players.PlayerWeight.values
    unrank_pd['play_pos'] = players.adam_encode_pos.values

    rank_pd = unrank_pd.sort_values(by='play_dis', axis=0, ascending=True, 
                          inplace=False, kind='quicksort', na_position='last')

    spacial_list.append(np.append(run_np, rank_pd.values.flatten()))
print(round((time.time() - start)/60),'minutes to run')


# In[ ]:


y = [df_train.loc[df_train.PlayId == play].Yards.values[0] for play in plays]
y = np.array([generate_cpd(i) for i in y])
y = torch.from_numpy(y)


# In[ ]:


np.array(spacial_list).shape


# In[ ]:


np_spacial = np.array(spacial_list)


# In[ ]:


torch_train = torch.from_numpy(np_spacial[101:,:]).float()
y_train = y[101:].view(torch_train.shape[0],199).float()
torch_test = torch.from_numpy(np_spacial[:101,:]).float()
y_test = y[:101].view(101,199).float()


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(175, 150)
        self.fc2 = nn.Linear(150, 125)
        self.fc3 = nn.Linear(125, 100)
        self.fc4 = nn.Linear(100, 150)
        self.fc5 = nn.Linear(150, 199)
        
        self.bn1 = nn.BatchNorm1d(150)
        self.bn2 = nn.BatchNorm1d(125)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(150)

    def forward(self, x):
        x = torch.sigmoid(self.bn1(self.fc1(x)))
        x = torch.sigmoid(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x
      
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.05)
        
model = Net()
model.apply(init_weights)


# In[ ]:


# evaluation metric
def crps(y_true, y_pred):
    y_true = torch.clamp(torch.cumsum(y_true, dim=1), 0, 1)
    y_pred = torch.clamp(torch.cumsum(y_pred, dim=1), 0, 1)
    y_loss = ((y_true - y_pred) ** 2).sum(dim=1).sum(dim=0) / (199 * y_true.shape[0]) 
    return(y_loss)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50): # 3 full passes over the data
    model.zero_grad()
    output = model(torch_train)
    loss = crps(y_train,output.float())
    loss.backward()
    optimizer.step()
    


# In[ ]:



output = model(torch_test)
loss = crps(y_test,output.float())
print('Testing Loss:',loss.item())


# In[ ]:


len(plays[:101])


# In[ ]:


col_list = ['Yards{}'.format(i) for i in range(-99,100)]
answer_pd = pd.DataFrame(y_test.numpy(), columns=col_list)
answer_pd['PlayId'] = plays[:101]
answer_pd


# In[ ]:


# from kaggle.competitions import nflrush
# env = nflrush.make_env()
# for (play, prediction_df) in env.iter_test():
#     make_pred(play, prediction_df, env, spatial_model)
# # env = nflrush.make_env()
# # env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample.columns))
# # env.write_submission_file()


# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[ ]:


# dummy = df_train.loc[df_train.PlayId == 20170907000118.0].copy()
# data, plays = preprocess(dummy,train=False)
# output = model(data)


# In[ ]:


for test, sample in tqdm.tqdm(env.iter_test()):
    data, plays = preprocess(test,train=False)
    model.eval()
    output = model(data)
#     col_list = ['Yards{}'.format(i) for i in range(-99,100)]
    answer_pd = pd.DataFrame(output.detach().numpy(), columns=sample.columns)
    env.predict(answer_pd)
env.write_submission_file()


# In[ ]:


# output = model(data)

# col_list = ['Yards{}'.format(i) for i in range(-99,100)]
# answer_pd = pd.DataFrame(output.numpy(), columns=col_list)
# answer_pd['PlayId'] = plays
# env.predict(answer_pd)
# env.write_submission_file()

