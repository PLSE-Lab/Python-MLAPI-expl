#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from scipy.stats import skew
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/shot_logs.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.isnull().sum()


# **Data Exploration**

# In[ ]:


# Shots type made percentage


# In[ ]:


sns.countplot(data.PTS_TYPE)


# In[ ]:


sns.countplot(x="PTS_TYPE", hue ="SHOT_RESULT", data=data)


# In[ ]:


shot_made = data[data.SHOT_RESULT == 'made'].groupby('PTS_TYPE')['SHOT_RESULT'].count()


# In[ ]:


total_shots = data.groupby('PTS_TYPE')['SHOT_RESULT'].count()


# In[ ]:


perc_shot = round(shot_made / total_shots,4) *100


# In[ ]:


perc_shot


# In[ ]:


# Expected value
print(round(perc_shot.iloc[0]/100 * 2,4))
print(round(perc_shot.iloc[1]/100 * 3,4))


# Still better to shoot threes, but it depends on the distance

# In[ ]:


# Shots repartition


# In[ ]:


sns.set(rc={'figure.figsize':(12,10)})
sns.distplot(data.SHOT_DIST)


# In[ ]:


# Adding shots distance


# In[ ]:


# In the Paint : 8-
# Mid Range 8-16
# Long 2 : 16-24
# 3 : 24+


# In[ ]:


sns.distplot(data.SHOT_DIST)
plt.axvline(8, 0,1)
plt.axvline(16, 0,1)
plt.axvline(24, 0,1)
plt.annotate('In the Paint', xy=(-1,0.08))
plt.annotate('Mid Range', xy=(8.5,0.08))
plt.annotate('Long 2', xy=(18,0.08))
plt.annotate('3 points', xy=(25,0.08))


# In[ ]:


# Clearly, the mid range shots are the less taken


# In[ ]:


# Sucess per distance
distance_made = data[data.SHOT_RESULT == 'made'].groupby('SHOT_DIST')['SHOT_RESULT'].count()
distance_total = data.groupby('SHOT_DIST')['SHOT_RESULT'].count()


# In[ ]:


distance_acc = distance_made / distance_total


# In[ ]:


distance_acc = distance_acc.dropna()


# In[ ]:


distance_acc = distance_acc.reset_index()
distance_acc.columns = ['SHOT_DIST', 'SHOT_PERC']


# In[ ]:


distance_acc.head()


# In[ ]:


# We limit the shots to 30 ft, 9 m
distance_acc = distance_acc[distance_acc.SHOT_DIST <= 30]


# In[ ]:


sns.lineplot(x='SHOT_DIST', y="SHOT_PERC", data=distance_acc)
plt.axvline(8, 0,1)
plt.axvline(16, 0,1)
plt.axvline(24, 0,1)
plt.annotate('In the Paint', xy=(1,0.08))
plt.annotate('Mid Range', xy=(10,0.08))
plt.annotate('Long 2', xy=(18,0.08))
plt.annotate('3 points', xy=(25,0.08))
plt.title('Shot percentage depending on distance')


# Best chance to score when you're close to the paint, logic

# In[ ]:


# Let's check the proportions


# In[ ]:


for i, row in data.iterrows():
    if row['SHOT_DIST'] <= 8:
        data.loc[i,'SHOT_TYPE'] = 'Paint'
    elif ((row['SHOT_DIST'] > 8) & (row['SHOT_DIST'] <= 16)):
        data.loc[i,'SHOT_TYPE'] = 'Mid_Range'
    elif ((row['SHOT_DIST'] > 16) & (row['SHOT_DIST'] < 24)):
        data.loc[i,'SHOT_TYPE'] = 'Long_2'
    else:
        data.loc[i,'SHOT_TYPE'] = '3_Points'


# In[ ]:


shot_type_freq = round(data.groupby('SHOT_TYPE')['SHOT_TYPE'].count() / data.groupby('SHOT_TYPE')['SHOT_TYPE'].count().sum(),4)*100


# In[ ]:


shot_type_freq


# In[ ]:


# And now the success rate
shot_type_made = data[data.SHOT_RESULT == 'made'].groupby('SHOT_TYPE')['SHOT_RESULT'].count()
total_shots_type = data.groupby('SHOT_TYPE')['SHOT_RESULT'].count()


# In[ ]:


shot_type_eff = round(shot_type_made / total_shots_type,4) *100


# In[ ]:


shot_type_eff


# In[ ]:


# Conclusion for Shot Expected Value


# In[ ]:


print("Paint : ", round(shot_type_eff['Paint']/100 * 2,3))
print("Mid Range : ", round(shot_type_eff['Mid_Range']/100 * 2,3))
print("Long 2 : ", round(shot_type_eff['Long_2']/100 * 2,3))
print("3 Points : ", round(shot_type_eff['3_Points']/100 * 3,3))


# In conclusion, it is better to take shots in the Paint or at 3 points
# Long twos are the worst shots to make

# In[ ]:


# Shots per period


# In[ ]:


data[['PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK']]


# In[ ]:


data['PERIOD_SECOND'] = [(i-1) * 12 * 60 if i > 1 else 0 for i in data.PERIOD]


# In[ ]:


# Convert to datetime
data["GAME_CLOCK_SECONDS"] = pd.to_datetime(data["GAME_CLOCK"], format="%M:%S")


# In[ ]:


# Convert clock to seconds : for seconds played, 12:00 - x, x as game clock and 12:00 as quarter time
data["GAME_CLOCK_SECONDS"] = data["GAME_CLOCK_SECONDS"].apply(lambda x: 12*60 - (x.minute * 60 + x.second))


# In[ ]:


# Convert clock to seconds
data['GAME_CLOCK_SECONDS'] = data['GAME_CLOCK_SECONDS'] + data['PERIOD_SECOND']


# In[ ]:


data[['PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'GAME_CLOCK_SECONDS']]


# In[ ]:


# Stop at the end of the fourth quarter
# 4*12*60 = 2880
data_time = data[data.GAME_CLOCK_SECONDS <= 2880]


# In[ ]:


# 1 possession equals 24s
data['GAME_CLOCK_BINS'] = pd.cut(data['GAME_CLOCK_SECONDS'], bins=np.arange(0,60*48+1,24))


# In[ ]:


# Sucess per distance
shot_time = data[data.SHOT_RESULT == 'made'].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()
time_total = data.groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()


# In[ ]:


time_acc = shot_time / time_total
time_acc = time_acc.dropna()


# In[ ]:


time_acc = time_acc.reset_index()
time_acc.columns = ['SHOT_TIME', 'SHOT_PERC']


# In[ ]:


time_acc


# In[ ]:


x = np.arange(0,60*48,24)


# In[ ]:


sns.lineplot(x=x, y="SHOT_PERC", data=time_acc)
plt.axvline(12*60-24, 0,1, linestyle='dashed', color='red') # -24 because of the arange gap
plt.axvline(2*12*60-24, 0,1, linestyle='dashed' ,color='red')
plt.axvline(3*12*60-24, 0,1, linestyle='dashed' ,color='red')
plt.axvline(4*12*60-24, 0,1, linestyle='dashed' ,color='red')
plt.title('Shot percentage during the match')


# Accuracy decrease at the end of the quarters
# What about 2 points vs 3 points in the match

# In[ ]:


# Sucess per distance
shot_time_3 = data[(data.SHOT_RESULT == 'made') & (data.SHOT_TYPE == '3_Points')].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()
time_total_3 = data[data.SHOT_TYPE == '3_Points'].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()

shot_time_2 = data[(data.SHOT_RESULT == 'made') & (data.SHOT_TYPE != '3_Points')].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()
time_total_2 = data[data.SHOT_TYPE != '3_Points'].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()


# In[ ]:


time_acc_3 = shot_time_3 / time_total_3
time_acc_3 = time_acc_3.dropna()

time_acc_2 = shot_time_2 / time_total_2
time_acc_2 = time_acc_2.dropna()


# In[ ]:


time_acc_3 = time_acc_3.reset_index()
time_acc_3.columns = ['SHOT_TIME', 'SHOT_PERC']

time_acc_2 = time_acc_2.reset_index()
time_acc_2.columns = ['SHOT_TIME', 'SHOT_PERC']


# In[ ]:


f, (ax1, ax2) = plt.subplots(2)
sns.lineplot(x=x, y="SHOT_PERC", data=time_acc_2, ax=ax1)
ax1.axvline(12*60-24, 0,1, linestyle='dashed', color='red') # -24 because of the arange gap
ax1.axvline(2*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.axvline(3*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.axvline(4*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.set_title('2 points percentage during the game')

sns.lineplot(x=x, y="SHOT_PERC", data=time_acc_3, ax=ax2)
ax2.axvline(12*60-24, 0,1, linestyle='dashed', color='red') # -24 because of the arange gap
ax2.axvline(2*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.axvline(3*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.axvline(4*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.set_title('3 points percentage during the game')


# There is a greater percentage decrease at 3 points than 2 points, it could be explain by the long shots attempts when the clock is almost finished, let's look at the shots repartition

# In[ ]:


# Sucess per distance
shot_attempt_3 = data[data.SHOT_TYPE == '3_Points'].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()
shot_attempt_2 = data[data.SHOT_TYPE != '3_Points'].groupby('GAME_CLOCK_BINS')['SHOT_RESULT'].count()


# In[ ]:


attempt_acc_3 = shot_attempt_3 / time_total
attempt_acc_3 = attempt_acc_3.dropna()

attempt_acc_2 = shot_attempt_2 / time_total
attempt_acc_2 = attempt_acc_2.dropna()


# In[ ]:


attempt_acc_3 = attempt_acc_3.reset_index()
attempt_acc_3.columns = ['SHOT_TIME', 'SHOT_PROP']

attempt_acc_2 = attempt_acc_2.reset_index()
attempt_acc_2.columns = ['SHOT_TIME', 'SHOT_PROP']


# In[ ]:


f, (ax1, ax2) = plt.subplots(2)
sns.lineplot(x=x, y="SHOT_PROP", data=attempt_acc_2, ax=ax1)
ax1.axvline(12*60-24, 0,1, linestyle='dashed', color='red') # -24 because of the arange gap
ax1.axvline(2*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.axvline(3*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.axvline(4*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax1.set_title('2 points attempts proportions during the game')

sns.lineplot(x=x, y="SHOT_PROP", data=attempt_acc_3, ax=ax2)
ax2.axvline(12*60-24, 0,1, linestyle='dashed', color='red') # -24 because of the arange gap
ax2.axvline(2*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.axvline(3*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.axvline(4*12*60-24, 0,1, linestyle='dashed' ,color='red')
ax2.set_title('3 points attempts proportions during the game')


# This shows that players attempt more 3 points at the end of the quarters (and less 2 points ), and this also increases during the game

# In[ ]:


# Is it because the 3 points are more valuable at the end of the game ?
# For 2 points, we have 0.46 * 2  = 0.92 expected points
# For 3 points, we have 0.21 * 3 = 0.63 expected points
# We have to look at special cases : Last 2 minutes, point difference = 5 pts or less.


# In[ ]:


# Clutch Time


# In[ ]:


final_second = 12*4*60


# In[ ]:


data.head()


# In[ ]:


clutch_time = data_time[(np.abs(data_time.FINAL_MARGIN) < 5) & (data_time.GAME_CLOCK_SECONDS >= (final_second - 120))]
# Not very precise beacause we don't have the point difference at the shot moment


# In[ ]:


clutch_time.head()


# In[ ]:


len(clutch_time)


# In[ ]:


sns.set(rc={'figure.figsize':(8,6)})
sns.countplot(x="PTS_TYPE", hue ="SHOT_RESULT", data=clutch_time)


# In[ ]:


shot_made_clutch = clutch_time[clutch_time.SHOT_RESULT == 'made'].groupby('PTS_TYPE')['SHOT_RESULT'].count()
total_shots_clutch = clutch_time.groupby('PTS_TYPE')['SHOT_RESULT'].count()
perc_shot_clutch = round(shot_made_clutch / total_shots_clutch,4) *100


# In[ ]:


perc_shot_clutch


# In[ ]:


# Expected points :
# 2 points : 0.468 * 2 = 0.936
# 3 points : 0.315 * 3 = 0.945
# Interesting... It is better to shoot a three in clutch time / But not in the last minute though


# In[ ]:


# Last observation : is home court a real advantage ? 


# In[ ]:


data[['GAME_ID', 'LOCATION', 'W']].groupby(['LOCATION', 'W']).count()


# In[ ]:


data_home = data[data.LOCATION == "H"]
data_away = data[data.LOCATION == "A"]


# In[ ]:


data_home[['LOCATION', 'W']].groupby(['W']).count() / len(data_home)


# In[ ]:


data_away[['LOCATION', 'W']].groupby(['W']).count() / len(data_away)


# Home teams have 55% chances of winning a game, home court is an advantage

# Do they shoot better ?

# In[ ]:


shot_made_home = data_home[data_home.SHOT_RESULT == 'made'].groupby('PTS_TYPE')['SHOT_RESULT'].count()
total_shots_home = data_home.groupby('PTS_TYPE')['SHOT_RESULT'].count()
perc_shot_home = round(shot_made_home / total_shots_home,4) *100
perc_shot_home


# In[ ]:


shot_made_away = data_away[data_away.SHOT_RESULT == 'made'].groupby('PTS_TYPE')['SHOT_RESULT'].count()
total_shots_away = data_away.groupby('PTS_TYPE')['SHOT_RESULT'].count()
perc_shot_away = round(shot_made_away / total_shots_away,4) *100
perc_shot_away


# In[ ]:


# Home expected points:
# 2 points : 0.492 * 2 = 0.984
# 3 points : 0.357 * 3 = 1.071

# Away expected points:
# 2 points : 0.485 * 2 = 0.97
# 3 points : 0.346 * 3 = 1.038

# This is not a big difference but there is still an home advantage

