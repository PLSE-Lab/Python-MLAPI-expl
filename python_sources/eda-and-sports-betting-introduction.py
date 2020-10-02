#!/usr/bin/env python
# coding: utf-8

# # **NBA Betting Primer**
# 
# There are three major bets made on NBA games (and most sporting events for that matter)
# 
# 1. Spread
# 2. Over/Under
# 3. Money Line
# 
# Money Line is the easiest to understand is based out of 100. When a money line is negative for a given tea, that team is favored to win. If the Orlando Magic play the Atlanta Hawks and the Magic have a money line of -200, that means if you bet on the Magic, you need to bet 200 dollars to make 100 dollars. Inherentely one can also back out that this means the market believes that the Magic have a 66% chance of beating the Hawks. (200/300=.66)
# 
# Spread was created by bookies to get more action on both sides of the market and create a more balanced book. It is a metric of how much better the market believes one team is over another. In our same example lets say that the spread is -4 for the Magic in their game versus the Hawks. This means that the Magic are expected to win by more than four points and if you bet on the Magic they must win by more than 4 points for your bet to cash in. If you bet on the Magic at -4 and they win by exactly 4 points, money is returned to bettors.
# 
# O/U also called Over/Under is the sum of the total points combined in the game. So for this particular game say the O/U is 230, which means the Magic and the Hawks are expected to score 230 points. If you bet the over, you win if the sum of both teams is more than 230 and lose if the sum is less. A tie again returns money back to bettors. 
# 
# This notebook is intended to serve as an introduciton to sports betting in the NBA. I will explore some aspects of the betting data as well as simulate a betting strategy. Further notebooks may be added for additional betting strategies.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


files = [filename for filename in os.listdir('/kaggle/input/nba-odds-and-scores') if os.path.isdir(os.path.join('/kaggle/input/nba-odds-and-scores',filename))]
odds = []
scores = []
for file in files:
    temp_odds = pd.read_csv('../input/nba-odds-and-scores/'+file+'/vegas.txt',index_col=0,parse_dates=True)
    temp_scores = pd.read_csv('../input/nba-odds-and-scores/'+file+'/raw_scores.txt',index_col=0,parse_dates=True)
    odds.append(temp_odds)
    scores.append(temp_scores)
master_odds = pd.concat(odds,axis=0)
master_scores = pd.concat(scores,axis=0)


# ## Spreads
# Lets explore some of this betting data. I have included almost 10 years of betting data and scores and we shall chain them up in a large frame and look at some statistics. We will use the average lines from the 5 betting books. Below we see the distribution of spreads from 2012-2019 from the perspective of the away team. Values above 0 indicate the home team is favored while values below 0 indicate the away team is favored. Taking the average of these values, we can see that the value playing at home is roughly 3 points.

# In[ ]:


spreads = master_odds[master_odds['Location']=='away']['Average_Line_Spread']
spreads.hist()
plt.title('Histogram of Away Team Spreads from 2012-2019')
plt.xlabel('Spread')
plt.show()
avg_spread = spreads.mean()
print('Average Spread is '+str(np.round(avg_spread,2))+' and this is how much home court is worth.')


# ## O/U
# Next we look at the average O/U. It is worth northing that there has been a three point revolution in recent years and that the average O/U has been increasing year by year as offensive basketball has become more efficient, through a more analytical modernization of offenses. This trend is very clear when looking at boxplots of the year by year O/U, there is a clear uptrend from around a 195 O/U average in the 2012-2013 season to around 220 in 2018-19.

# In[ ]:


ou = master_odds[master_odds['Location']=='away']['Average_Line_OU']
ou.hist()
plt.title('Histogram of Road O/U from 2012-2019')
plt.xlabel('Spread')
plt.show()
avg_ou = ou.mean()
print('Average O/U is '+str(np.round(avg_ou,2))+' and this is how much teams typically score during this time period.')
temp_data = master_odds[master_odds['Location']=='away'][['Average_Line_OU']]
temp_data['year'] = temp_data.index.year
sns.boxplot(x='year',y='Average_Line_OU',data=temp_data)
plt.title('Box Plot of Year by Year O/U')
plt.show()


# ## Money Line
# Finally we take a look at the money line which is the odds a team will win the game. We can see that home teams are favored as road teams have a higher ML meaning they get paid out more if they win. Remeber that ML is quoted in 100 dollars. 

# In[ ]:


ml = master_odds[master_odds['Location']=='away']['Average_Line_ML']
ml.hist()
plt.title('Histogram of Road ML from 2012-2019')
plt.xlabel('ML')
plt.show()
avg_ml = ml.mean()
print('Average ML is '+str(np.round(avg_ml,2))+' for road teams, meaning according to the implied odds they have around a 44% chance of winning.')


# ## Model Predictions
# Spread and O/U rely on numeric targets while ML is a classification problem just determining win or loss. However classification has two other targets, if a team won the spread bet and if the team went over or under the total. So in total there are potentially 2 different regression models and 3 different classification models. Below are the 5 number summaries for the regression targets (again based on away team). 

# In[ ]:


regression_frame = pd.concat([spreads,ou],axis=1)
regression_frame.columns = ['Spread','OU']
regression_frame.describe().T


# Here we can see that the spread and over/under models do not suffer from class imbalance, but money line model does.

# In[ ]:


def classification_stats(x):
    return pd.Series([np.round(x.sum()/len(x),3),int(x.sum()),int(len(x))],index=['Percent','Counts','Total'])

road_cover = master_odds[master_odds['Location']=='away']['Spread']>master_odds[master_odds['Location']=='away']['Average_Line_Spread']*-1
over = master_odds[master_odds['Location']=='away']['Total']>master_odds[master_odds['Location']=='away']['Average_Line_OU']
road_win = master_odds[master_odds['Location']=='away']['Result']=='W'
classification_frame = pd.concat([road_cover,over,road_win],axis=1)
classification_frame.columns = ['Road Cover','Over Cover','Road Win']
classification_frame.apply(classification_stats).T

