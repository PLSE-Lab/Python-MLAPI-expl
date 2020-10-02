#!/usr/bin/env python
# coding: utf-8

# I was asked a rather interesting question by a friend. And always looking to bone up on my Python (which is rookie-level at best), I gave it a shot. Out of all of the winners of the World Series in the modern era (since 1900), we want to find out who was the "worst" team to actually win it. The analysis is rather cursory, so we'll simply use low winning percentage and worst run differential per 162 games as our metrics. Also, thanks to Nel Abdiel who wrote a script for this dataset...much of my code is modeled after your own. Let's begin!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

#Ignore warnings
warnings.filterwarnings('ignore')

#Read through team.csv
data = pd.read_csv('../input/team.csv')

#Get the rows needed
d1 = data[['year','team_id','g','w','r','ra','ws_win']]

#We want only WS winners
d2 = d1[d1['ws_win']=='Y']

#Modern era (year > 1900)
d3 = d2[d2['year']>1900]


# Now, the reason we adjust for run differential is because not all WS winning teams played 162 games. Most notably, the strike-shortened season of 1981. So we adjust for a 162-game season to make things as level as possible. 

# In[ ]:


#Calculate winning %
d3['winpct'] = d3['w']/d3['g']*100

#Adjusted Run Diff
d3['ard'] = (d3['r']/d3['g']*162) - (d3['ra']/d3['g']*162)

#New data
d4 = d3[['year','team_id','ard','winpct']]

sorted = pd.DataFrame.sort_values(d4,'ard')

print(sorted.head(10))


# Sorted by Adjusted Run Differential, the 1987 Twins and 2006 Cardinals look like outliers. The Twins even had a negative run differential. Plot the points:

# In[ ]:


#Plot data
x = d4['ard']
y = d4['winpct']

plt.axis([-30,x.max()+10, 50, y.max()+10])
plt.xlabel('Adjusted Run Differential')
plt.ylabel('Win %')
plt.title("Worst WS team, modern era")

plt.scatter(x,y)

plt.annotate('1987 Twins', xy=(-19,52.7), xycoords='data', xytext=(-20, 63), size=10, arrowprops=dict(arrowstyle="simple",fc="0.6", ec="none"))
plt.annotate('2006 Cardinals', xy=(25,51.552795), xycoords='data', xytext=(100, 52), size=10, arrowprops=dict(arrowstyle="simple",fc="0.6", ec="none"))


# Conclusion:
# 
# The Twins and Cardinals are clear outliers. Without exploring too much into it, we can consider these 
# two teams to be the "worst" World Series winners. Of course, there are plenty of factors that could 
# influence this. For instance, the Twins of 1987 actually held the 5th best record out of 14 teams 
# in the AL. But because of playing in a weak division, only two divisions in the AL, and no divisional 
# series/wild card, they were able to knock off the Tigers in 5 games and the Cardinals in 7 games to win it 
# all. Injuries may also play a factor as well, as seemingly no championship team goes uninjured. But based
# solely on these metrics, Kirby Puckett's Twins and Albert Pujols' Cardinals certainly couldn't be considered
# "favorites" to win it all, to say the least.
