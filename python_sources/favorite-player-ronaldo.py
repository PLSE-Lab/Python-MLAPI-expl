#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
# Cristiano Ronaldo goals from 2008-2009 season to 2019-2020
goals = [18,26,40,46,34,31,48,35,25,26,21,14]
print( "the number of seasons: ", len(goals))
print( "range: ", np.min(goals), "-", np.max(goals) )      #np->numpy linear algebra
print( "goals per season: ", np.around(np.mean(goals),1))  #used around to keep 1 decimal place
print( "median: ", np.median(goals))
from statistics import mode #import mode method
print( "mode: ", mode(goals))

plt.hist(goals,6)
plt.xlabel("Goals") #label x-axis
plt.ylabel("Frequency") #label y-axis
plt.title("Cristiano Ronaldo Goal Distribution") #mark title
plt.show()


# The histogram is close a normal distribution, but the tails also have a relatively large frequency. This is probably due to Ronaldo's weak performance in his early career and recent years when he is aged.
# 

# That the distribution does not look normal means Ronaldo's ability is not stable across seasons.

# In[ ]:


season_number = np.arange(len(goals))
print(season_number + 2008)


# The printed out Season Number matches the year begins for one season. For example, 2008 here means 2008-2009 season

# In[ ]:


plt.plot(season_number + 2008,goals)
plt.xlabel("Season Year")
plt.ylabel("Goals")
plt.title("Cristiano Ronaldo's goals over year")
plt.show()


# This graph shows Ronaldo's goals from 2008-2009 season in Manchester United, then to Real Madrid until 2019-2020 season in Juventus. We can see the peak are made in 2014-2015 season with 48 goals.

# In[ ]:


#add 4 other players
Hazard_goals = [4,5,7,20,9,14,14,4,16,np.nan,12,16]
Bale_goals = [np.nan,3,7,10,21,15,13,19,7,16,8,2]
Benzema_goals = [17,8,15,21,11,17,15,24,11,5,21,12]
Modric_goals = [3,3,3,4,3,1,1,2,1,1,3,3]

plt.plot(season_number + 2008, goals, label = "Ronaldo" )
plt.plot(season_number + 2008, Hazard_goals, label = "Hazard")
plt.plot(season_number + 2008, Bale_goals, label = "Bale")
plt.plot(season_number + 2008, Benzema_goals, label = "Benzema")
plt.plot(season_number + 2008, Modric_goals, label = "Modric")
plt.legend()

plt.xlabel("Season Year")
plt.ylabel("Goals")
plt.title("Real Madrid's goals over time")
plt.show()


# The Season Year is marked from 2008-2009 up to 2019-2020 season.
# As we can see from the graph, Ronaldo, Benzema, and Hazard all performed well in 2011-2012 season. Ronaldo's performance was best when he was at Real Madrid, and started to drop when joined Juventus. Hazard, Bale, Benzema, Modric all have relatively stable performance over the years.
# One systematic error could be that the number of games the player played. For example, Ronaldo played 38 games in 2014-2015 season and had 48 goals, while he only played 30 games with 31 goals in 2013-2014 season.
# Modric's number of goals cannot be marked as inferior to other players. Unlike others who are Forwards, Modric is in Midfielder's position, which means that chances for him to shoot goals are relatively small. However, comparison among all Forward players shows that Ronaldo was more skillful than others in the last ten years.

# In[ ]:


goals_std = np.std(goals)
print("Standard Deviation:", goals_std)


# This standard deviation tells that average season is about 10.24 points from the mean, which is 30.3 as calculates in HW1.

# In[ ]:


plt.errorbar(season_number + 2008 ,goals,yerr=goals_std) 
#The error bar is added to each season to show uncertainty, yerr means error bar in y-axis, in this case
#the standard deviation of goals
plt.xlabel("Season Year")
plt.ylabel("Goals")
plt.title("Cristiano Ronaldo's goals over the years")
plt.ylim(0,60)
plt.show()


# I chose the limit from 0 to 60 because it does not make sense if the goal can reach negative. The top is 60 is because Ronaldo's max goal did not reach over 50 and with error bar not above 60.

# The code "from scipy.optimize import curve_fit" extract the tool called curve_fit from the scipy.optimize to make a regression line

# In[ ]:


from scipy.optimize import curve_fit
def f(x,A,B): return A*x + B
popt,pcov = curve_fit(f,season_number, goals)
print("Slope:", popt[0])
#popt returns the array of A and B, so popt[0] means A
print("Intercept:", popt[1])
#This returns the value B in popt array


# In[ ]:


y_fit = popt[0]*season_number + popt[1]
#y= A*x + B
plt.errorbar(season_number + 2008, goals, yerr = goals_std)
plt.plot(season_number + 2008, y_fit, '--') #'type of line used','-'would be a continuous line
plt.xlabel("Season Year")
plt.ylabel("Goals")
plt.title("Cristiano Ronaldo's goals over the years")
plt.ylim(0,60)
plt.show()


# The regression line fits relatively well considering Ronaldo's declining performance in his recent seasons,
# but a polynomial fit would better take his early rising years into account.
# He is trending downward after year 2014, so the slope is downward.
# Not all my error bars reached the fit. Year 2011 and 2014 are his top seasons while 2008 and this year are too poor.

# In[ ]:


goals_per_90min = [0.59,0.95,1.24,1.24,1.13,1.10,1.39,0.99,0.89,1.02,0.90,1.02]
plt.plot(season_number + 2008,goals_per_90min)
plt.xlabel("Season Year")
plt.ylabel("Goals per 90 minutes")
plt.title("Ronaldo's Goals/90min over time")
plt.show()


# Goals per season is not enough to tell the player's performance, but considering him as a forward player, this graph of his goal per 90 minutes is great to show his scoring ability. It also considers the scenario that he might just play a part of the game. Therefore, looking at this graph alone, we can tell Cristiano Ronaldo's peak years is 2014.

# In general, this project overviews Cristiano Ronaldo's performance from 2008-2009 to 2019-2020 season, starting from his service at Manchester United to Real Madrid and now at Juventus. He reached the peak of his career in 2014-2015 season with 48 goals, and 1.4 goals per 90 minutes. Though in recent years his performance starts to decline, his outstanding scoring ability still outperforms many of other forward players. Hope he keeps going as a top star into 2020 and bring glories to Juventus.
