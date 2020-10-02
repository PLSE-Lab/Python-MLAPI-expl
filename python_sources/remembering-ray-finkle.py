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


# # Loading in the Data
# 
# We begin by loading in the data and taking a look at the first five observations.

# In[ ]:


df = pd.read_csv("/kaggle/input/superbowl-history-1967-2020/superbowl.csv")
df.head()


# # Separating Years from the Dates
# 
# It would be useful to have a `year` column, so let's do that.

# In[ ]:


df['year']= df['Date'].apply(lambda x: x.split()[2])
df.head()


# # Dealing with Data Types
# 
# Let's check to see what data types are we dealing with.

# In[ ]:


df.dtypes


# Let's make the `year` column an integer column.

# In[ ]:


df = df.astype({'year': 'int'})
df.dtypes


# # Data Visualization
# 
# Let's make a plot that helps us think about the following questions:
# 
#  * How many points did the winners and losers of each SuperBowl game score?
#  * Were there any games that were very close?
#  * Were there any blowouts?
#  * Is there a "trend" in margin of victory over time?  In other words, are games getting "closer"?
# 

# In[ ]:


import matplotlib.pyplot as plt
style = dict(size = 10, color = 'gray')
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (20,6)
plt.plot(df['year'], df['Loser Pts'], marker = "o")
plt.plot(df['year'], df['Winner Pts'], marker = "o")
plt.title("Number of Points Scored by the Winners and Losers of Each SuperBowl Game")
plt.xlabel("Year")
plt.ylabel("Number of Points")
plt.legend()
plt.annotate('Very close!', xy = (1991,18), xytext=(1992,5), arrowprops = dict(facecolor = 'green', shrink = 0.05))
plt.annotate('Blowout!', xy = (1990,55), xytext=(1992,55), arrowprops = dict(facecolor = 'green', shrink = 0.05))
plt.annotate('Wide Margins of Victory for a few years', xy = (1984,40), xytext=(1970,50), arrowprops = dict(facecolor = 'green', shrink = 0.05))
plt.annotate("Fairly Tight Margins of Victory Since the Mid 2000's", xy = (2004,35), xytext=(2010,50), ha = "center", arrowprops = dict(facecolor = 'green', shrink = 0.15))
plt.show()


# Out of curiousity, what was the very close game in 1991?

# In[ ]:


df[df['year'] == 1991]


# You can see highlights of the game on [YouTube](https://www.youtube.com/watch?v=XxsZf9G_W14).  The Bills kicked a field goal with a few seconds left to try to win the game, but they missed!  It appears as though the story of the [kicker](https://en.wikipedia.org/wiki/Scott_Norwood) was the insipration for Ray Finkle in [Ace Ventura: Pet Detective](https://en.wikipedia.org/wiki/Ace_Ventura:_Pet_Detective).

# # A Closer Look at Victory Margins
# 
# Has there been any sort of change in 'Victory Margins' over time?
# 
# Let's create a variable called 'Victory Margin' and then compute its mean and standard deviation.

# In[ ]:


df['Margin'] = df['Winner Pts'] - df['Loser Pts']
df['Margin'].describe()


# Now let's plot the margin of victory over time.

# In[ ]:


from scipy.interpolate import InterpolatedUnivariateSpline
df = df.sort_values('year')
ius = InterpolatedUnivariateSpline(df['year'],df['Margin'])
xi = np.linspace(1967,2020, 1000)
yi = ius(xi)

# Mean Margin
plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407)
# One Standard Deviation Above
plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407 + 10.314431)
# One Standard Deviation Below
plt.plot(df['year'], np.ones(df['year'].shape[0])*13.907407 - 10.314431)

plt.scatter(df['year'], df['Margin'])
plt.plot(xi, yi)

plt.title("Is there any pattern in 'Margin of Victory' over the years?")
plt.xlabel("Year")
plt.ylabel("Margin of Victory")
plt.show()


# I don't really see a pattern.

# # Winners by Conference
# 
# [Wikipedia](https://en.wikipedia.org/wiki/List_of_Super_Bowl_champions) has information on the conferences of the winners.  
# 
# It appears as though the NFL had a different conference structure before 1970 onwards, so we will restrict our attention to 1970 and onwards.

# In[ ]:


# entered by hand from wikipedia
winningConferenceEncoded = [0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1]
past70 = df[df['year'] >= 1970].copy()
past70['WinningConferenceEncoded'] = pd.Series(winningConferenceEncoded)
past70['LosingConferenceEncoded'] = 1 - past70['WinningConferenceEncoded']
past70['WinningConference'] = past70['WinningConferenceEncoded'].map({0:'NFC', 1:'AFC'})
past70['LosingConference'] = past70['LosingConferenceEncoded'].map({0:'NFC', 1:'AFC'})
past70.drop('WinningConferenceEncoded', axis = 1, inplace = True)
past70.drop('LosingConferenceEncoded', axis = 1, inplace = True)
past70.head()


# To help create the next couple of plots, I want to create two dataframes:  
# 
#  * one with all of the AFC info (including whether they won or not) and
#  * one with all of the NFC info (including whether they won or not).
#  
# There may be a better way to do this, but, for now, we begin by creating a long data frame with the following variables:
# 
#  * Conference
#  * Points
#  * Year
#  * Won/Lost

# In[ ]:


winCols = ['WinningConference', 'Winner Pts' , 'year']
loseCols = ['LosingConference', 'Loser Pts' , 'year']


win_df = past70[winCols].copy()
win_df = win_df.rename(columns = {'WinningConference':'Conference', 'Winner Pts' :'Points', 'year':"Year"})
win_df['Win'] = 1

lose_df = past70[loseCols].copy()
lose_df = lose_df.rename(columns = {'LosingConference':'Conference', 'Loser Pts' :'Points', 'year':"Year"})
lose_df['Win'] = 0

conference_info = win_df.append(lose_df, ignore_index = True, sort = False)
conference_info.head()


# Then we split this up into an "NFC" data frame and an "AFC" data frame.

# In[ ]:


nfc_info = conference_info[conference_info['Conference']  == 'NFC'].copy()
nfc_info.rename(columns = {'Points':'NFC'}, inplace = True)
nfc_info.sort_values('Year', inplace= True)

afc_info = conference_info[conference_info['Conference']  == 'AFC'].copy()
afc_info.rename(columns = {'Points':'AFC'}, inplace = True)
afc_info.sort_values('Year', inplace= True)


# Then we create a "Cumulative Wins" column.

# In[ ]:



nfc_info['NFC Cumulative Wins'] = nfc_info['Win'].cumsum()
afc_info['AFC Cumulative Wins'] = afc_info['Win'].cumsum()


# > Now we can make two plots.  One is the the scores over the years by conference.

# In[ ]:


import matplotlib.pyplot as plt
style = dict(size = 10, color = 'gray')
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (20,6)
plt.plot(nfc_info['Year'], nfc_info['NFC'], marker = "o")
plt.plot(afc_info['Year'], afc_info['AFC'], marker = "o")
plt.title("Number of Points Scored by the AFC and NFC of Each SuperBowl Game since 1970")
plt.xlabel("Year")
plt.ylabel("Number of Points")
plt.legend()
plt.annotate('The NFC Reigned for Several Years Here', xy = (1996,40), xytext=(1998,52), ha = "left", arrowprops = dict(facecolor = 'green', shrink = 0.05))
plt.show()


# The next is the "running total" of superbowl wins by conference.

# In[ ]:


import matplotlib.pyplot as plt
style = dict(size = 10, color = 'gray')
plt.style.use('seaborn-whitegrid')
plt.rcParams["figure.figsize"] = (20,6)
plt.title("Number of Superbowl Wins Accumulated by the AFC and NFC since 1970")
plt.plot(nfc_info['Year'], nfc_info['NFC Cumulative Wins'], marker = "o")
plt.plot(afc_info['Year'], afc_info['AFC Cumulative Wins'], marker = "o")
plt.xlabel("Year")
plt.ylabel("Number of Wins")
plt.legend()
plt.show()


# It looks like the NFC has had 26 superbowl victories since 1970, while the AFC has had 25 victories.  We can check this as follows.

# In[ ]:


conference_info['Tally'] = 1
print(conference_info.groupby(['Conference', 'Win']).count()['Tally'])


# # Conclusion
# 
# WIth this notebook, I mainly wanted to brush up on my `matplotlib` skills.  Still, I found out a couple of interesting facts:
# 
#  * Ace Ventura: Pet Detective based a key plot point off of a real-life event (SuperBowl 25)
#  * The NFC and the AFC are very close when it comes to "total number of SuperBowl wins" (26 to 25).
#  
# Also, this notebook showed how to use `scipy.interpolate`'s `InterpolatedUnivariateSpline` to smooth out time series data (in the Victory Margins plot).
# 
# 

# In[ ]:




