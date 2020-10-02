#!/usr/bin/env python
# coding: utf-8

# Favorite Player Notebook: Messi
# 
# Erik Peterson
# 
# Last Modified: 1/26/2020

# # Homework 1

# The first cell has been made for you by Kaggle.
# 
# It imports a few packages and sets up a directory where your notebooks are saved.

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


# We must import a plotting package first

# In[ ]:


import matplotlib.pyplot as plt


# Next I am making a list that contains my data. I call that list **goals**. Beginning a line with # starts a comment

# In[ ]:


# Lionel Messi goals in La Liga
goals = [1,6,14,10,23,34,31,50,46,28,43,26,37,34,36,13]


# I want to print to my screen the number of seasons Messi has played in. This is the length of my list

# In[ ]:


print("number of seasons: ", len(goals)) # len() is the length of the list


# In order to find the range of goals scored, I find the minimum and maximum values in my list and print to my screen

# In[ ]:


print("range: ", np.min(goals), "-", np.max(goals))


# Printing the mean and median

# In[ ]:


print("goals per season: ", np.mean(goals))
print("median: ", np.median(goals))


# And mode (which is a bit trickier since we need to import something extra)

# In[ ]:


from statistics import mode #import the function 'mode'
print("mode: ", mode(goals))


# And now the histogram. We use the plotting package (which we've renamed as plt), label our axes, and show the plot.

# In[ ]:


plt.hist(goals,10) # Even though the homework says 6 bins, I've put 10 to see the histogram better
plt.xlabel("Goals")
plt.ylabel("N")
plt.title("Lionel Messi Goal Distribution")
plt.show()


# Looks a bit skewed left due to Messi's first 4 seasons!

# # Homework 2

# We want to make a new plot with our statistic over time. We start by creating another list which numbers our game or season

# In[ ]:


# goals over time

season_number = np.arange(len(goals))
print(season_number) # keep in mind, computers start counting at 0!


# Look at what you've printed to your screen. This new list matches up with the corresponding game or season (starting at 0)

# Now we use plt.plot in order to make this new diagram, we label, and we show the plot

# In[ ]:


plt.plot(season_number,goals)
plt.xlabel("Season Number")
plt.ylabel("Goals")
plt.title("Lionel Messi's goals over the years")
plt.show()


# Then we collect data on the other players, but we ensure that the seasons line up (inputting np.nan 'not a number' if they did not play in that season).

# In[ ]:


# Adding 4 other players
# Making sure these are the same seasons that they played in and filling in nothing if they didn't play
suarez_goals = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,16,40,29,25,21,11]
neymar_goals = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,9,22,24,13,np.nan,np.nan,np.nan]
pedro_goals = [np.nan,np.nan,np.nan,0,0,12,13,5,7,15,6,np.nan,np.nan,np.nan,np.nan,np.nan]
iniesta_goals = [2,0,6,3,4,1,8,2,3,3,0,1,0,1,np.nan,np.nan]


# We use the plot function each time for each player, we also label each line according to the player name and add a legend, and we label the axes and show the plot

# In[ ]:


plt.plot(season_number,goals,label = "Messi")
plt.plot(season_number,suarez_goals,label = "Suarez")
plt.plot(season_number,neymar_goals,label = "Neymar")
plt.plot(season_number,pedro_goals,label = "Pedro")
plt.plot(season_number,iniesta_goals,label = "Iniesta")
plt.legend()

plt.xlabel("Season Number")
plt.ylabel("Goals")
plt.title("Barcelona's goals over time")
plt.show()


# Messi's goals over time drastically increase as he's become more experienced until about season 7. He has remained relatively constant, but the sudden drop in the most recent season can be explained by the fact that the season is only 1/3 the way done.

# # Homework 3

# In this homework, we are going to work on adding uncertainties or error bars to our points/goals/statistic over time plot. We are also going to fit our data to a simple linear fit. (As a side note, try to comment your code with #'s so that you remember what you did later. This is very good coding practice and it is very hard to remember what you've done easily when your code gets messy!)

# Remember in Homework 1 you plotted your statistic in a histogram. Let's get a standard deviation of that histogram first

# In[ ]:


goals_std = np.std(goals)
print("Standard Deviation:", goals_std)


# We are going to use this standard deviation as an error bar for every point. There are many other ways we could calculate an uncertainty, but this is a simple and effective way.

# In[ ]:


# What I've commented off below is what we plotted in Homework 2. Look how similar the code is!
# plt.plot(season_number,goals)
plt.errorbar(season_number,goals,
             yerr=goals_std)
plt.xlabel("Season Number")
plt.ylabel("Goals")
plt.title("Lionel Messi's goals over the years")
# I also want to restrict the width of my y axis because negative goals are not possible (well, maybe own goals)
# Use your discression here, but think about what limits might make sense in your case!
plt.ylim(0,65)
plt.show()


# Now let's add a linear fit to the plot too. Import this tool for fitting curves

# In[ ]:


from scipy.optimize import curve_fit


# Now we need to define the function curve_fit will be using. In this case, a simple line

# In[ ]:


def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B


# We now apply curve_fit on our data. This function uses f (which we defined above), our x, and our y. It gives us a lot of good information, some of which I have printed below.

# In[ ]:


popt, pcov = curve_fit(f, season_number, goals) # your data x, y to fit
print("Slope:",popt[0]) 
print("Intercept:",popt[1])


# Now we need to define the y values that correspond to the fit

# In[ ]:


# y = m*x + b
y_fit = popt[0]*season_number + popt[1]


# And then we plot the same stuff as above, adding the extra plt.plot for our fit

# In[ ]:


plt.errorbar(season_number,goals,
             yerr=goals_std)
# the fit!
plt.plot(season_number, y_fit,'--')
plt.xlabel("Season Number")
plt.ylabel("Goals")
plt.title("Lionel Messi's goals over the years")
plt.ylim(0,65)
plt.show()


# The linear fit does not look good here! We should probably do a polynomial fit.
