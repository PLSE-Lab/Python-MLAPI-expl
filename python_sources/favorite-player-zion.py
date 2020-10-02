#!/usr/bin/env python
# coding: utf-8

# Favorite Player Notebook: Zion Williamson
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


# Next I am making a list that contains my data. I call that list **points**. Beginning a line with # starts a comment

# In[ ]:


# Zion Williamson
points = [28,27,21,13,13,22,25,17,18,20,17,17,25,30,11,35,27,25,22,26,29,16,18,27,32,0,29,31,21,25,32,23,24]


# I want to print to my screen the number of games Zion played in. This is the length of my list

# In[ ]:


print("number of games: ", len(points)) # len() is the length of the list


# In order to find the range of scores, I find the minimum and maximum values in my list and print to my screen

# In[ ]:


print("range: ", np.min(points), "-", np.max(points))


# Printing the mean and median

# In[ ]:


print("points per game: ", np.mean(points))
print("median: ", np.median(points))


# And mode (which is a bit trickier since we need to import something extra)

# In[ ]:


from statistics import mode #import the function 'mode'
print("mode: ", mode(points))


# And now the histogram. We use the plotting package (which we've renamed as plt), label our axes, and show the plot.

# In[ ]:


plt.hist(points,6) # 6 bins
plt.xlabel("Points")
plt.ylabel("N")
plt.title("Zion Williamson Point Distribution")
plt.show()


# Looks a bit skewed left due to the 0 points against UNC!

# # Homework 2

# We want to make a new plot with our statistic over time. We start by creating another list which numbers our game or season

# In[ ]:


# points over time

game_number = np.arange(len(points))
print(game_number) # keep in mind, computers start counting at 0!


# Look at what you've printed to your screen. This new list matches up with the corresponding game or season (starting at 0)

# Now we use plt.plot in order to make this new diagram, we label, and we show the plot

# In[ ]:


plt.plot(game_number,points)
plt.xlabel("Game Number")
plt.ylabel("Points")
plt.title("Zion Williamson's points over time")
plt.show()


# Then we collect data on the other players, but we ensure that the games played line up (inputting np.nan 'not a number' if they did not play in that game).

# In[ ]:


# Adding 4 other players
# Making sure these are the same games that they played in and filling in nothing if they didn't play
rj_points = [33,23,20,20,18,23,22,26,27,30,27,16,13,21,32,23,30,26,24,17,15,19,26,13,23,33,23,15,17,26,16,18,21]
tre_points = [6,8,2,14,10,17,15,0,10,6,3,13,10,6,8,2,np.nan,np.nan,6,9,13,11,13,6,13,3,15,11,18,5,11,22,4] #np.nan means "not a number"
cam_points = [22,25,3,16,18,10,13,23,5,10,9,8,4,10,23,np.nan,9,15,7,13,16,24,17,22,9,27,7,6,11,12,13,np.nan,8]
bold_points = [7,0,8,4,11,6,0,4,4,2,7,0,11,12,3,12,2,7,2,8,10,6,5,2,9,0,np.nan,np.nan,np.nan,2,0,4,0]


# We use the plot function each time for each player, we also label each line according to the player name and add a legend, and we label the axes and show the plot

# In[ ]:


plt.plot(game_number,points,label = "Williamson")
plt.plot(game_number,rj_points,label = "Barrett")
plt.plot(game_number,tre_points,label = "Jones")
plt.plot(game_number,cam_points,label = "Reddish")
plt.plot(game_number,bold_points,label = "Bolden")
plt.legend()

plt.xlabel("Game Number")
plt.ylabel("Points")
plt.title("Duke's points over time")
plt.show()


# # Homework 3

# In this homework, we are going to work on adding uncertainties or error bars to our points/goals/statistic over time plot. We are also going to fit our data to a simple linear fit. (As a side note, try to comment your code with #'s so that you remember what you did later. This is very good coding practice and it is very hard to remember what you've done easily when your code gets messy!)

# Remember in Homework 1 you plotted your statistic in a histogram. Let's get a standard deviation of that histogram first

# In[ ]:


points_std = np.std(points)
print("Standard Deviation:", points_std)


# We are going to use this standard deviation as an error bar for every point. There are many other ways we could calculate an uncertainty, but this is a simple and effective way.

# In[ ]:


# What I've commented off below is what we plotted in Homework 2. Look how similar the code is!
# plt.plot(game_number,points)
plt.errorbar(game_number,points,
             yerr=points_std)
plt.xlabel("Game Number")
plt.ylabel("Points")
plt.title("Zion Williamson's points over time")
# I also want to restrict the width of my y axis because negative points are not possible (well, maybe...)
# Use your discression here, but think about what limits might make sense in your case!
plt.ylim(0,45)
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


popt, pcov = curve_fit(f, game_number, points) # your data x, y to fit
print("Slope:",popt[0]) 
print("Intercept:",popt[1])


# Now we need to define the y values that correspond to the fit

# In[ ]:


# y = m*x + b
y_fit = popt[0]*game_number + popt[1]


# And then we plot the same stuff as above, adding the extra plt.plot for our fit

# In[ ]:


plt.errorbar(game_number,points,
             yerr=points_std)
# the fit!
plt.plot(game_number, y_fit,'--')
plt.xlabel("Game Number")
plt.ylabel("Points")
plt.title("Zion Williamson's points over time")
plt.ylim(0,45)
plt.show()


# This actually looks like a reasonable fit! It is good to see his trend going upwards over time.
