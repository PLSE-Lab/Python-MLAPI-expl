#!/usr/bin/env python
# coding: utf-8
Does the videogame platform influences it's favorites genres?
# As a way of simplifying our analysis and make it fit our data, in this analysis, we are considering the variable
# global sales as a sign game's popularity, and therefore, an indication of the average preferred genres.

# In[ ]:


#Importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# First of all, what are the platforms with the highest global sales?
# I'll be working with only the 10 top global sales platforms, for the sake of organization and keeping the analysis short.

# In[ ]:


filepath = '../input/videogamesales/vgsales.csv'
columns_to_drop = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
df = pd.read_csv(filepath)
df = df.drop(columns=columns_to_drop)
df_bar = df.groupby(by=df['Platform']).sum()
df_bar = df_bar.sort_values(by='Global_Sales', ascending=False)
df_bar = df_bar.iloc[0:10] #Amount of videogames to plot(Bigger to smaller)
plt.figure(figsize=(12,8))
plt.title('Global sales by platform')
sns.barplot(x=df_bar.index, y=df_bar['Global_Sales'])


# Now, let's plot the **most sold genres of the 10 most sold platforms**, to look out for any correlations.
# 
# My hypothesis is that there are some variables that cause interference on the preferred genres, e.g:
#   * The company who made the platform
#   * Popularity of games avaliable to that platform(on that, platforms that are a sequence of another would have a heavy influence on preferred genres(PS3 and PS4, for example.)
#   * The nature of the platform(Portable, played by controller, keyboard and mouse)
#   * The technology avaliable(hardware and software) at the time of it's popularity peak.

# At the code below, we simply iterated over common indexes to the Platform mask and the x ax list. Then, we simply plot with barplot()
#  to 
# 
# *I initially wanted genres to have the same color on different plots, but with time I realized that not every genre is contained in all plots, which would make it way too complicated to implement and distract from the important concepts*

# In[ ]:



fig, axs=plt.subplots(5,2, figsize=(16,25))
axes_split = [axs[0, 0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1], axs[4,0], axs[4,1]]
top_platforms = ['PS','PS2', 'PS3','PS4','PSP', 'Wii','DS','X360',  'GBA', 'PC']
#ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10
fig.suptitle('Global sales of Genres')

for i in range(10):
    df_mask = df['Platform'] == top_platforms[i]
    df_general = df[df_mask]
    df_general = df_general.groupby(by=df['Genre']).sum().sort_values(by='Global_Sales', ascending=False)
    df_general = df_general.iloc[0:10]
    sns.barplot(x=df_general.index, y=df_general['Global_Sales'], ax=axes_split[i])
    axes_split[i].set_title(top_platforms[i])
    axes_split[i].xaxis.set_tick_params(rotation=20)
    axes_split[i].set_xlabel('')


# # So, what can we partially conclude, so far, from basic bar plottings?

# * The platform genre became way less played with the advent of better hardware and software, only remaining relevant on portable platforms(DS and PSP).
# 
# * Also, there is a big difference between DS and PSP on the platform genre, which could indicate that the titles available to the platform play a big role(All the super mario's titles on DS, for example, seem to play a big role.
# 
# * The sports genre is very popular on Wii, which could be partially explained by the single-handed controllers with movement sensors(games such as Wii Tennis become more intuitive).
# 
# * Genres that are popular on computer usually are very time demanding(like role-play and strategy games), which means that, usually, computer gamers don't tend to be casual players(Also, the fact that it is usually more comfortable to play by yourself on computers makes less acessible to casual players).
# 
# * Genres that are popular on Wii are usually casual played, which could mean that the way the hardware was designed tend to lead people to use it as a way of interacting with other people in real life.
# 
# * Controller platforms(Such as PS2,PS3,PS4,XBOX)tend to prefer more intense genres.

# To finish up the first analysis, there is a lot that points towards validating the theory that, yes, videogame platforms influences It's favorites genres. 

# **This is my first exploratory analysis, any feedback is welcome.**

# 
