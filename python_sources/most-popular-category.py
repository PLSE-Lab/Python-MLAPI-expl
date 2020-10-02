#!/usr/bin/env python
# coding: utf-8

# This notebook is for the Most popular category task from the google play store apps. 
# "What is the most popular category that has the largest number of installs."
# 
# My thanks to:
# * Lavanya Gupta for the dataset: https://www.kaggle.com/lava18/google-play-store-apps
# * Tinotenda Mhlanga for creating the task: https://www.kaggle.com/lava18/google-play-store-apps/tasks?taskId=276
# 
# as one side note: I'm trying to do one Kaggle task a week (this would be my second week) if there is anything i could do to improve the format I'd appriciate any constructive feedback. You can message me directly or comment on the notebook itself.

# The numbers used for this graphic are aproximations based on availible data. The dataset isn't precice when it comes to number of downloads so I got a bit creative to reach my solution. I'd love to hear your opinions on my solution down bellow. Did you agree with my aproximation or do you have a better idea?
# 
# ![image.png](attachment:image.png)

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


app_dat = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
app_dat.head()
## Notice how the installs column does not have an integer value


# In[ ]:


##Probably should have seen this mistake coming, gave me a good laugh
##essentially this just adds all the strings from "Installs" into one large superstring for each catagory

agg_df = app_dat.groupby(['Category'])['Installs'].sum()
display(agg_df)


# In[ ]:


#now to see exactly what string values are in the 'Installs' catagory
app_dat.Installs.unique()


# In[ ]:


app_dat.info()


# In[ ]:


app_dat.isnull().sum()


# In[ ]:


over_list = ['1,000,000,000+', '500,000,000+', '100,000,000+', '50,000,000+', '10,000,000+', '5,000,000+', '1,000,000+', '500,000+', '100,000+', '50,000+', '10,000+', '5,000+', '1,000+', '500+', '100+', '50+', '10+', '5+', '1+', '0+', '0']
#this list orders the bins for easier graphing             


# In[ ]:


import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot


# In[ ]:


matplotlib.pyplot.figure(figsize=(22,6))
sns.countplot(app_dat.Installs, order=over_list, orient='v')


# In[ ]:


#'Free' is a rather wacky value to have here, let's look closer
free = app_dat['Installs'] == 'Free'
display(app_dat[free])
wack = app_dat['Category'] == '1.9'
display(app_dat[wack])

##turns out this value is also unique in catagories as well. clearly this data is not intact
##this is an anomoly in the dataset and should be ignored
##by shifting all entries one column this entry makes sence but we wouldn't have an accurate catagory for it


# In[ ]:


## removing entry
app_dat.drop(app_dat[wack].index, axis=0, inplace=True)


# At this point I'll want to experiment with three different options for replacing the current values for 'Installs' and see how it affects the projection for most downloaded catagory:
# 
# * First option: set each value to the minimum possible
# 
# * Second option: set each value to the mid point between its minimum and maximum values.
# 
# * Third option: set each value to the maximum possible for the bin.
# 
# I've hidden the code cell bellow for the functions I'll use to populate the three test columns. The code is nothing impressive and has alot of lines.

# In[ ]:


## I could have used a dictionary and mapping function, which would have been more concise
## This perticular layout just helped me to see the gaps between values better
def installs_min(stng):
    if stng == '1,000,000,000+':
        return 1000000001
    if stng == '500,000,000+':
        return 500000001
    if stng == '100,000,000+':
        return 100000001
    if stng == '50,000,000+':
        return 50000001
    if stng == '10,000,000+':
        return 10000001
    if stng == '5,000,000+':
        return 5000001
    if stng == '1,000,000+':
        return 1000001
    if stng == '500,000+':
        return 500001
    if stng == '100,000+':
        return 100001
    if stng == '50,000+':
        return 50001
    if stng == '10,000+':
        return 10001
    if stng == '5,000+':
        return 5001
    if stng == '1,000+':
        return 1001
    if stng == '500+':
        return 501
    if stng == '100+':
        return 101
    if stng == '50+':
        return 51
    if stng == '10+':
        return 11
    if stng == '5+':
        return 6
    if stng == '1+':
        return 2
    if stng == '0+':
        return 1
    if stng == '0':
        return 0
    if stng == 'Free': ##this entry should be deleted in the final version but lets not take chances
        return 0
    
def installs_mid(stng):
    if stng == '1,000,000,000+': ##I assume 5 billion would come next, yes this is getting silly
        return 2500000000
    if stng == '500,000,000+':
        return 750000000
    if stng == '100,000,000+':
        return 250000000
    if stng == '50,000,000+':
        return 75000000
    if stng == '10,000,000+':
        return 25000000
    if stng == '5,000,000+':
        return 7500000
    if stng == '1,000,000+':
        return 2500000
    if stng == '500,000+':
        return 750000
    if stng == '100,000+':
        return 250000
    if stng == '50,000+':
        return 75000
    if stng == '10,000+':
        return 25000
    if stng == '5,000+':
        return 7500
    if stng == '1,000+':
        return 2500
    if stng == '500+':
        return 750
    if stng == '100+':
        return 250
    if stng == '50+':
        return 75
    if stng == '10+':
        return 25
    if stng == '5+':
        return 8
    if stng == '1+':
        return 3
    if stng == '0+':
        return 1
    if stng == '0':
        return 0
    if stng == 'Free':
        return 
    
def installs_max(stng):#shows max value
    if stng == '1,000,000,000+':
        return 5000000000
    if stng == '500,000,000+':
        return 1000000000
    if stng == '100,000,000+':
        return 500000000
    if stng == '50,000,000+':
        return 100000000
    if stng == '10,000,000+':
        return 50000000
    if stng == '5,000,000+':
        return 10000000
    if stng == '1,000,000+':
        return 5000000
    if stng == '500,000+':
        return 1000000
    if stng == '100,000+':
        return 500000
    if stng == '50,000+':
        return 100000
    if stng == '10,000+':
        return 50000
    if stng == '5,000+':
        return 10000
    if stng == '1,000+':
        return 5000
    if stng == '500+':
        return 1000
    if stng == '100+':
        return 500
    if stng == '50+':
        return 100
    if stng == '10+':
        return 50
    if stng == '5+':
        return 10
    if stng == '1+':
        return 5
    if stng == '0+':
        return 1
    if stng == '0':
        return 0
    if stng == 'Free': ##this entry should be deleted in the final version but lets not take chances
        return 0


# In[ ]:


app_dat['min_installs'] = pd.Series([installs_min(x) for x in app_dat.Installs], index=app_dat.index)
app_dat['mid_installs'] = pd.Series([installs_mid(x) for x in app_dat.Installs], index=app_dat.index)
app_dat['max_installs'] = pd.Series([installs_max(x) for x in app_dat.Installs], index=app_dat.index)
app_dat.head(3)


# In[ ]:


#Now to aggrigate and sort the data
full_app = app_dat.groupby(['Category'])['max_installs', 'mid_installs', 'min_installs'].sum()
full_app = full_app.sort_values(by=['min_installs'],ascending=False)
full_app


# In[ ]:


#mako
matplotlib.pyplot.figure(figsize=(10,10))
sns.barplot(x=full_app['min_installs'], y=full_app.index, palette='mako')

#Additional ideas: bin bottum 5% of apps


# In[ ]:


#full_app = full_app.sort_values(by=['mid_installs'],ascending=False)
matplotlib.pyplot.figure(figsize=(10,10))
#with sns.palplot(sns.color_palette("BuGn_r")):
sns.barplot(x=full_app['mid_installs'], y=full_app.index, palette='BuGn_r')
#sns.palplot(sns.color_palette("BuGn_r"))


# In[ ]:


matplotlib.pyplot.figure(figsize=(10,10))
sns.barplot(x=full_app['max_installs'], y=full_app.index, palette='Blues_r')
#when we replace each value with the max for its bin we start seeing communication overtake games as the installed catagory


# So looking above we can see that the most popular catagory is games, with communication being a close second (when viewing the middle values the difference is less than 5% and it only shifts to communication when all values are set to max). There is still some room for debate, but I'm reasonably confident in my result.

# In[ ]:


short_app = full_app.head(10) ##shorter list of the top entries
##one other alternative that I did not try was to group all the entries that have fewer than 5% of the installs


# The code for the below graphic was highly influenced by the tutorial on the website for the seaborn library:
# https://seaborn.pydata.org/examples/horizontal_barplot.html

# In[ ]:


import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 8))

sns.set_color_codes("pastel")
sns.barplot(x=short_app['max_installs'], y=short_app.index,
            label="Maximum possible value", color="b")

sns.set_color_codes("muted")
sns.barplot(x=short_app['mid_installs'], y=short_app.index,
            label="Median value", color="b")

sns.set_color_codes("dark")
sns.barplot(x=short_app['min_installs'], y=short_app.index,
            label="Minimum possible value", color="b")

ax.legend(ncol=3, loc="lower right", frameon=True)
ax.set( ylabel="Catagory",
       xlabel="Number of Installs")
sns.despine(left=True, bottom=True)


# # Further thoughts:
# 
# * I could have removed some of the uncertainty by viewing the distrobution of the # of ratings within each bin.
# * I was also curious, but ran out of time, about what the average number of installs for each catagory might be (I'm trying to do a task a week so thats were the self imposed dealine comes from).
# * Finally, this time I decided to show the final graphic in the first cell and then show my work below. Was this a more optimal layout for a response to a task?
# 
# As always I'd love to hear your thoughts, ideas, and constructive criticism in the comments below. And if you liked what you see here feel free to upvote the notebook and check out my profile for links to my other notebooks.
