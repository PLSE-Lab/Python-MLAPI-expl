#!/usr/bin/env python
# coding: utf-8

# 
# ## Abstract ##
# This report is focused on trying to figure out what factors of a video game result in the most sales. To figure this out we figure out the average amount of sales of each video game that have a certain trait. We tested five variables - critic score, user score, genre, platform, and ESRB rating. We believed that the a high quality (high user and critic score), E-rated, role-playing, PC game would be the best combination but from our analysis we discovered that a high quality, M-rated, platforming, PS4 game had the most average sales per game. 

# ## Introduction ##
# My friends and I are interested in video games and have played them our whole lives. During our years of video game experience, both highly anticipated and barely advertised video games have reached the top of the selling charts. So, what makes the most profitable video game? 
# 
# Others have hypothesized and tried to prove their hypothesis with reasoning (see "Recipe for Creating the Best Selling Video Games - EXPOSED"), but I have not seen a report using big data to prove these hypotheses. There was a similar report using this data set (see "Video games sales vs ratings and critic biases") but it only focused on the rating aspect of video games and we would like to explore as many of these characteristics as possible. These top selling games varied in many ways - genre, quality, rating, platform availability, pricing, etc - and we are wondering which combination of these factors would make for the highest selling game. 
# 
# The data set used in this report only had five variables we could test to see if it contributed to the video game's sales - ESRB rating, genre, platform, critic score, and user score. We hypothesize that the golden combination would be an E-rated game in the role-playing genre available on the PC with high critic and user scores. An E-rated game would have the highest audience availability and most likely appeal to people of all ages. A game in the role-playing genre would allow players to try play the game as if it were real or fill a fantasy and role play as a hero. Essentially, a role-playing game would be best selling because your personality and choice affects the game which would attract a lot of sales and players. We believe the PC would be the highest selling platform because although everyone may not have a console, most households have computers so you would have more overall sales. Finally, critic score and user score should be higher for high selling games as a higher quality game would get more sales purely by reputation.

# ## The Data ##
# The data set I will be using is "Video Game Sales and Ratings." This data was collected by a "...web scrape from VGChartz and Metacritic along with manually entered year of release values for most games with a missing year of release" (Video Games Sales and Ratings) and then put into a CSV file which makes the data easy to read and to utilize.

# ## Importing Modules ##
# 
#  - csv: type of file we are reading
#  - pandas: puts the data in a dictionary structure that is easy to work with
#  - matplotlib: plotting library that is essential for most other plotting module
#  - seaborn: easy to use plotting module

# In[ ]:


import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Analysis ##
# 
# Here we read the csv data set and convert it into a pandas data frame which will make the data easier to work with

# In[ ]:


reader = pd.read_csv('../input/Video_Game_Sales_as_of_Jan_2017.csv')
gameData = pd.DataFrame(reader)
gameData


# Now we want to see if there is a relationship between critic score/user score and global sales to see if quality games actually do sell more so we create a scatter plot comparing critic score to global sales and figure out the correlation value.

# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(gameData['Critic_Score'], gameData['Global_Sales'], '.', alpha = 0.5)
plt.xlabel('Critic Score')
plt.ylabel('Gloabl Sales')
plt.title('Does Critic Score affect Global Sales?')


# As predicted, higher critic scores generally lead to higher sales. Now let's check if a higher user score also leads to higher sales.

# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(gameData['User_Score'], gameData['Global_Sales'], '.', alpha = 0.5)
plt.xlabel('User Score')
plt.ylabel('Gloabl Sales')
plt.title('Does User Score affect Global Sales?')


# Again, a higher user score correlates to higher global sales as predicted but finding the correlation values for critic/user score and global sales we see that the correlation isn't strong enough.

# In[ ]:


gameData.corr()


# Next, let's move on to analyzing rating and which ESRB rating results in the most sales.

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ratingCountPlot = sb.countplot(x='Rating',data=gameData)
ratingCountPlot.set_ylabel("Number of Games")


# This countplot shows that there are game ratings that are insignifcant so let's make a new dictionary only using the ratings needed. Also, we see that there are a lot more games with the rating E versus M, T, or E10+ so we total global sales would be skewed. Instead, we should calculate the mean global sales which would calculate the average amount of copies sold per rating.

# In[ ]:


ratingGameData = gameData[gameData['Rating'].isin(['E','E10+','M','T'])]
ratingGameData[['Global_Sales']].groupby(ratingGameData['Rating']).mean()


# Interestingly enough, this data proves our hypothesis wrong and shows that M rated games have the most sales per video game.
# 
# The process to see if genre affects global sales is similar to that of rating. Let's check to see if there are any irrelevant genres that do not have a strong enough sample size and then calculate the mean global sales to calculate the average amount of copies sold per genre.

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
genreCountPlot = sb.countplot(x='Genre',data=gameData)
genreCountPlot.set_ylabel("Number of Games")


# All genres seem to have a large enough sample size so let's see which genre sells the most copies per video game.

# In[ ]:


gameData[['Global_Sales']].groupby(gameData['Genre']).mean()


# From this, we can see that platforming games sell the most copies per game.
# 
# Now we move on to platform. Which console has the highest amount of copies sold per game? This process is again similar to the processes for genre and rating but first let's sort out and see all the platforms.

# In[ ]:


gameData.groupby(gameData['Platform']).count()


# Turns out there are a lot of gaming consoles out there. However, in a modern market only the latest and most recent consoles would be relevant to high selling games. So let's make a new dictionary with only relevant consoles.

# In[ ]:


platformGameData = gameData[gameData['Platform'].isin(['PC','3DS','PS4','WiiU','XOne'])]


# Now that we have the most relevant consoles, let's calculate the mean global sales for each of them and see which platform sells the most copies of a single video game.

# In[ ]:


platformGameData[['Global_Sales']].groupby(platformGameData['Platform']).mean()


# Here we can see that the PS4 is by far the highest selling console for the amount of sales per video game.

# ## Conclusion ##
# From the analysis above it seems that the highest selling video game is an M rated game in the platforming genre on the PS4 console that is also a high quality game. 
# 
# In the future, using a data set that has more variable including pricing and funding could also lead to see which games make for the best selling video games. 

# ## Teamwork ##
# We were all pretty involved with every single step of the process. We all searched through data sets together and found that the video game idea was the most intriguing. Since we did the report in a notebook, the write-up and code were all worked on simultaneously and we all took turns coding and writing. Nathaniel did the most work on the coding end, learning the seaborn module in for better plots and visualizations. Liam did most of the write-up outside of the coding like the abstract, introduction, conclusion etc. CJ put more work into researching other reports similar to ours and directed our approach to analyzing the data and what data analyzing methods make the most sense.

# 

# In[ ]:




