#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# This dataset below shows a list of video games of different Genre with sales greater than 100,000 copies.
# 
# Rank - Ranking of overall sales
# 
# Name - The games name
# 
# Platform - Platform of the games release (i.e. PC,PS4, etc.)
# 
# Year - Year of the game's release
# 
# Genre - Genre of the game
# 
# Publisher - Publisher of the game
# 
# NA_Sales - Sales in North America (in millions)
# 
# EU_Sales - Sales in Europe (in millions)
# 
# JP_Sales - Sales in Japan (in millions)
# 
# Other_Sales - Sales in the rest of the world (in millions)
# 
# Global_Sales - Total worldwide sales.

# In[ ]:


video_game_records = pd.read_csv("../input/vgsales/vgsales.csv")
video_game_records


# In[ ]:


Total_North_America_Sales = round(sum(video_game_records.NA_Sales) * 1000000)
Total_North_America_Sales


# In[ ]:


Total_Europe_Sales = round(sum(video_game_records.EU_Sales) * 1000000)
Total_Europe_Sales


# In[ ]:


Total_Japan_Sales = round(sum(video_game_records.JP_Sales) * 1000000)
Total_Japan_Sales


# In[ ]:


Total_Other_Sales = round(sum(video_game_records.Other_Sales) * 1000000)
Total_Other_Sales


# In[ ]:


video_game_records.describe()


# From the above Dateframe, North America sales Standard Deviation is the highest, which highlight how the sales is far spread out than the sales in Europe, Japan and the the rest of the world. In contrast, Sales in the rest of the world has the lowest Standard Deviation, which shows the sales are closer to each other.
# 
# 

# In[ ]:


Genres_total_games = video_game_records.groupby("Genre").size()


# In[ ]:


Genres_total_games 


# In[ ]:


Genres_total_games.plot.pie(autopct="%0.0f%%",  radius=2)
plt.show()


# The pie chart above shows Action and Sports video games are the most popular with 20% and 14% respectively. Strategy and Puzzle games seems not to interest people, with both being the least at 4% each.

# In[ ]:


# AFTER FILTERING THE DATAFRAME ABOVE USING MYSQL

#Genre - Genre of the game

#NORTH_AMERICA SALES - Sales in North America (in millions)

#EUROPE_SALES - Sales in Europe (in millions)

#JAPAN_SALES - Sales in Japan (in millions)

#OTHER_COUNTRY_SALES - Sales in the rest of the world (in millions)

#GLOBAL_SALES - Total worldwide sales.


# In[ ]:


video_game_data = pd.read_csv("../input/video-game-sales/Video_game_sales.csv")
video_game_data


# In[ ]:


# GLOBAL SALES PERCENTAGE AMONG GENRES


plt.pie(video_game_data.GLOBAL_SALES, labels = video_game_data.Genre, radius = 2, autopct = "%0.0f%%")
plt.show()


# In[ ]:


#VISUAL REPRESENTATION OF THE TOTAL SALES AMONG FIVE VARIABLES OF DIFERENT GENRE
    



plt.figure(figsize = (15, 10))
plt.plot(video_game_data.Genre, video_game_data.NORTH_AMERICA_SALES, marker = "o")
plt.plot(video_game_data.Genre, video_game_data.EUROPE_SALES, marker = "*")
plt.plot(video_game_data.Genre, video_game_data.JAPAN_SALES, marker = "s")
plt.plot(video_game_data.Genre, video_game_data.OTHER_COUNTRIES_SALES, marker = "v")
plt.plot(video_game_data.Genre, video_game_data.GLOBAL_SALES, marker = "P")
plt.xlabel("Genre Type")
plt.ylabel("Sales(Millions) ")
plt.title("VIDEO GAME SALES TREND")
plt.legend(["NORTH AMERICA","EUROPE","JAPan","OTHER","GLOBAL"])
plt.show()


# In[ ]:




