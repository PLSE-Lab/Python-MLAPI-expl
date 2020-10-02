#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# The Console wars. The epic battles fought (and still being fought) amongst the biggest and baddest players in the console gaming industry. This notebook will focus on the 7th Generation and the 8th Generation of the Console wars and the belligerents are as follows : 
# 
# **7th GENERATION** : Playstation 3 vs XBOX360 vs Nintendo Wii
# 
# **8th GENERATION** : Playstation 4 vs XBOXONE vs Nintendo WiiU
# 
# The aim is to run some visualisations on how some of the features in the dataset are correlated to one another as well as to provide some summary statistics and data analysis on the choice of genres and overall sales made by the different consoles to observe which one emerges with bragging rights. 

# In[ ]:


# Import the relevant libaries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's load the video game sales data as "video" and explore the first 5 rows 

# In[ ]:


video = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
video.head()


# In[ ]:


print(video.shape)


# # GENERAL ANALYSIS & DATA CLEANSING
# 
# Before we start delving into the visuals and stats for the 7th Gen and 8th Gen console wars, let's run some general statistics to get a feel for what we have. Let's start by checking if there exists any nulls in the dataframe by calling the method "isnull().any()" as such:

# In[ ]:


video.isnull().any()


# Well well, it seems that there are nulls indeed. So let's start by getting rid of these pesky nulls via the "dropna" call.

# In[ ]:


video = video.dropna(axis=0)


# In[ ]:


video


# By calling the dataframe method "info()", we also discover that the "User_Score" column which should be best represented numerically, contains strings. Therefore we'll also convert that column to string type

# In[ ]:


video.info()


# In[ ]:


video.Platform.unique()


# Per the above, one can see that there are quite a few different platforms in our dataset. For the scope and purpose of this notebook, there are more platforms than required (as we want only the 7th and 8th Generational consoles). Therefore we have to undertake some selective trimming of the data which we will get to in due time.

# ## Jointplots and Correlations of the numeric features
# 
# Now, lets plot some of the numeric features against each other to explore the relations between them  as well as getting accustomed to using Seaborn's plotting capabilities. I therefore first plot the "Critic_Score" and "User_Score" columns as a jointplot (which is good to observe how two different variables are distributed) first to see how they interact:

# In[ ]:


video['User_Score'] = pd.to_numeric(video['User_Score'])


# In[ ]:


sns.jointplot(x='Critic_Score',y='User_Score',data=video,kind='hex')
# video.plot(y= 'Critic_Score', x ='User_Score',kind='hexbin',gridsize=35, sharex=False, colormap='afmhot_r', title='Hexbin of Critic_Score and User_Score')


# As expected, these 2 scores exhibit quite positive Pearson Correlation with one another. This should not be a surprise as on average, if a game is good, both the critic and the user will derive enjoyment out of it and therefore tend to score the game higher and vice-versa. Let's now look at "Critic_Count" and "Critic_Score"

# In[ ]:


sns.jointplot('Critic_Score','Critic_Count',data=video,kind='hex')
#video.plot(y= 'Critic_Score', x ='Critic_Count',kind='hexbin',gridsize=40, sharex=False, colormap='cubehelix', title='Hexbin of Critic_Score and Critic_Count')


# And finally, to inspect the correlations between all the numeric features and see how one feeds into the other, I shall plot another swanky heatmap. First I extract all the numeric columns into a list and create a simple dataframe "video_num" for this heatmap plotting as such

# In[ ]:


video.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(7, 7))
plt.title('Pearson Correlation of Video Game Numerical Features')
# Draw the heatmap using seaborn
sns.heatmap(video.corr(), square=True, annot=True)


# Cool. With this heatmap, the darker colors represent more postive correlations and vice-versa. Therefore, we can already see quite logical connections like "Global_Sales" being very positively correlated to "EU_Sales" etc. Just some interesting things so far. 

# In[ ]:


video.select_dtypes(include=[np.number]).head()


# ___

# # 7th GENERATION CONSOLE WAR 
# 
# #Tale of the tape : PS3 vs XBOX360 vs Wii
# 
# Onto our first event of the evening, I'll provide some visualisations and summary statistics on the 7th Gen Console wars fought between the 3 main parties alluded to above. First, I will create a dataframe ("video7th") to contain only these 7th Gen consoles and then it's time to do some data poking and inspecting.

# In[ ]:


# Dataframe contain info only on the 7th Gen consoles
video7th = video[video.Platform.isin(['Wii', 'PS3', 'X360'])]
video7th.shape


# ### GLOBAL SALES OVER THE YEARS
# 
# First let's look at these console's global sales over the years and see if we can identify any which left with bragging rights.
# To do so, I shall aggregate the data via a "groupby" call on the Year_of_Release and "Platform" and then sum the Global_Sales. For visualisation, I will plot stacked barplots and hopefully this will be intuitive enough

# In[ ]:


yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()
yearlySales.unstack().plot(kind='bar')
plt.title('Stacked Barplot of Global Yearly Sales of the 7th Gen Consoles')
plt.ylabel('Global Sales')


# First Impressions : Seems the PS3 sales went from strength to strength , XB360 sales (bar a dip in 2009) also generally increased while the Wii sales, which had a strong headstart in the early years of 2006 and 2007 had it's lead eroded by the other 2.

# In[ ]:


yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()
yearlySales.unstack().head()


# ### SALES AGGREGATED BY VIDEO GAME RATINGS
# 
# Here, I will take a look at the different video game ratings (i think its E : Everyone, M: Mature, T: Teens) and look at how many sales each of the 3 consoles made

# In[ ]:


ratingSales = video7th.groupby(['Rating','Platform']).Global_Sales.sum()
ratingSales.unstack().plot(kind='bar')
plt.title('Stacked Barplot of Sales per Rating type of the 7th Gen Consoles')
plt.ylabel('Sales')


# First Impressions : Well not much surprise here as we know that the Wii primarily catered to family-oriented fun and therefore it made the largest sales at Rating E for Everyone while it sold negligible M for Mature games. On the other hand, the PS3 and XB360 sold the most M-rated games, something also pretty obvious from both their plethora of shooters, sandbox games and hacking/slashing games. Heck Yeahhhhh!!!

# ### SALES BY GENRE
# 
# Finally, let's drill down into the data even further and look at the sales made by the 3 consoles and look at what kind of Genre games defined each console and what differentiated one console from the other. To do, I will aggregate the data via a "groupby" call on the Genre feature and Platform. The resultant plot is as follows:

# In[ ]:


genreSales = video7th.groupby(['Genre','Platform']).Global_Sales.sum()
genreSales.unstack(level=0).plot(kind='bar', figsize=(16,9))
plt.title('Stacked Barplot of Sales per Game Genre')
plt.ylabel('Sales')


# First Impressions : It seems that for both the PS3 and XB360, their 2 main genres were Action and Shooter games which as we know was the case as they appeal more to the hardcore, action-oriented gamer. The Wii on the other hand, focused on the genre of Sports, Platformers as well as some other Misc games.

# ### TOTAL SALES AND TOTAL USERS
# 
# Finally let us look at pie chart visualisations of the total number of Global Sales and total number of users attributed to each of the 3 consoles. The way I am going to present this is to simply add up the Global sales and number of users value for all games. Therefore as a caveat, take the numbers and visualisation with a pinch of salt as this output will be dependent on whether the original dataset was fully inclusive in the first instance.

# In[ ]:


# Plotting our pie charts
plt.figure(figsize=(8,4))
plt.subplot(121)
sales_by_platform = video7th.groupby('Platform').Global_Sales.sum()
plt.pie(
    sales_by_platform,
    # with the labels being platform
    labels=sales_by_platform.index,
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%.1f%%'
)
plt.axis('equal')
plt.title('Global Sales')

plt.subplot(122)
user_count_by_platform = video7th.groupby('Platform').User_Count.sum()
plt.pie(
    user_count_by_platform,
    labels=user_count_by_platform.index,
    startangle=90,
    autopct='%.1f%%'
)
plt.axis('equal')
plt.title('User Base')
plt.tight_layout()
plt.show()


# **Concluding Remarks** 
# 
# From the pie charts above as well as the earlier barplots, it seems that both the PS3 and the XB360 were very evenly matched, with the XB360 having the slight edge in global sales. What is obvious is that from these metrics alone, the showing from the Wii could not compete against its other 2 competitors. 
# 
# # WINNER : PS3 & XB360 (Two-way Tie)

# ---

# # 8th GENERATION CONSOLE WAR 
# 
# ### Tale of the tape : PS4 vs XBOXONE vs WiiU
# 
# Onto our second event of the evening and as per the earlier sections,  I'll provide some visualisations and summary statistics on the 8th Gen Console wars. First up, I will create a dataframe ("video8th") to only contain data pertaining to these 3 particular consoles:

# In[ ]:


video8th = video[video.Platform.isin(['WiiU', 'PS4', 'XOne'])]
video8th.shape


# ### GLOBAL SALES OVER THE YEARS
# 
# Following our approach with the 7th Gen data, let's first have a high-level grasp of the sales performance of these 8th Gen games over the years. Therefore I shall once again aggregate the data via a "groupby" call on the Year_of_Release and "Platform" and then sum the Global_Sales with stacked barplots for visualisation.

# In[ ]:


yearlySales = video8th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()
yearlySales.unstack().plot(kind='bar',stacked=True, colormap= 'Blues',  grid=False)
plt.title('Stacked Barplot of Global Yearly Sales of the 8th Gen Consoles')
plt.ylabel('Global Sales')


# First Impression : It is obvious just by one look that the PS4 global sales exceed those of BOTH the WiiU and XOne combined. This is a very marked deviation from its predecessor's performance in the 7th Gen when the PS3 and XB360 where neck to neck in sales performance over the years. So how can be explain this dominance this time round?

# ### SALES AGGREGATED BY VIDEO GAME RATINGS
# 
# Well let's boil our analysis down even further to see if we can investigate this PS4 dominance. Let's look at what kind of audiences (hence looking at the Ratings) these consoles catered their games to

# In[ ]:


ratingSales = video8th.groupby(['Rating','Platform']).Global_Sales.sum()
ratingSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greens', grid=False)
plt.title('Stacked Barplot of Sales per Rating type of the 8th Gen Consoles')
plt.ylabel('Sales')


# First Impression : An Interesting result this time round. Unlike the 7th Gen where there was a clear demarcation in the sense that the PS3 and XB360 produced games primarily for the M for Mature audience while the Wii was for the E for Everyone audience, it seems that the PS4 has decided to cater ( or has somehow appealed more) to both the M and E audience. This could explain their earlier dominance in global sales as they are now taking up both the hardcore gaming audience as well as the casual, family-friendly audience.

# ### SALES BY GENRE
# 
# Finally, let's look at the breakdown by Genre via an aggregation the data with a "groupby" call on the Genre feature and Platform. The resultant plot is as follows:

# In[ ]:


genreSales = video8th.groupby(['Genre','Platform']).Global_Sales.sum()
genreSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greys', grid=False)
plt.title('Stacked Barplot of Sales per Game Genre')
plt.ylabel('Sales')


# First Impression : Again, this plot is very telling in the sense that the PS4 is clearly trying to gain an inroad into more genres than its predecessor, the PS3. Just from a quick visual glance, one can observe that the PS4 already has the majority of sales in 7 out of 12 of the genres ( 4 out of 12 for the PS3).

# ### TOTAL SALES AND TOTAL USERS

# In[ ]:


# Plotting our pie charts
# Create a list of colors 
colors = ['#008DB8','#00AAAA','#00C69C']
plt.subplot(121)
plt.pie(
   video8th.groupby('Platform').Global_Sales.sum(),
    # with the labels being platform
    labels=video8th.groupby('Platform').Global_Sales.sum().index,
    # with no shadows
    shadow=False,
    # stating our colors
    colors=colors,
    explode=(0.05, 0.05, 0.05),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%'
    )
plt.axis('equal')
plt.title('Pie Chart of 8th Gen Global Sales')
plt.subplot(122)
plt.pie(
   video8th.groupby('Platform').User_Count.sum(),
    labels=video8th.groupby('Platform').User_Count.sum().index,
    shadow=False,
    colors=colors,
    explode=(0.05, 0.05, 0.05),
    startangle=90,
    autopct='%1.1f%%'
    )
plt.axis('equal')
plt.title('Pie Chart of 8th Gen User Base')
plt.tight_layout()
plt.show()


# **Concluding Remarks** 
# 
# Unlike the 7th Gen consoles, we have a very clear leader for the 8th Gen consoles, and that would be the PS4, far outstripping the XBOXONE and the WiiU. Although this console war is far from over, it is undoubted that the PS4 has a very big headstart.
# 
# # WINNER : PS4 (so far)
