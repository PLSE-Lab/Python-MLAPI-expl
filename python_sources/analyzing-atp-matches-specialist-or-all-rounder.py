#!/usr/bin/env python
# coding: utf-8

# # **Analyzing ATP Matches**
# 
# Introduction

# Index
# 

#  ## Preparing the environment
#  
# We will load the packages we are going to use during the analysis, as well as our raw dataframe

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM


atp = pd.read_csv("../input/atp-matches/atp.csv", delimiter = ";", parse_dates= True)


# This dataframe contains every match played (ranging from challenger tournaments to grand slams) between 2000 and 2020 (until March). As we can see right away, there is a ton of information to be analyzed, so it is important to come up with a particular question before digging deeper into the dataframe.
# 

# In[ ]:


print(atp.head())


# We will try to answer the following question in this analysis: **Do tennis players typically grow towards being specialized in one type of surface (grass, for example), or do they become all-rounders?**
# 
# This question can be interpreted in several ways, and with different data to back it up, so we will concrete it even more: **Does winning matches on one type of surface correlate positively with winning matches on the rest of surfaces?**

#  ## Manipulating the dataframe
#  
# Now that we have a clear idea about what are we looking for, we know which kind data we need. Ideally, we should have:
# 
# - The amount of matches that every player has played
# - The amount of wins and loses for each player
# - And more importantly: the amount of matches, wins and loses per player **per surface**
# 
# In that regard, we will start streamlining our dataframe. We will create a "cleaning" function for getting rid the non-useful data and merging the useful one into the categories we thought about:
# 

# In[ ]:


def cleaning (variable, groupc, groupr):
    variable2= variable.groupby([groupc, groupr]).count().unstack(level=1)
    variable3= variable2.iloc[:,0:4]
    variable4= variable3.droplevel(level=0, axis=1)
    variable5 = variable4.fillna(0) 
    return variable5


# The three arguments for this function will be:
# - variable => the source dataframe
# - groupc => the column that will become the index of the new dataframe (player names)
# - groupr => the column(s) that will contain the data itself (i.e. number of matches won)
# 
# We will retrieve the data for both won and lost matches:

# In[ ]:


df_winners = cleaning(atp, "winner", "surface")


print(df_winners.head())


# In[ ]:


df_losers =  cleaning(atp, "loser", "surface")

print(df_losers.head())
    


# Before merging the two dataframes, notice how columns for the surface type are the same. Let's rename them first!
# 

# In[ ]:


df_winners.columns = ['carpet_w','clay_w','grass_w',
                     'hard_w']

df_losers.columns = ['carpet_l','clay_l', 'grass_l',
                     'hard_l']


# Now we can proceed with the merging.
# 
# We don't need to know about people who only have loses for this analysis, so we use **left join** to only keep players who have at least 1 win

# In[ ]:


df_full = pd.merge(df_winners, df_losers, left_index=True, right_index=True, how="left")  

print(df_full.head())


# Let's add some additional columns to find total matches,total wins and total loses

# In[ ]:


df_full["wins"] = df_full.iloc [:, 0:4].sum(axis=1)
df_full["loses"] = df_full.iloc [:, 4:8].sum(axis=1)
df_full["matches"] = df_full.iloc[:, 0:8].sum(axis=1)

print(df_full.head())


#  ## Approaching unexpected events
#  
# Remember to always keep checking your data during this phase. Through spyder IDE, I realized there are some duplicate values that have been not erased during the cleaning function, mostly because of typos on the raw dataframe. Let's see an example:

# In[ ]:


df_full.loc["Diez S.":"Dodig I."]


# Found it? Exactly, there is two rows referencing Novak Djokovic here. Probably the second one has an space(" ") at the end of the name, and that's why it bypassed the cleaning function we made.
# 
# Unfortunately, the raw data was not clean, and we need to go back on track to find similar cases (this step will be done on another notebook coming soon!).
# 
# But now, let's imagine that this analysis needs to be reported really soon. We have to move on, but what do we do?
# 
# We can assume, for example, that typos regarding the player names are unusual, and no typo should have been repeated more than 10 times. We could, then, drop all the rows of our dataframe where "wins" column is lower than 10. By doing this, we will also drop all the "non-typo" players who have less than 10 victories overall, but in hindsight, those players are not as useful for answering our question, as they have little information to give us.
# 
# Let's see the shape of the dataframe once we drop these values:

# In[ ]:


df_plusten = df_full[df_full["wins"] >10]

print(df_plusten.shape)


# Let's see what this dataframe has to offer!
# 

#  ## Distribution and descriptive statistics
#  
#  First of all, we need to take a look at the value distribution. We will use seaborn's distplot for that:
# 
# 

# In[ ]:


sns.distplot(df_plusten)


# We can clearly see that the distribution of the whole dataframe does not follow the normal curve. That's worrying. In this case, let's plot the rest of the values, this time using matplotlib's histograms:

# In[ ]:


plt.hist(df_plusten["carpet_w"], alpha = 0.4, color="grey")
plt.hist(df_plusten["clay_w"], alpha = 0.8, color="orange")
plt.hist(df_plusten["grass_w"], alpha = 0.7, color ="green")
plt.hist(df_plusten["hard_w"], alpha = 0.4, color="blue")

plt.show()


# Same story here. The distribution does not appear to be normal. We can dig deeper into it by using 3 different approaches:
# 
# - Comparing mean and median values
# - Obtaining the skewness of our distribution
# - Using normailty test, such as Kolmogorov-Smirnov's
# 
# We can easily compare the mean and median (as well as other statistics) here:

# In[ ]:


winsbycourt = df_plusten.loc[:,"carpet_w":"hard_w"]
print(winsbycourt.describe())


# In[ ]:


print(winsbycourt.median())


# We can see here how mean and median are quite dissimilar, mainly because outliers. Think of the big 3 (Federer, Nadal & Djokovic). They will probably have 75+ wins on every surface type, and that increases the mean value (but the median does not get affected). While outliers may not necessarily indicate the existence of a "not normal" distribution, they give us hints to investigate on the right tracks.
# 
# Also,interestingly enough, the median for carpet is 0. That means, at least, 50% of the players do not have ANY WIN on carpet. As many of you tennis fans know, carpet tournaments were discontinued a few years ago. Given the scarcity of data on this one, we will remove carpet from our analysis.

# In[ ]:


wins3surfaces = winsbycourt.drop(columns=["carpet_w"])

print(wins3surfaces.skew(axis = 0, skipna = True))


# Again, the skewness of the three surface types tell us that we are probably facing a non normal distribution. A positive value greater than 1 indicates that most of out players have won less matches than the average.

# In[ ]:


print(stats.kstest(wins3surfaces["clay_w"], "norm"))
print(stats.kstest(wins3surfaces["grass_w"], "norm"))
print(stats.kstest(wins3surfaces["hard_w"], "norm"))


# Sooo...yep! Distribution is definitely not normal. KS test retrieves us a p value. We can basically interpret it in two ways:
# 
# - greater than 0.05 --> the analyzed variable follows a normal distribution
# 
# - less than 0.05 --> the analyzed variable **does not follow** a normal distribution

# ## Inferential statistics and conclusion
# 
# Finally, we can find answers to our desired question : **Does winning matches on one type of surface correlate positively with winning matches on the rest of surfaces?**
# 
# We will use seaborn's heatmap to see how wins on one court do correlate with wins on the rest. As we mentioned before, our variables do not follow the normal distribution, so **we cannot use Pearson's Correlation in this case**. We will use Spearman's instead!

# In[ ]:


wins3surfaces = winsbycourt.loc[:,"clay_w":"hard_w"]

sns.heatmap(wins3surfaces.corr("spearman"),
            vmin=-1,cmap='coolwarm',
            annot=True);


# This is surprising. Let's sum up what we can see here:
# 
# - Hard court wins relate strongly with grass wins and decently with clay wins. 
# - Grass and clay do not mash up well together.
# - Hard + grass mantain us the stronger correlation
# 
# So, does winning on a surface type imply that a tennis player will be winning on the rest of surfaces? Can we theorize about players typically being all-rounders or specialists?
# 
# We don't have a definitive answer on that, but we can agree that having good results in hard court will allow a player to do good in grass (and probably in clay too!). One of the explanations can be the importance of serve, due to hard and grass courts being fast. A player with a good serve will sure find more sucess on those surfaces than other with a subpar serve.
# 
# 

# In[ ]:





# 
