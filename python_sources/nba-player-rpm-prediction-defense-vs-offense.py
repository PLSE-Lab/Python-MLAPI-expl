#!/usr/bin/env python
# coding: utf-8

# An exloration around how to improve the prediction of an NBA player's [RPM](https://cornerthreehoops.wordpress.com/2014/04/17/explaining-espns-real-plus-minus/) (Real Plus Minus) metric.  This Kernel was forked from [Noah Gift](https://www.kaggle.com/noahgift), and is based off of his [NBA Player Power Influence And Performance](https://www.kaggle.com/noahgift/nba-player-power-influence-and-performance/notebook) Kernel.
# 
# 
# **Contents:**
# * Assessing the data.
# * Honing in on a key analytical question & initial EDA.
# * Creating subsets of data for analysis.
# * Expanding EDA.
# * Conclusion.

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
sns.set_style("whitegrid")
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# **Assessing the Data:**
# 
# Let's bring in our intial data set, and take a brief look at the variables available.  This should help us formulate a key business question.

# In[ ]:


## This data set will be our starting point
player_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv");player_df.head()


# **Honing in on a Key Analytical Question:**
# 
# There is quite a bit of data here to sift through, and a great deal of analysis has already been completed on this specific data set.  Let's take a fresh approach.
# 
# There is a common saying in athletics, that *"Defense Wins Championships"*.  While that may be true, the argument that scoring a point, or a goal, is absolutely required to win any athletic competition.  So how can we justify this statement?  Let's take a stab at answering a key analytical question around this topic:
# 
# ***Is an NBA player's defensive performance better suited to predict that player's RPM, than their offensive performance?***
# 
# Let's find out . . . 

# Defense, and offense aside, Is a player's RPM correlated with their total minutes played?

# In[ ]:


plt.figure(figsize=(15,7))
sns.regplot(x="MP", y="WINS_RPM", data=player_df)


# Based off of the regression plot above, it would appear as though there is a correlation between a higher RPM, and more minutes being played.  However there is a tradeoff with an increase in the amount of time a player is on the court.  One could argue that if a player isn't doing so well when they're in the game (a large number of turnovers, missed shots, and fouls), their RPM would actually drop.
# 
# Let's assess the strength of this relationship:

# In[ ]:


minutes = smf.ols('WINS_RPM ~ MP', data=player_df).fit()
print(minutes.summary())


# While we have observed a weak positive correlation (Adjusted R-Squared = .481), overall minutes played may not be the best predictor of a player's RPM.*  Let's take one more look at this analysis, with a position overlay.
# 
# **Variable transformation could be used from here, to help enhance the linear regression model.*
# 
# Is it possible that a player's position *and* minutes played may provide some additional insight?

# In[ ]:


ax = sns.lmplot(x="MP", y="WINS_RPM", data=player_df, hue='POSITION', fit_reg=False, size=6, aspect=2, legend=False, scatter_kws={"s": 200})
ax.set(xlabel='Minutes Played', ylabel='RPM (Real Plus Minus)', title="Minutes Played vs RPM (Real Plus Minus) by Position: 2016-2017 Season")
plt.legend(loc='upper left', title='Position')


# Now that we have overlayed a player's position over their minutes played vs their RPM, we can see some very interesting findings:
# 
# There are no shooting guards in the upper echelon of player RPMs.
# Small forwards and point guards seem to be yielding the highest RPM per minutes played.
# There is only one power forward in the most elite group.
# 
# Have shooting guards and power forwards been marginalized, when it comes to contributing to a team's win?  It could appear so.
# 
# Let's reshift our focus to targeting specific deffensive and offensive statistics related to a player's RPM.

# **Creating Subsets of Data for Analysis:**

# In[ ]:


## Defense
player_df_def = player_df[["DRB","STL","BLK","WINS_RPM"]].copy();player_df_def.head()


# In[ ]:


## Offense
player_df_off = player_df[["eFG%","FT%","ORB","AST","POINTS","WINS_RPM"]].copy()
player_df_off.rename(columns={'eFG%': 'eFG','FT%':'FT'}, inplace=True)
player_df_off.head()


# **Expanding EDA:**

# Let's look at the correlation between defensive statistics:

# In[ ]:


player_df_def.corr()


# There does appear to be a stronger correlation between defensive rebounds and steals, and a player's RPM.  Now let's visualize this relationship.

# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(player_df_def.corr(), xticklabels=player_df_def.columns.values, yticklabels=player_df_def.columns.values, cmap="Reds")


# How about a covariance summary?

# In[ ]:


player_df_def.cov()


# Now let's built out a regression model, to identify the strength of this predictive model.

# In[ ]:


defense = smf.ols('WINS_RPM ~ DRB + STL', data=player_df_def).fit()
print(defense.summary())


# Both Independent variables are reflecting a p-value < 0, and we are showing a Adjusted R-Squared value of .61.  This would generally be considered a moderate positive correlation, and is a much better indicator of predicting a player's RPM, vs overall minutes played.*
# 
# **Blocks were removed from the model, given a p-value > .05.*

# Let's run some additional regression diagnostics.

# In[ ]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(defense, fig=fig)


# While we can still see a positive correlation between steals, defensive rebounds, and a player's RPM, is one variable more valuable than the other?

# In[ ]:


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(defense, fig=fig)


# When assessing the residual plots, there are clearly a set of values in each category that are having an effect on the relationship, but when looking for general [homoskedasticity](http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html) in the residual plots, steals appears to be a stronger predictor of a player's RPM.*
# 
# **Additional diagnostics can be run to further assess homoskedasticity, but for now we will assume that the variability of steals is fairly equal across the range of RPM values.*

# Let's run another set of regression diagnostics, to further assess the model.

# In[ ]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(defense, "STL", fig=fig)


# In[ ]:


sns.jointplot("STL", "WINS_RPM", data=player_df_def,size=10, ratio=3, color="r")


# When isolating steals as a sole predictor of a player's RPM, we can see that there is still a positive correlation.

# In[ ]:


steals = smf.ols('WINS_RPM ~ STL', data=player_df_def).fit()
print(steals.summary())


# However, when we limit the number of independent variables in our model to only steals, we can see that our Adjusted R-Squared value drops significantly, to a weak positive correlation.
# 
# It is now safe to assume that blocks and steals combined, serve to be a better prediction of a player's RPM, relative to their defensive performance. 

# ***How about the offense?  Let's run through the same analysis for a player's offensive performance.***

# Let's look at the correlation between offensive statistics:

# In[ ]:


player_df_off.corr()


# There does appear to be a stronger correlation between points and assists, and a player's RPM.  Now let's visualize this relationship.

# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(player_df_off.corr(), xticklabels=player_df_off.columns.values, yticklabels=player_df_off.columns.values, cmap="Greens")


# How about the covariance summary?

# In[ ]:


player_df_off.cov()


# Now let's built out a regression model, to identify the strength of this predictive model.

# In[ ]:


offense = smf.ols('WINS_RPM ~ eFG + ORB + AST + POINTS', data=player_df_off).fit()
print(offense.summary())


# All Independent variables are reflecting a p-value < 0, and we are showing a Adjusted R-Squared value of .65.  This would generally be considered a moderate positive correlation, and is a much better indicator of predicting a player's RPM, vs overall minutes played, and slightly stronger than our deffensive model.

# Let's run some additional diagnostics.

# In[ ]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(offense, fig=fig)


# While we can still see a positive correlation between each variable and a player's RPM, is one variable more valuable than the other?

# In[ ]:


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(offense, fig=fig)


# When assessing the residual plots, there are clearly a set of values in each category that are having an effect on the relationship, but when looking for general [homoskedasticity](http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html) in the residual plots, points appears to be a stronger predictor of a player's RPM.*
# 
# **Additional diagnostics can be run to further assess homoskedasticity, but for now we will assume that the variability of points is fairly equal across the range of RPM values.*

# Let's run another set of regression diagnostics, to further assess the model.

# In[ ]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(offense, "POINTS", fig=fig)


# In[ ]:


sns.jointplot("POINTS", "WINS_RPM", data=player_df_off,size=10, ratio=3, color="g")


# When isolating points as a sole predictor of a player's RPM, we can see that there is still a positive correlation.

# In[ ]:


eFGs = smf.ols('WINS_RPM ~ POINTS', data=player_df_off).fit()
print(eFGs.summary())


# However, when we limit the number of independent variables in our model to only points, we can see that our Adjusted R-Squared value drops significantly, to a moderate positive correlation.
# 
# After removing variables from the model one by one, It is now safe to assume that all offensive variables combined, serve to be a better prediction of a player's RPM, relative to their offensive performance. 

# Now let's add in all of our variables, and look at a more holistic predictive model.

# In[ ]:


## Final Variables
player_df_full = player_df[["PLAYER","STL","DRB","eFG%","ORB","AST","POINTS","WINS_RPM"]].copy()
player_df_full.rename(columns={'eFG%': 'eFG'}, inplace=True)
player_df_full.head()


# In[ ]:


combined = smf.ols('WINS_RPM ~ STL + DRB + eFG + AST + POINTS', data=player_df_full).fit()
print(combined.summary())


# The Adjusted R-Squared value in this model appears to be the strongest yet, reflecting a strong positive correlation.

# **Conclusion:**

# So what did we actually learn here?
# 
# We started with the baseline assumption that an individual RPM is a valid metric to assess a player's performance and contributions towards a team's victory.  When trying to understand what individual statistics help predict that RPM, we quickly passed on minutes played, and player positions.  As we began to leverage various EDA techniques, we quickly arrived a key analytical question that were aiming to solve: are defensive or offensive statistics a better predictor of a player's RPM?
# 
# We leveraged heatmaps, scatter plots, and regression modeling diagnostics to arrive at a conclusion that yields one point of view, but does warrant additional analysis.
# 
# ***In essense:***
# 
# Offensive statistics are generally a better predictor of a player's RPM, than a player's deffensive statistics.  However, a combined deffensive and offensive regression model may serve as a better means of predicting a player's RPM.
# 
# An ideal player:
# 
# * A player who plays a significant amount of minutes.
# * A point guard or small forward.
# * A player who's defensive skills are emphasized via deffensive rebounds and steals.
# * A player who's offensive skills are emphasized via an excellent effective field goal percentage, an increased frequency of assists, and a strong ability to score a lot of points.
# 
# Sounds fairly logical right?
# 
# Now you just need to find the money to pay them . . . :)
# 
