#!/usr/bin/env python
# coding: utf-8

# # Skillcraft
# 
# Starcraft is a Real Time Strategy (RTS) game developed by Blizzard that is played competitively.  There is a ranked system within the game that divides players into leagues (Bronze, Silver, Gold, Platinum, Diamond, Master and Grandmaster).  These have been indexed into 7 leagues from 1-7.  
# 
# The dataset can be obtained from here: https://www.kaggle.com/danofer/skillcraft
# 
# The goal of this notebook is to do an exploratory data analysis of the data as well as conduct some basic multinomial classification modelling on the data.

# ## Imports
# 
# Import useful libraries and the data itself.  

# In[ ]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn for naive-bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# random forest classifier
from sklearn.ensemble import RandomForestClassifier

# support vector machine
from sklearn import svm


# In[ ]:


starcraft = pd.read_csv("../input/SkillCraft.csv")
starcraft.head()


# ## Clean Up
# 
# Some checking to see if there exists any null data.  Clean up if there is.

# In[ ]:


# missing data check
starcraft.apply(lambda x: sum(x.isnull()), axis=0)


# ## Analysis
# 
# Visualizing and describing the dataset helps us to see if there are any features worth removing from the get-go before modelling itself.  It also helps to show if there is a correlation between the League a player is in and their stats.

# ### Summary Statistics
# 
# Basic overview of the data.

# In[ ]:


# dataset summary
starcraft.describe()


# ### Feature Plotting
# 
# Boxplots give us a good idea of the distribution of features within each league and how they compare between leagues.

# In[ ]:


# Age by League
sns.set(style = "whitegrid", rc = {"figure.figsize":(11.7, 8.27)})
ax = sns.boxplot(x = "LeagueIndex", y = "Age", data = starcraft).set_title("Age by League")


# Age has a fairly even distribution between leagues.

# In[ ]:


# HoursPerWeek by League
ax = sns.boxplot(x = "LeagueIndex", y = "HoursPerWeek", data = starcraft).set_title("HoursPerWeek by League")


# Unsurprisngly, playing the game more results in a better league.  More practice.

# In[ ]:


# TotalHours by League
ax = sns.boxplot(x = "LeagueIndex", y = "TotalHours", data = starcraft).set_title("TotalHours by League")


# Total hours played by everyone appears to be about the same.  Makes sense when viewed from the lens of "work smarter, not harder".

# In[ ]:


# APM per league
ax = sns.boxplot(x = "LeagueIndex", y = "APM", data = starcraft).set_title("APM by League")


# Higher APM leads to higher league placement.  

# In[ ]:


# SelectByHotkeys per league
ax = sns.boxplot(x = "LeagueIndex", y = "SelectByHotkeys", data = starcraft).set_title("SelectByHotkeys per League")


# Selecting more things by hotkeys means more efficient movement since constantly clikcing the buttons instead of using hotkeys makes a player slower.

# In[ ]:


# AssignToHotkeys per League
ax = sns.boxplot(x = "LeagueIndex", y = "AssignToHotkeys", data = starcraft).set_title("AssignToHotkeys by League")


# From starcraft guides, the "select all army" hotkey is most frequently used, leading to less hotkeys used overall.  More hotkeys, though, means better microing of your army.  Shows here as a higher leagues placement relative to number of hotkeys assigned.

# In[ ]:


# UniqueHotkeys by League
ax = sns.boxplot(x = "LeagueIndex", y = "UniqueHotkeys", data = starcraft).set_title("UniqueHotkeys by League")


# The mean rises while the interquartile ranges stay about the same.

# In[ ]:


# MinimapAttacks by League
ax = sns.boxplot(x = "LeagueIndex", y = "MinimapAttacks", data = starcraft).set_title("MinimapAttakcs by League")


# Attacking by clicking on the minimap seems to increase with league.  Likely due to allowing a player to do other things while assigning their army to attack somewhere.

# In[ ]:


# MinimapRightClicks by League
ax = sns.boxplot(x = "LeagueIndex", y = "MinimapRightClicks", data = starcraft).set_title("MinimapRightClicks by League")


# The right clicks allow for movement.  Again, more efficiency and multi-tasking allowed with this method.

# In[ ]:


# NumberOfPACs by League
ax = sns.boxplot(x = "LeagueIndex", y = "NumberOfPACs", data = starcraft).set_title("NumberOfPACs by League")


# Perception Action Cycle(PAC) or basically the ability to move the camera to an area, execute actions and then changes camera to another location.  Higher PACs means a shorter time to figure out what needs to be done in each area, leading to a higher league.

# In[ ]:


# GapBetweenPACs by League
ax = sns.boxplot(x = "LeagueIndex", y = "GapBetweenPACs", data = starcraft).set_title("GapBetweenPACs by League")


# Shorter gaps means less time spent thinking about what to do and more time spent doing.  Shorter gap leads to a higher league.

# In[ ]:


# ActionLatency by League
ax = sns.boxplot(x = "LeagueIndex", y = "ActionLatency", data = starcraft).set_title("ActionLatency by League")


# The time required between moving camera to an area and then initiating an action.  Lower Action Latency likely means better understanding of the game/better muscle memory which leads to a higher league.

# In[ ]:


# ActionsInPAC by League
ax = sns.boxplot(x = "LeagueIndex", y = "ActionsInPAC", data = starcraft).set_title("ActionsInPAC by League")


# Not much correlation with higher league likely due to it being more important to take less time to execute actions versus more actions on each screen.  

# In[ ]:


# TotalMapExplored by League
ax = sns.boxplot(x = "LeagueIndex", y = "TotalMapExplored", data = starcraft).set_title("TotalMapExplored by League")


# More map explored leads to a higher league.  This is likely due to greater emphasis on scouting as the league increases.  If you don't know what your opponent is doing, then you won't be able to counter what they're doing.

# In[ ]:


# WorkersMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "WorkersMade", data = starcraft).set_title("WorkersMade by League")


# Consistent worker production leads to a higher league placement.  More workers means more mining and building so that makes pretty good sense.

# In[ ]:


# UniqueUnitsMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "UniqueUnitsMade", data = starcraft).set_title("UniqueUnitsMade by League")


# Not much correlation here.  There are only so many different units you can build for each race so there won't be a huge difference.

# In[ ]:


# ComplexUnitsMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "ComplexUnitsMade", data = starcraft).set_title("ComplexUnitsMade by League")


# Complex units include ghosts, infestors and high templar.  Units that require greater skill than just A-clicking the ground with them given their abilities.  On the other hand, there's a greater range as the league increases but the mean number built stays at roughly the same, around 0.

# In[ ]:


# ComplexAbilitiesUsed by League
ax = sns.boxplot(x = "LeagueIndex", y = "ComplexAbilitiesUsed", data = starcraft).set_title("ComplexAbilitiesUsed by League")


# Also not a huge correlation.  Complex actions require extra input (like specific targetting) so the amount used increases with league, but not by a huge amount.

# ## Modelling
# 
# We can attempt to model the dataset.  Since there are 7 classes for targets, we are looking at a multinomial classification problem and can select our models accordingly.

# ### Naive Bayes Classification
# 
# A simple model for multinomial classification here.  We drop some columns as our analysis indicates that they will not have a suitable impact on the result.

# In[ ]:


# feature cleanup
drops = ["GameID", "Age", "TotalHours", "UniqueUnitsMade", "ComplexUnitsMade", "ComplexAbilitiesUsed"]

starcraft.drop(drops, axis = 1, inplace = True)
starcraft.head()


# We need to split the data into training and test sets to train and test the model respectively.

# In[ ]:


# split into training and test sets
y = starcraft.LeagueIndex
X_train, X_test, y_train, y_test = train_test_split(starcraft, y, test_size = 0.2)

# remove the target from the training data
X_train.drop("LeagueIndex", axis = 1, inplace = True)
X_test.drop("LeagueIndex", axis = 1, inplace = True)

# easy to read statements
print("X_train: ", X_train.shape) 
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape) 
print("y_test: ", y_test.shape)


# And we can finally build the naive-bayes model.

# In[ ]:


# fit the model
clf = MultinomialNB()

# train the model
model_1 = clf.fit(X_train, y_train)


# In[ ]:


# predictions
predictions_1 = clf.predict(X_test)

# view predictions
predictions_1[:10]


# In[ ]:


# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_1), 2))


# ### Random Forest Classification
# 
# We can borrow the train and test data splits from the Naive Bayes Classification to run a random forest classifier.

# In[ ]:


# build the model
clf2 = RandomForestClassifier(n_estimators = 64, random_state = 123)

# fit the model
model_2 = clf2.fit(X_train, y_train)


# In[ ]:


# predictions
predictions_2 = clf2.predict(X_test)

# view them
predictions_2[:10]


# In[ ]:


# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_2), 2))


# ### Support Vector Machine

# In[ ]:


# build the model
clf3 = svm.SVC(gamma = 0.00001, decision_function_shape = "ovr")

# fit the model
model_3 = clf3.fit(X_train, y_train)


# In[ ]:


# predictions
predictions_3 = clf3.predict(X_test)

predictions_3[:10]


# In[ ]:


# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_3), 2))


# ## Conclusions
# 
# The models explored here are fairly simple, especially considering the lack of feature engineering (outside of removing some columns) to make the features better explain variance in the data.  For better classification, more feature engineering is likely required as well as hyperparameter tuning to attain a better classification rate.
