#!/usr/bin/env python
# coding: utf-8

# Hypothesis: Some turf is safer than other turf.
# 
# Disclaimer: All analysis here is by correlation and does not imply causality. 
# Furthermore the dataset is too small for statistically significant results at the typical alpha levels used in scientific literature. I have no direct relationship with UBU Speed Series or DD GrassMaster.
# 
# Suggested rule change: Change the rules such that turf is standardized on all stadiums to a single type of artificial turf.
# We suggest using either UBU Speed Series or DD GrassMaster or some hybrid of these.
# We believe standardizing artificial turf will reduce variability relative to natural grass given rain or snow. 
# Further testing of this hypothesis should be done with future and past datasets not present in this analysis. 

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
from IPython.display import Image

print('finished loading libraries')


# In[ ]:


# GLOBAL variables
PATH_PREFIX = "../input/NFL-Punt-Analytics-Competition/"


# In[ ]:


# Load all relevant data
ground_truth_df = pd.read_csv(PATH_PREFIX+"video_review.csv", 
                              usecols=["PlayID","GameKey","Primary_Impact_Type"])

play_df = pd.read_csv(PATH_PREFIX+"play_information.csv",
                     usecols=["PlayID","GameKey","PlayDescription"])


game_df = pd.read_csv(PATH_PREFIX+"game_data.csv",
                     usecols = ["GameKey","Turf","GameWeather","Stadium"])
print("loaded all relevant CSVs")


# In[ ]:


# Create concussion feature
train_df = pd.merge(play_df, ground_truth_df, how="left", on=["PlayID","GameKey"])
train_df["is_concussion"] = 1
train_df.loc[train_df["Primary_Impact_Type"].isnull(),"is_concussion"] = 0
print(train_df["is_concussion"].value_counts())


# In[ ]:


# Create turf feature
train_df = pd.merge(train_df, game_df, how="left", on=["GameKey"])

# Notice there are spelling errors in the Turf names
print(train_df["Turf"].unique())

# The null values all correspond to Hard Rock Stadium which is grass based.
train_df[train_df["Turf"].isnull()]


# In[ ]:


# Create the grass feature
# GrassMaster is part artificial https://en.wikipedia.org/wiki/GrassMaster
train_df["is_Natural_Grass_flag"] = 0
train_df.loc[train_df["Turf"]=='Grass', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Natural Grass', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Natural grass', "is_Natural_Grass_flag"] = 1

train_df.loc[train_df["Turf"]=='grass', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Natural Grass ', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Natural', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Natrual Grass', "is_Natural_Grass_flag"] = 1
train_df.loc[train_df["Turf"]=='Naturall Grass', "is_Natural_Grass_flag"] = 1

#https://www.lawnstarter.com/blog/sports-turf/nfl-stadiums-turf-or-grass/
# Hard Rock Stadium is grass
train_df.loc[train_df["Stadium"]=='Hard Rock Stadium', "is_Natural_Grass_flag"] = 1


turf_list = ['Grass' 'Natural Grass' 'DD GrassMaster' 'Natural grass' 'A-Turf Titan'
 'FieldTurf' 'UBU Speed Series-S5-M' 'UBU Sports Speed S5-M' 'Field Turf'
 'Artificial' 'Synthetic' 'grass' 'Natural Grass ' 'Natural',
 'Natrual Grass' 'FieldTurf 360' 'UBU Speed Series S5-M' 'Artifical'
 'FieldTurf360' 'Naturall Grass' 'Field turf']


# "The genetically-enhanced Platinum TE Paspalum turf installed this week was the result of nearly two years of research into determining the most workable grass for a safe and durable field for football that will hold up to the impediments imposed by the new shade canopy covering the stands."
# Source: https://www.sun-sentinel.com/sports/miami-dolphins/fl-dolphins-stadium-turf-0805-20160804-story.html

# In[ ]:


# Hard Rock Stadium appears to be grass
Image("../input/hard-rock-turf/i_will.jpg")

# Sources:
# https://www.sbnation.com/2017/11/5/16610770/dolphins-raiders-field-conditions-virginia-tech-miami-hard-rock-stadium


# In[ ]:


print(train_df.groupby(["is_Natural_Grass_flag"])["is_concussion"].describe())

# There is a lower rate of concussion for artificial grass 
# Natural grass has a concussion probability of 0.006064 which is higher than 0.004848 for artificial


# In[ ]:


# Does this hold up to specific types of artificial turf?

train_df["is_FieldTurf_flag"] = 0
train_df.loc[train_df["Turf"]=='FieldTurf', "is_FieldTurf_flag"] = 1
train_df.loc[train_df["Turf"]=='Field Turf', "is_FieldTurf_flag"] = 1
train_df.loc[train_df["Turf"]=='FieldTurf 360', "is_FieldTurf_flag"] = 1
train_df.loc[train_df["Turf"]=='FieldTurf360', "is_FieldTurf_flag"] = 1
train_df.loc[train_df["Turf"]=='Field turf', "is_FieldTurf_flag"] = 1
print(train_df.groupby(["is_FieldTurf_flag"])["is_concussion"].describe())


# FieldTurf is about the same as non-FieldTurf


# In[ ]:


train_df["is_Titan_flag"] = 0
train_df.loc[train_df["Turf"]=='A-Turf Titan', "is_Titan_flag"] = 1
print(train_df.groupby(["is_Titan_flag"])["is_concussion"].describe())
# Titan is about the same as non-Titan


# In[ ]:


train_df["is_DD_flag"] = 0
train_df.loc[train_df["Turf"]=='DD GrassMaster', "is_DD_flag"] = 1
print(train_df.groupby(["is_DD_flag"])["is_concussion"].describe())
# DD GrassMaster has no concussions but only 201 observations out of 6681


# In[ ]:


# We suggest using either UBU or DD GrassMaster
train_df["is_UBU_or_DD_flag"] = 0
train_df.loc[train_df["Turf"]=='UBU Speed Series-S5-M', "is_UBU_or_DD_flag"] = 1
train_df.loc[train_df["Turf"]=='UBU Sports Speed S5-M', "is_UBU_or_DD_flag"] = 1
train_df.loc[train_df["Turf"]=='UBU Speed Series S5-M', "is_UBU_or_DD_flag"] = 1
train_df.loc[train_df["Turf"]=='DD GrassMaster', "is_UBU_or_DD_flag"] = 1
print(train_df.groupby(["is_UBU_or_DD_flag"])["is_concussion"].describe())


# In[ ]:


# UBU Speed Series Turf Picture
Image("../input/ubu-turf/ubu-artificial-turf-speed-series-kiefer-usa.png")

# Sources:
# https://www.reddit.com/r/CFB/comments/63v5ne/lets_look_at_some_turf/


# In[ ]:


# Preprocess data for bootstrap

UBU_DD_df = train_df[train_df["is_UBU_or_DD_flag"]==1]
not_UBU_DD_df = train_df[train_df["is_UBU_or_DD_flag"]==0]


# In[ ]:


# Create bootstrap empirical intervals for is_UBU_or_DD_flag feature
np.random.seed(seed=0)
N_ITERATIONS = 30000

UBU_DD_mean_list = []
not_UBU_DD_mean_list = []
for j in range(N_ITERATIONS):
    UBU_DD_mean_list.append(UBU_DD_df["is_concussion"].sample(frac=1, replace=True).mean())
    not_UBU_DD_mean_list.append(not_UBU_DD_df["is_concussion"].sample(frac=1, replace=True).mean())
    
print('done')


# In[ ]:


bootstrap_df = pd.DataFrame({'UBU_DD_concussion_prob':UBU_DD_mean_list, 
                             'not_UBU_DD_concussion_prob':not_UBU_DD_mean_list})
print(bootstrap_df.describe())


# In[ ]:


# Bootstrap distribution comparison

pyplot.hist(bootstrap_df['UBU_DD_concussion_prob'], alpha=0.4, label='UBU_DD_concussion_prob')
pyplot.hist(bootstrap_df['not_UBU_DD_concussion_prob'], alpha=0.4, label='not_UBU_DD_concussion_prob')
pyplot.legend(loc='upper right')
pyplot.show()


# In[ ]:


LOWER_VAL = 0.15
UPPER_VAL = 0.85
confidence_level = 1-LOWER_VAL - (1-UPPER_VAL)
print(confidence_level)


# In[ ]:




lowerbound_UBU_DD = bootstrap_df['UBU_DD_concussion_prob'].quantile(LOWER_VAL)
upperbound_UBU_DD = bootstrap_df['UBU_DD_concussion_prob'].quantile(UPPER_VAL)
print(lowerbound_UBU_DD)
print(upperbound_UBU_DD)


# In[ ]:


lowerbound_not_UBU_DD = bootstrap_df['not_UBU_DD_concussion_prob'].quantile(LOWER_VAL)
upperbound_not_UBU_DD = bootstrap_df['not_UBU_DD_concussion_prob'].quantile(UPPER_VAL)
print(lowerbound_not_UBU_DD)
print(upperbound_not_UBU_DD)


# **I am 70% confident that using UBU or DD artificial turf is correlated with lower concussion rates**
