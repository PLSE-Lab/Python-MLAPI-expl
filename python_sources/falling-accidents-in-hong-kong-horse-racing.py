#!/usr/bin/env python
# coding: utf-8

# # Falling Accidents in Hong Kong Horse Racing
# 
# Horse racing is one of the most popular sports/betting activities in Hong Kong. Instead of looking at the betting side of horse racing, this analysis looks into a less-explored but potentially dangerous aspect: jockeys falling off from horses during the race.
# 
# * <a href='part0'>Initial Exploration of Data</a>
# * <a href='part1'>Part 1: Course, Distance and Class</a>
# * <a href='part2'>Part 2: Track Condition</a>
# * <a href='part3'>Part 3: Track Used (width)</a>
# * <a href='part4'>Part 4: Number of Horses in the Race</a>
# * <a href='part5'>Part 5: When did the accidents happen?</a>
# * <a href='part6'>Part 6: Where did the accidents happen?</a>

# ## <a id='Part0'>Initial Exploration of Data</a>

# In[ ]:


# Loading packages and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
plt.style.use('seaborn-colorblind')
horse = pd.read_csv('../input/race-result-horse.csv')
race = pd.read_csv('../input/race-result-race.csv')


# In[ ]:


horse.head()


# In[ ]:


race.head()


# In[ ]:


print (horse.shape, race.shape)


# There are 2367 races in the dataset with a total of 30189 runners.

# In[ ]:


print ('The dataset covers races from ' + race.race_date.min() + ' to ' + race.race_date.max())


# Let's look into how the races are distributed:

# In[ ]:


plt.figure(figsize=(12,9))

plt.subplot(221)
temp = race.race_course.value_counts(ascending=True)
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.5, i, str(v), fontsize=12)
plt.title('Race Course')

plt.subplot(222)
temp = race.race_distance.value_counts(ascending=True)
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.3, i-0.05, str(v), fontsize=12)
plt.title('Race Distance (m)')

plt.subplot(223)
temp = race.track.value_counts(ascending=True)
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.5, i, str(v), fontsize=12)
plt.title('Track')

plt.subplot(224)
temp = race.track_condition.value_counts(ascending=True)
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.3, i-0.05, str(v), fontsize=12)
plt.title('Track Condition')

plt.show()


# Let's look at some combination of variables:

# In[ ]:


pd.crosstab(race.race_course, race.track)


# In[ ]:


pd.crosstab(race.race_course, race.track_condition)


# In[ ]:


pd.crosstab(race.track, race.track_condition)


# Some points to note according to [information from the Hong Kong Jockey Club](https://racing.hkjc.com/racing/english/racing-info/racing_course.asp):
# * Sha Tin has both turf and all weather (or dirt) courses; Happy Valley only has turf course;
# * For a turf course, "A" course is the widest, "B" is narrower, which in turn wider than "C". The width of "A+3" is between "A" and "B"; "B+2" between "B" and "C", and "C+3" is the narrowest of all
# * Track condition is specific to track, where (1) Turf track has conditions "GOOD TO FIRM", "GOOD", "GOOD TO YIELDING", "YIELDING", "YIELDING TO SOFT", and "SOFT" (there should be "FAST" and "HEAVY" according to HKJC but none of them occurred in the 3 seasons in the dataset; (2) All weather track has "FAST", "WET FAST", "GOOD" and "WET SLOW", with some other indicators without occurrence in the dataset concerned.
# 
# For analysis in the following sections, we add a variable "course_type" to classify into 3 types, namely **"Sha Tin turf", "Happy Valley turf" and "Sha Tin all weather"**:

# In[ ]:


race.loc[race.race_course=='Happy Valley','course_type'] = 'Happy Valley turf'
race.loc[race.track=='ALL WEATHER TRACK','course_type'] = 'Sha Tin all weather'
race.loc[(race.race_course=='Sha Tin') & (race.track!='ALL WEATHER TRACK'),'course_type'] = 'Sha Tin turf'
race.course_type.value_counts()


# In[ ]:


pd.crosstab(race.race_distance, race.course_type)


# The most common races are Sha Tin turf 1400m (399 races), followed by Happy Valley turf 1200m (343) and Sha Tin turf 1200m (321). Sha Tin all weather only has 1200m, 1650m and 1800m races.
# 
# Then we look at race class:

# In[ ]:


plt.figure(figsize=(8,6))
temp = race.race_class.value_counts(ascending=True)
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.3, i, str(v), fontsize=12)
plt.title('Race Class')
plt.show()


# For those not familiar with Hong Kong racing system, races in Hong Kong are separated from Class 1 to Class 5, with Class 1 is the highest. Group races are races above Class 1 and include international races. Griffin races are for young horses before they are assigned classes.
# 
# The race class distribution needs some tidying. For class races with "Restricted" or "Special Condition", they are seen as races from respective class. For "Restricted Races", they are either "The Griffin Trophy" or the 2017 4-year-old series (including Hong Kong Derby) , so I will classify them together with the group races into **"Group and Restricted"**.

# In[ ]:


def classify(x):
    if x[0:6]=='Class ':
        return x[0:7]
    elif x.find('Group') != -1 or x=='Restricted Race':
        return 'Group and Restricted'
    else:
        return x
race['class_adj'] = race['race_class'].apply(lambda x: classify(x))

plt.figure(figsize=(7,5))
temp = race.class_adj.value_counts(ascending=True)
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.3, i, str(v), fontsize=12)
plt.title('Race Class (adjusted)')
plt.show()


# Class 4 and Class 3 races are the majority of races.
# 
# Then it's time to look at our theme, falling. The indicators of jockeys falling off are in **"finishing_position**" variable of "horse" table. 

# In[ ]:


horse.finishing_position.unique()


# One can refer to the [Special Incidents Index](https://racing.hkjc.com/racing/english/racing-info/special_race_index.asp) in [Results](https://racing.hkjc.com/racing/Info/meeting/Results/english/) section of HKJC website to understand the meaning of each code.
# 
# After checking with racing videos, it is found that the falling incidents are marked with **'FE'** or **'UR'** indicators.

# In[ ]:


fall = horse.loc[horse.finishing_position.isin(['FE','UR']),:]
print ('No of races with falling incidents: ' + str(fall.race_id.nunique()))
print ('No of horses involved with falling incidents: ' + str(fall.shape[0]))
fall_p = fall.race_id.nunique()*1.0/race.race_id.nunique()
print ('Proportion of races with falling indicents: '+ '{:.2%}'.format(fall_p))


# In some races more than one jockeys fell off from their horses:

# In[ ]:


fall.race_id.value_counts().head()


# There are 2 races which 3 jockeys fell and 2 other races with 2 jockeys.
# 
# A quick check of whether any horse has thrown off its jockey more than once. The answer is "no":

# In[ ]:


fall.horse_name.nunique()


# In[ ]:


race_w_fall = fall.race_id.unique().tolist()
race['has_fall'] = race['race_id'].isin(race_w_fall)*1
races_fallen = race.loc[race.has_fall==1,:]
races_fallen.head()


# ## <a id='Part1'>Part 1: Course, Distance and Class</a>
# 
# What are the distribution of races which involves jockey(s) falling? Which kind of races have higher probability with falling accidents? We first look at three factors, course, distance and class:

# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(121)
temp = races_fallen.course_type.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-2, i, str(v), fontsize=12)
plt.title('Accident Count by Race Course')

plt.subplot(122)
temp = (race.groupby('course_type').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Race Course')
plt.axvline(x=fall_p*100)

plt.show()


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(121)
temp = races_fallen.race_distance.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-1, i, str(v), fontsize=12)
plt.title('Accident Count by Race Distance')

plt.subplot(122)
temp = (race.groupby('race_distance').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Race Distance')
plt.axvline(x=fall_p*100)

plt.show()


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(121)
temp = races_fallen.class_adj.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-1, i, str(v), fontsize=12)
plt.title('Accident Count by Race Class')

plt.subplot(122)
temp = (race.groupby('class_adj').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Race Class')
plt.axvline(x=fall_p*100)

plt.show()


# Initial findings:
# - Sha Tin all weather has a slightly higher probability (1.74%) of having falling accidents than turf races;
# - 2.5% of 1400m races (vs overall average of 1.27%) involved jockey(s) fell off. No falling accident in all (213) 1600m races.
# - Class 2 races has the highest probability (1.8%) of falling accidents. No falling accident in Class 1, Group and Restricted, and griffin races;
# 
# (Note: While the dataset covers races for 3 seasons from 2014-2017, on November 25 2018 the most tragic fall happened in recent years happened over a [1600m race](https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2018/11/25&Racecourse=ST&RaceNo=9), in which jockey [Tye Angland fell and became quadriplegic](https://www.smh.com.au/sport/racing/tye-angland-confirmed-quadriplegic-as-result-of-hong-kong-fall-20190305-p511we.html) as a result)

# ## <a id='part2'>Part 2: Track Condition</a>
# 
# We now investigate whether wet and slow tracks more prone to falling. As track condition are specific to whether the course is turf or dirt (all weather), we separate those races accordingly:

# In[ ]:


plt.figure(figsize=(16,10))

plt.subplot(221)
temp = races_fallen[races_fallen.course_type!='Sha Tin all weather'].track_condition.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-1, i, str(v), fontsize=12)
plt.title('Accident Count by Track Condition (Turf)')

plt.subplot(222)
temp = (race[race.course_type!='Sha Tin all weather'].groupby('track_condition').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.02, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Track Condition (Turf)')
plt.axvline(x=fall_p*100)

plt.subplot(223)
temp = races_fallen[races_fallen.course_type=='Sha Tin all weather'].track_condition.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-0.5, i, str(v), fontsize=12)
plt.title('Accident Count by Track Condition (All Weather)')

plt.subplot(224)
temp = (race[race.course_type=='Sha Tin all weather'].groupby('track_condition').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.02, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Track Condition (All Weather)')
plt.axvline(x=fall_p*100)

plt.show()


# Initial findings:
# - For turf courses, **slower tracks have lower probabililty** of falling accidents. Among a total of 156 races on "good to yielding", "yielding", "yielding to soft" and "soft" grounds, only one has jockey fell.
# - For all weather courses, all 5 falling accidents happened on "Good" track.

# ## <a id='part3'>Part 3: Track Used (width)</a>
# 
# Do narrower tracks have more risk to jockeys falling? As all weather track has only one width, only turf races are considered in this analysis.

# In[ ]:


pd.crosstab(races_fallen.track, races_fallen.course_type)


# In[ ]:


plt.figure(figsize=(16,10))

plt.subplot(221)
temp = races_fallen[races_fallen.course_type=='Sha Tin turf'].track.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-0.5, i, str(v), fontsize=12)
plt.title('Accident Count by Track (Sha Tin Turf)')

plt.subplot(222)
temp = (race[race.course_type=='Sha Tin turf'].groupby('track').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.02, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Track (Sha Tin Turf)')
plt.axvline(x=fall_p*100)

plt.subplot(223)
temp = races_fallen[races_fallen.course_type=='Happy Valley turf'].track.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-0.3, i, str(v), fontsize=12)
plt.title('Accident Count by Track (Happy Valley Turf)')

plt.subplot(224)
temp = (race[race.course_type=='Happy Valley turf'].groupby('track').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.02, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Track (Happy Valley Turf)')
plt.axvline(x=fall_p*100)

plt.show()


# Initial findings:
# - For Sha Tin races, narrower tracks are more prone to falling. "C course" and "C+3 course" have probability of 2.48% and 1.99% respectively. Sha Tin "A+3 course" has no falling accident in all 200 races.
# - No significant pattern in Happy Valley races

# ## <a id='part4'>Part 4: Number of Horses in the Race</a>
# 
# Are falling accidents easier to happen when there are more horses in the race? To calculate the number of starters in a race, we need to exclude all horses that withdrew. It is indicated at the "finishing_position" variable of "horse" table by values started with "W"; Upon checking some entries are 'NaN' and are treated as non-starters:

# In[ ]:


horse['starter'] = horse.finishing_position.apply(lambda x: 1- (str(x)[0]=="W" or str(x)=='nan'))
in_race = horse.groupby('race_id').sum()['starter']
in_race = in_race.reset_index()
race = race.merge(in_race, how='left', on='race_id')
race.head()


# In[ ]:


pd.crosstab(race.starter, race.course_type, margins=True)


# Note that Happy Valley turf races have a maximum of 12 starters in any race.

# In[ ]:


races_fallen = race.loc[race.has_fall==1,:]

plt.figure(figsize=(14,5))
plt.subplot(121)
temp = races_fallen.starter.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v-1, i, str(v), fontsize=12)
plt.title('Accident Count by No. of Starters')

plt.subplot(122)
temp = (race.groupby('starter').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by No. of Starters')
plt.axvline(x=fall_p*100)

plt.show()


# No race (out of 168 races) with 10 or less starters has a falling accident.  
# Then we further break down by type of course:

# In[ ]:


pd.crosstab(races_fallen.starter, races_fallen.course_type, margins=True)


# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(131)
temp = (race[race.course_type=='Sha Tin turf'].groupby('starter').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.xlabel('% with Fall')
plt.title('Sha Tin Turf')
plt.axvline(x=fall_p*100)

plt.subplot(132)
temp = (race[race.course_type=='Happy Valley turf'].groupby('starter').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.xlabel('% with Fall')
plt.title('Happy Valley Turf')
plt.axvline(x=fall_p*100)

plt.subplot(133)
temp = (race[race.course_type=='Sha Tin all weather'].groupby('starter').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.xlabel('% with Fall')
plt.title('Sha Tin All Weather')
plt.axvline(x=fall_p*100)

plt.show()


# Initial findings:
# - All Sha Tin turf races with starters 11 or below had no falling accidents;
# - 3 of 76 Happy Valley turf races with 11 horses in the race had falling accidents;
# - 14.29% of Sha Tin all weather races with 13 starters had jockey(s) falling. While it is only 2 out of 14 races, **if we combine races with 13 and 14 starters, falling accidents happened in 4 out of 110 races**.
# 

# ## <a id='part5'>Part 5: When did the accidents happen?</a>
# 
# Here we examine whether falling accidents happen more frequently at the later races of a race day, where it is expected the jockeys may be more tired and more prone to accidents.
# 
# As a reference, day races usually have 10 (sometimes 11) races, and night races usually have 8 (sometimes 9).

# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(121)
temp = races_fallen.race_number.value_counts().sort_index()
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.1, i, str(v), fontsize=12)
plt.title('Accident Count by Race Number')

plt.subplot(122)
temp = (race.groupby('race_number').mean()['has_fall']*100).sort_index()
g = temp.plot(kind='barh', color='forestgreen')
for i, v in enumerate(temp):
    g.text(v+0.05, i, str(round(v,2)), fontsize=12)
plt.ylabel('')
plt.title('% with Fall by Race Number')
plt.axvline(x=fall_p*100)

plt.show()


# Falling could happen in race 1 to 11. Race 8 has the lowest probability of accident, only 1 in 256 races (maybe lucky number 8 for Chinese?).

# ## <a id='part6'>Part 6: Where did the accidents happen?</a>
# 
# For the 36 horses that threw off their jockeys, where did the accidents happen?
# 
# The "horse" table records running position at each section in columns "running_position_1" to "running_position_6". Sections are generally reported by HKJC at 400m intervals. In particular, the number of sections reported by each distance is as follows:
# - 1000 and 1200m: 3 sections
# - 1400, 1600 and 1650m: 4 sections
# - 1800 and 2000m: 5 sections
# - 2200 and 2400m: 6 sections
# 
# I will divide the section when the falling happen into four categories: **Last section ("Final 400"), the section before last ("Final 800"), first section ("At start") and anywhere else ("Others")**

# In[ ]:


fall = horse.loc[horse.finishing_position.isin(['FE','UR']),:]
fall = fall.merge(race.loc[:,['race_id','race_distance','course_type']], how='left', on='race_id')


# In[ ]:


dist_sec_map = {1000:3, 1200:3, 1400:4, 1600:4, 1650:4, 1800:5, 2000:5, 2200:6, 2400:6}

def get_fall_pos(temp):
    sections = dist_sec_map[temp['race_distance']]
    last = 'running_position_' + str(sections)
    bflast = 'running_position_' + str(sections-1)
    bf2last = 'running_position_' + str(sections-2)
    if np.isnan(temp['running_position_1']):
        return "At start"
    elif np.isnan(temp[last]) and ~np.isnan(temp[bflast]):
        return "Final 400"
    elif np.isnan(temp[bflast]) and ~np.isnan(temp[bf2last]):
        return "Final 800"
    else:
        return "Others"

for index, row in fall.iterrows():
    fall.loc[index, 'fall_pos'] = get_fall_pos(row)


# In[ ]:


plt.figure(figsize=(7,5))
temp = fall.fall_pos.value_counts(ascending=True)
g = temp.plot(kind='barh', color='darkseagreen')
for i, v in enumerate(temp):
    g.text(v+0.3, i, str(v), fontsize=12)
plt.title('Section Where the Falling Happened')
plt.show()


# Most of the accidents happened either at the start or the last 400m.
# 
# Thank you for reading! I hope we will have a more updated dataset (currently it is up to July 2017) to update my analysis accordingly.
