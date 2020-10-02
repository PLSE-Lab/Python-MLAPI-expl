#!/usr/bin/env python
# coding: utf-8

# [The murder of George Floyd](https://en.wikipedia.org/wiki/George_Floyd) by the hands of 4 policemen in the streets of Minneapolis sparked protests across the USA and the world. The horrifying sequence of events and the cold brutality of those images showed an aspect of the life of many African Americans that was denied or ignored for far too long by far too many people.
# 
# As a white person, I think this is a moment of stay silent, listen, understand, and act upon that understanding without occupying the public conversation as we often tend to do. However, after weeks of protests, a question remains unanswered: why was this crime so powerful to shake the western world?
# 
# The more I think about it, the more I come to the horrifying realization that it is because this crime was filmed almost in its entirety, we can't hide behind any ambiguity from the terrifying reality.
# 
# The questions then are: what would have happened if there was no film? Can we hide behind excuses and ignore a systemic problem then?
# 
# Therefore, this notebook will show the reality of Minneapolis (a city taken as an example but by no means an isolated case) that comes from a potentially biased source (the police) and analyzed by me, a definitely biased person. I will try to keep my beliefs in the background and let the number speak but it is undeniable by this premise and the type of questions I will ask in my analysis where I stand. It thus makes no sense to try saying that I want to be impartial, it is not the moment to be impartial and I want to be clear: I want to use police reports to show that we were in denial if we had to wait for a broadcasted murder to acknowledge the discrimination and the brutality of the police.
# 
# Let's start by looking at the population distribution of Minneapolis. Thanks to [Paul Mooney](https://www.kaggle.com/paultimothymooney) and [his notebook](https://www.kaggle.com/paultimothymooney/minneapolis-police-interactions-by-race) we know that the 2010 census shows the following (we simplify the distribution of the races as we will focus primarily on the situation of Black people)

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
pd.set_option('max_columns', 100)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 5), facecolor='#f7f7f7')

wikipedia_2010_census = {'White': [63.8], 
        'Black': [18.6], 
        'Other': [23.7]}
wikipedia_2010_census_df = pd.DataFrame(wikipedia_2010_census).transpose()

wikipedia_2010_census_df.plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
ax.legend().set_visible(False)
ax.set_title('Percentage of population in Minneapolis by race', fontsize=18)
ax.grid(axis='y')

plt.show()


# In other words, Black people are **18.6% of the population in Minneapolis**, all the other minorities combined are 23.7% of the population.
# 
# # Police stops
# 
# The data show all the police stops in Minneapolis since 2016, providing useful information like race and gender of the subject and reason for the police action.

# In[ ]:


sd = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_stop_data.csv', low_memory=False)

sd['responseDate'] = pd.to_datetime(sd['responseDate'])
sd['year'] = sd.responseDate.dt.year
sd['time'] = sd.responseDate.dt.time
sd['hour'] = sd.responseDate.dt.hour
sd['reason'] = sd['reason'].fillna('Unknown')
sd['citationIssued'] = sd['citationIssued'].fillna('Unknown')

sd.head()


# We can immediately see how the police stops involved mostly black people and mostly males. This is a consistent pattern since 2016.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12, 5), facecolor='#f7f7f7', sharey=True)
fig.suptitle('Comparison between race distribution in the population and in police stops', fontsize=18)

tmp = sd[sd.race != 'Unknown'].copy()
tmp.loc[~tmp.race.isin(['Black', 'White']), 'race'] = 'Other'

(tmp.race.value_counts() / tmp.shape[0] * 100).plot(kind='bar', color= ['r', 'g', 'b'], ax=ax[1])
pop = pd.Series({'Black': 18.6, 'Other': 23.7, 'White': 63.8})

pop.plot(kind='bar', ax=ax[0], color= ['r', 'g', 'b'])

ax[0].set_title('Population', fontsize=14)
ax[1].set_title('Police stops', fontsize=14)
ax[0].set_ylabel('% of population')

for axes in ax:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
    axes.grid(axis='y')

plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 10), facecolor='#f7f7f7') 
fig.subplots_adjust(top=0.92)
fig.suptitle('Police stops by race and gender', fontsize=18)

gs = GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1:])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1, 2])

tmp = sd.copy()
tmp.loc[tmp.race == 'Native American', 'race'] = 'Native'
tmp.race.value_counts().plot(kind='bar', ax=ax0)
tmp = sd[sd.race != 'Unknown'].copy()
tmp.loc[~tmp.race.isin(['Black', 'White']), 'race'] = 'Other'
tmp.groupby(['race', 'year'], as_index=False).size().unstack(0).plot(ax=ax1, color=['r', 'g', 'b'])
tmp = sd.copy()
tmp.loc[tmp.gender == 'Gender Non-Conforming', 'gender'] = 'Unknown'
tmp.groupby(['gender', 'year']).size().unstack(0).plot(ax=ax2, color=['m', 'tab:orange', 'k'])
tmp.gender.value_counts().plot(kind='bar', ax=ax3)

ax0.set_xticklabels(ax0.get_xticklabels(), rotation=30)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

ax0.set_title('Number of stops by race')
ax1.set_title('Number of stops by race (2016-2020)')
ax3.set_title('Number of stops by gender')
ax2.set_title('Number of stops by gender (2016-2020)')


plt.show()


# This plot also shows that we are grouping all the other races into one generic one exclusively to keep the analysis simple.
# 
# Among what the data call problems triggering the police intervention, the majority of them falls in one of the following categories
# * Traffic Law Enforcement
# * Suspicious Person
# * Suspicious Vehicle
# 
# If we break down these stops by race we obtain

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(15, 6), facecolor='#f7f7f7')
fig.suptitle('Problems vs race', fontsize=18)

tmp = sd[sd.race != 'Unknown'].copy()
tmp.loc[~tmp.race.isin(['Black', 'White']), 'race'] = 'Other'

tmp[tmp.problem == 'Traffic Law Enforcement (P)'].race.value_counts().plot(kind='bar', ax=ax[0], color=['r', 'b', 'g'])
tmp[tmp.problem == 'Suspicious Person (P)'].race.value_counts().plot(kind='bar', ax=ax[1], color=['g', 'r', 'b'])
tmp[tmp.problem == 'Suspicious Vehicle (P)'].race.value_counts().plot(kind='bar', ax=ax[2], color=['g', 'r', 'b'])

for axes in ax:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
    
ax[0].set_title('Traffic Law Enforcement', fontsize=14)
ax[1].set_title('Suspicious Person', fontsize=14)
ax[2].set_title('Suspicious Vehicle', fontsize=14)

plt.show()


# Let's put things into perspective
# 
# * Black people are 18.6% of the population but are 40.9% of the stops for Traffic Law Enforcement, 36% of the stops of a Suspicious Person and Suspicious Vehicle
# * White people are 63.8% of the population but are 31% of the stops for Traffic Law Enforcement, 22% of the stops of a Suspicious Person, and 20% of a Suspicious Vehicle
# * All other races are 23.7% of the population but are 27% of the stops for Traffic Law Enforcement, 41% of the stops of a Suspicious Person, and 42% of a Suspicious Vehicle
# 
# Taking into account the population distribution, we see then that **non-white people are suspicious 6 times more often than white people**.
# 
# The top reasons for the police to intervene are
# 
# * Moving violation
# * 911 call
# * Investigative reasons
# * Equipment violation
# 
# We can again see a disparity in how the reasons are distributed across the races

# In[ ]:


fig, ax = plt.subplots(1, 4, figsize=(15, 5), facecolor='#f7f7f7')
fig.suptitle('Reasons vs race', fontsize=18)

tmp = sd[sd.race != 'Unknown'].copy()
tmp.loc[~tmp.race.isin(['Black', 'White']), 'race'] = 'Other'

tmp[tmp.reason == 'Moving Violation'].race.value_counts().plot(kind='bar', ax=ax[0], color=['r', 'b', 'g'])
tmp[tmp.reason == 'Citizen / 9-1-1'].race.value_counts().plot(kind='bar', ax=ax[1], color=['r', 'b', 'g'])
tmp[tmp.reason == 'Investigative'].race.value_counts().plot(kind='bar', ax=ax[2], color=['r', 'b', 'g'])
tmp[tmp.reason == 'Equipment Violation'].race.value_counts().plot(kind='bar', ax=ax[3], color=['r', 'b', 'g'])

for axes in ax:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
    
ax[0].set_title('Moving Violation', fontsize=14)
ax[1].set_title('911 Call', fontsize=14)
ax[2].set_title('Investigative', fontsize=14)
ax[3].set_title('Equipment Violation', fontsize=14)

plt.show()


# While moving violation seems to target mostly black and white people in equal number, meaning that a black person gets stopped about 3 times more often than a white one, all the other reasons appear to have a **very large skew towards African American**. In particular,
# 
# * 45% of the 911 calls are about a black person
# * 54% of the stops for investigative reasons are about a black person
# * 51% of the stops for equipment violation are about a black person
# 
# Since 911 calls are coming from the population and not the police, it is worth zooming in these calls, in particular the ones regarding suspicious activity

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 5), facecolor='#f7f7f7')

fig.suptitle('911 calls for suspicious activity by race', fontsize=18)

tmp[(tmp.reason == 'Citizen / 9-1-1') & 
    (tmp.problem.str.contains('Suspicious'))].groupby(['problem', 'race']).size().unstack().plot(kind='bar', color=['r', 'g', 'b'], ax=ax)

plt.xticks(rotation=0)

plt.show()


# Hence **44% of the 911 calls for suspicious activity are involving a black person and only 33% a white one**. We must recall that there are about 3 times more white citizens than black ones in Minneapolis.
# 
# Once a police stop is done, the person or the vehicle may be searched. Breaking down how often this happens by race, we get another evident disparity in treatment.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5), facecolor='#f7f7f7')

fig.suptitle('Proportion of stops resulted in a search by race', fontsize=18)

sizes = tmp.groupby(['race', 'personSearch']).size()
props = sizes.div(sizes.sum(level=0), level=0)
props.unstack().plot(kind='barh', stacked=True, color=['k', 'y'], ax=ax[0])

sizes = tmp.groupby(['race', 'vehicleSearch']).size()
props = sizes.div(sizes.sum(level=0), level=0)
props.unstack().plot(kind='barh', stacked=True, color=['k', 'y'], ax=ax[1])

ax[0].set_title('Search on a person', fontsize=14)
ax[1].set_title('Search on a vehicle', fontsize=14)

for axes in ax:
    axes.legend().set_visible(False)
    
plt.show()


#  * Black people get searched 22% of the times they get stopped, against the 9% of white people and the 14% of other races. 
#  * Black people get their vehicle searched 14% of the times they get stopped, against the 4% of white people and 7% of other races.
#  
# This means that not only an African American **gets stopped way more often than a Caucasian, but these stops also end up in a search more than twice as often**.
# 
# 
# # Use of force
# 
# With this disparity in how often a person gets stopped given their race, it is no surprise to see that also the use of force is **more frequently towards a person of color**

# In[ ]:


uof = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_use_of_force.csv')

uof['ResponseDate'] = pd.to_datetime(uof['ResponseDate'])
uof['year'] = uof['ResponseDate'].dt.year
uof['time'] = uof['ResponseDate'].dt.time
uof['hour'] = uof['ResponseDate'].dt.hour
uof = uof[uof.year>1980].copy()
uof['ForceTypeAction'] = uof['ForceTypeAction'].fillna('Other')
uof.loc[uof.ForceTypeAction.str.lower().str.contains('body weight'), 'ForceTypeAction'] = 'Body Weight to Pin'
uof.loc[uof.ForceTypeAction.str.lower().str.contains('neck restraint'), 'ForceTypeAction'] = 'Neck Restraint'
uof.loc[uof.ForceTypeAction.str.lower().str.contains('punch'), 'ForceTypeAction'] = 'Punch'
uof.loc[uof.ForceTypeAction.str.lower().str.contains('joint lock'), 'ForceTypeAction'] = 'Joint Lock'
uof.loc[uof.ForceTypeAction.str.lower().str.contains('mace'), 'ForceTypeAction'] = 'Mace'
uof['TypeOfResistance'] = uof['TypeOfResistance'].fillna('Other')
uof.loc[uof.TypeOfResistance.str.lower().str.contains('tensed'), 'TypeOfResistance'] = 'Tensed'
uof.loc[uof.TypeOfResistance.str.lower().str.contains('commission'), 'TypeOfResistance'] = 'Commission of a Crime'
uof.loc[uof.TypeOfResistance.str.lower().str.contains('fled'), 'TypeOfResistance'] = 'Fled'
uof.loc[uof.TypeOfResistance.str.lower().str.contains('compliance'), 'TypeOfResistance'] = 'Verbal Non-Compliance'
uof.loc[uof.TypeOfResistance.str.lower().str.contains('assault'), 'TypeOfResistance'] = 'Assaulted Police Officer/Horse/K9'
uof.loc[(uof.TypeOfResistance.str.lower().str.contains('other')) | (uof.TypeOfResistance == 'Unspecified'), 'TypeOfResistance'] = 'Other/Unspecified'
uof['SubjectInjury'] = uof['SubjectInjury'].fillna('Not Reported')

uof.head()


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12, 5), facecolor='#f7f7f7', sharey=True)
fig.suptitle('Comparison between race distribution in the population and in police use of force', fontsize=18)

tmp = uof[uof.Race != 'Unknown'].copy()
tmp.loc[~tmp.Race.isin(['Black', 'White']), 'Race'] = 'Other'

(tmp.Race.value_counts() / tmp.shape[0] * 100).plot(kind='bar', color= ['r', 'b', 'g'], ax=ax[1])
pop = pd.Series({'Black': 18.6, 'White': 63.8, 'Other': 23.7})

pop.plot(kind='bar', ax=ax[0], color= ['r', 'b', 'g'])

ax[0].set_title('Population', fontsize=14)
ax[1].set_title('Police use of force', fontsize=14)
ax[0].set_ylabel('% of population')

for axes in ax:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
    axes.grid(axis='y')

plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 10), facecolor='#f7f7f7') 
fig.subplots_adjust(top=0.92)
fig.suptitle('Use of Force by race and gender', fontsize=18)

gs = GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1:])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1, 2])

tmp = uof.copy()
tmp.loc[tmp.Race == 'Native American', 'Race'] = 'Native'
tmp.loc[tmp.Race == 'Other / Mixed Race', 'Race'] = 'Other'
tmp.loc[tmp.Race == 'not recorded', 'Race'] = 'Unknown'
tmp.loc[tmp.Race == 'Pacific Islander', 'Race'] = 'Islander'
tmp.Race.value_counts().plot(kind='bar', ax=ax0)
tmp = uof.copy()
tmp.loc[~tmp.Race.isin(['Black', 'White']), 'Race'] = 'Other'
tmp.groupby(['Race', 'year'], as_index=False).size().unstack(0).plot(ax=ax1, color=['r', 'g', 'b'])
tmp = uof.copy()
tmp.loc[tmp.Sex == 'not recorded', 'Sex'] = 'Unknown'
tmp.groupby(['Sex', 'year']).size().unstack(0).plot(ax=ax2, color=['m', 'tab:orange', 'k'])
tmp.Sex.value_counts().plot(kind='bar', ax=ax3)

ax0.set_xticklabels(ax0.get_xticklabels(), rotation=30)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

ax0.set_title('Use of Force by race')
ax1.set_title('Use of Force by race (2008-2020)')
ax3.set_title('Use of Force by gender')
ax2.set_title('Use of Force by gender (2008-2020)')

plt.show()


# This appears to be a constant during the day as we can see that the proportion of each race stays roughly the same during the 24 hours

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(15, 10), facecolor='#f7f7f7')
fig.subplots_adjust(top=0.92)
fig.suptitle('Use of force per time of day, by race', fontsize=18)

tmp = uof.copy()
tmp.loc[~tmp.Race.isin(['Black', 'White']), 'Race'] = 'Other'
sizes = tmp.groupby(['Race', 'hour']).size()
props = sizes.div(sizes.sum(level=1), level=1)

sizes.unstack(0).plot(kind='bar', stacked=True, color=['r', 'g', 'b'], ax=ax[0])
props.unstack(0).plot(kind='bar', stacked=True, color=['r', 'g', 'b'], ax=ax[1])

ax[1].legend().set_visible(False)

for axes in ax:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0) 

plt.show()


# It is evident that the disproportion between black and white people is much larger with respect to the one we saw in the police stops. Assuming what we have are *all the cases* of use of force and *all the police stops*, we see that 
# 
# * Every 100 African Americans stopped by the police, the police use some kind of force **11 times**
# * Every 100 White Americans stopped by the police, the force is used 6 times

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15, 6), facecolor='#f7f7f7')
fig.subplots_adjust(top=0.92)
fig.suptitle('Number of cases with use of force every 100 police stops (2016-2020)', fontsize=18)

tmp_2 = sd[sd.race != 'Unknown'].copy()
tmp_2.loc[~tmp_2.race.isin(['Black', 'White']), 'race'] = 'Other'

(tmp[tmp.year >= 2016].Race.value_counts().div(tmp_2.race.value_counts()) * 100).plot(kind='bar', ax=ax, color=['r', 'g', 'b'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14)
ax.grid(axis='y')

plt.show()


# Thus **not only the stops are more frequent in the african american community, but they involved the use of force nearly twice more often**

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15, 10), facecolor='#f7f7f7')

sizes = tmp.groupby(['Race', 'ForceType']).size()
props = sizes.div(sizes.sum(level=0), level=0)

props.unstack().plot(kind='barh', stacked=True, ax=ax)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('How is the type of force used by the police distributed across races', fontsize=19)

plt.show()


# Here we see that **chemical irritants and gunpoint display are used more frequently for non-white people**, who instead see more frequent use of bodily force.
# 
# However, the fraction of reported injuries is higher among white people. This is the only statistics so far that does not seem to penalize non-white citizens in Minneapolis. However, the proportion of cases for which this is not reported is very high for all 3 race groups, leaving the doubt of whether or not this is the case.

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(12, 8), facecolor='#f7f7f7')

sizes = tmp.groupby(['Race', 'SubjectInjury']).size()
props = sizes.div(sizes.sum(level=0), level=0)
props.unstack().plot(kind='barh', stacked=True, ax=ax, color=['k', 'grey', 'y'])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title("Proportion of subject's injury by race", fontsize=19)

plt.show()


# We can also see what type of resistance the subject was displaying while the police used the force

# In[ ]:


fig = plt.figure(figsize=(15, 10), facecolor='#f7f7f7') 
fig.subplots_adjust(top=0.95)
fig.suptitle('Type of Resistance', fontsize=18)

gs = GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :2])

# ax4 = fig.add_subplot(gs[2, :2])
# ax5 = fig.add_subplot(gs[2, 1])
# ax6 = fig.add_subplot(gs[2, 2])

tmp.TypeOfResistance.value_counts().plot(kind='bar', ax=ax0, color=['tab:purple', 'tab:orange', 
                                                                    'tab:green', 'tab:red', 
                                                                    'tab:blue', 'tab:brown'])
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=10)
ax0.grid(axis='y')

# tmp[tmp.TypeOfResistance == 'Fled'].groupby('Race').size().plot(kind='barh', ax=ax4)
# tmp[tmp.TypeOfResistance == 'Other/Unspecified'].groupby('Race').size().plot(kind='barh', ax=ax2)
# tmp[tmp.TypeOfResistance == 'Verbal Non-Compliance'].groupby('Race').size().plot(kind='barh', ax=ax3)
# tmp[tmp.TypeOfResistance == 'Tensed'].groupby('Race').size().plot(kind='barh', ax=ax4)
# tmp[tmp.TypeOfResistance == 'Commission of a Crime'].groupby('Race').size().plot(kind='barh', ax=ax5)
# tmp[tmp.TypeOfResistance == 'Assaulted Police Officer/Horse/K9'].groupby('Race').size().plot(kind='barh', ax=ax6)

sizes = tmp.groupby(['Race', 'TypeOfResistance']).size()
props = sizes.div(sizes.sum(level=0), level=0)
props.unstack().plot(kind='barh', stacked=True, ax=ax1)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()


# * Tensed appears to be a generic police term for 'Other unspecified reason', which is unfortunate since it is the most frequent reason.
# * The commission of a crime appears more often for Black people
# * A larger proportion of case for which the police used the force against a white person is a consequence of an assault towards an officer or their animals
# * The police used the force more frequently when a black person was fleeing (either on foot or with a vehicle).
# 
# 
# 
# # Conclusions
# 
# The picture coming out from the data does not leave too much space to interpretation
# 
# * Black people are the minority of the population (18.6%)
# * It is more likely that a 911 call is about a black person (45%)
# * More than half of the stops for investigative reasons or equipment violation involve a black person (54% and 51%)
# * Most of the calls about suspicious activities (person or vehicle) are about a non-white person (44%)
# * Most of the stops for suspicious activity are about a non-white person (77%)
# * When stopped, a non-white person get searched **twice more often** than a white one
# * Most of the times the police uses the force, it is against a black person (60%)
# * When the police stop a black person, use the force **twice more often** than with other races
# * When the police use the force, it is more likely they use chemical irritants or display their weapon if the subject is not white
# * It is more likely the police use the force while you are fleeing if you are black, while it is more frequent the use of force for assaulting a police officer among white people.
# 
# This situation is not uncommon (older analyses on [police shootings](https://www.kaggle.com/lucabasa/exploring-racial-bias-in-fatal-police-shootings) and [differences in policing](https://www.kaggle.com/lucabasa/automate-combine-explore)) and it is not something emerging in the last few years. It is a problem that it has always been there, that has roots in systemic discrimination and disparities in opportunity and treatment. We did not need a video to see it, but the video is there and we can't keep pretending this is not a problem.

# In[ ]:




