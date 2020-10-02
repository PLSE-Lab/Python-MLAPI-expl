#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook aims to explore the dataset containing data about fatal police shootings in the U.S.A.
# 
# After getting a general idea of what the dataset contains, I will focus on exploring the race of the victims and their behavior according to the dataset.
# 
# Since each State has a different set of rules, population composition, and socio-economic situation, I will dedicate the second part of this notebook to observe differences between the States and possibly elaborate better on the findings of part 1.
# 
# At last, I want to see if the introduction of body cam had some kind of effect in deadly shootings.
# 
# **A word of caution before we move forward**. This dataset defines "*Hispanic*" as a race while the census stats refer to it as an ethnicity, being present across what they define the races. Therefore, every time I will confront a sample with the population, my analysis will be limited by this fact. Moreover, I will refer to the races as they are called in the dataset.
# 
# Secondly, there is no escape from bias, even an analysis like this one will have some (just considering the questions I will look for an answer to will make that clear). Moreover, data collection can also be biased (for example, "*signs of mental illness*" is very subjective and it also sounds like an opinion from a non-expert in mental illness). Therefore, I see this notebook as the healthiest contribution to this conversation I can bring, which doesn't mean it will be led you to the answers you were looking for.
# 
# At last, this dataset **does not include much about the situation**, does not include the times a representative of the law enforcement draws their gun rather than not, the times they didn't shoot to kill, etc. In my opinion, without data about these aspects, any conclusion can only be partial.

# In[ ]:


import pandas as pd
import numpy as np

import scipy.stats as stats

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_columns",None)


# In[ ]:


# importing dataset
shot = pd.read_csv('../input/PoliceKillingsUS.csv', encoding = "ISO-8859-1")
income = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding = "ISO-8859-1")
poverty = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding = "ISO-8859-1")
education = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding = "ISO-8859-1")
racedist = pd.read_csv('../input/ShareRaceByCity.csv', encoding = "ISO-8859-1")


# I want to keep the preprocessing of the data at a minimum, I will thus just do the following:
# 
# * manipulate the date to a convenient format, including creating some features about it
# * remove some anomalous data from the datasets regarding the cities
# * convert the data into the format they are supposed to be

# In[ ]:


#cleaning formats and similar

#Shootings
shot.date = pd.to_datetime(shot.date)
shot.insert(3, 'year', pd.DatetimeIndex(shot.date).year)
shot.insert(4, 'month', pd.DatetimeIndex(shot.date).month)
shot.insert(5, 'day', pd.DatetimeIndex(shot.date).day)
shot.insert(6, 'dayofweek', pd.DatetimeIndex(shot.date).weekday_name)

#Race shares
racedist.rename(columns = {'Geographic area':'Geographic Area'}, inplace = True)
racedist.share_asian = racedist.share_asian.replace("(X)", np.nan)
racedist.share_black = racedist.share_black.replace("(X)", np.nan)
racedist.share_hispanic = racedist.share_hispanic.replace("(X)", np.nan)
racedist.share_native_american = racedist.share_native_american.replace("(X)", np.nan)
racedist.share_white = racedist.share_white.replace("(X)", np.nan)
racedist.share_asian = pd.to_numeric(racedist.share_asian)
racedist.share_black = pd.to_numeric(racedist.share_black)
racedist.share_hispanic = pd.to_numeric(racedist.share_hispanic)
racedist.share_native_american = pd.to_numeric(racedist.share_native_american)
racedist.share_white = pd.to_numeric(racedist.share_white)

#Incomes
income['Median Income'] = income['Median Income'].replace("(X)", np.nan)
income['Median Income'] = income['Median Income'].replace("-", np.nan)
income['Median Income'] = income['Median Income'].replace("2,500-", "2500")
income['Median Income'] = income['Median Income'].replace("250,000+", "250000")
income['Median Income'] = pd.to_numeric(income['Median Income'])

#Poverty rate
poverty.poverty_rate = poverty.poverty_rate.replace("-", np.nan)
poverty['poverty_rate'] = pd.to_numeric(poverty['poverty_rate'])

#Education rate
education.percent_completed_hs = education.percent_completed_hs.replace("-", np.nan)
education['percent_completed_hs']  = pd.to_numeric(education['percent_completed_hs'])


# As a manipulation, I know that the "armed" feature contains things from a gun to a beer bottle, I decided to encode them as follows. The idea is clear enough for short, mid, and long range weapons, then there are vehicles and unknown weapons (which at the end I both considered as Unknown), and at last unarmed.
# 
# I do know that a person with a toy weapon is unarmed, but some rule of engagement may include the plausible doubt and trigger the police to shoot.
# 
# On fleeing and threat level I was less inclined to grasp nuances, as you can see.

# In[ ]:


short_range = ['hammer','pick-axe','glass shard','box cutter','sharp object', 'meat cleaver',
              'stapler', 'chain saw', 'metal object', 'bayonet', 'baton', 'tire iron', 
               'baseball bat and fireplace poker','machete', 'knife','garden tool','pipe',
              'straight edge razor','blunt object','ax','scissors','hatchet and gun','pole and knife',
              'hatchet','carjack','lawn mower blade','metal hand tool','beer bottle','metal stick',
              'piece of wood','screwdriver']

long_range = ['Taser','gun and knife', 'crossbow', 'gun','bean-bag gun','guns and explosives',
              'machete and gun','fireworks']

mid_range = ['shovel', 'pitchfork', 'metal pipe',"contractor's level", 'pole', 'crowbar', 'flagpole',
            'rock', 'oar', 'metal pole', 'chain','brick', 'metal rake',  'sword','spear','baseball bat']

vehicle = ['motorcycle', 'vehicle'] 

unknown = ['unknown weapon', 'hand torch', 'toy weapon', 'cordless drill', 'undetermined', 'flashlight',
           'nail gun']

unarmed = ['unarmed']

shot.loc[shot.armed.isin(short_range), 'Armed_class'] = 'Short range weapon'
shot.loc[shot.armed.isin(long_range), 'Armed_class'] = 'Long range weapon'
shot.loc[shot.armed.isin(mid_range), 'Armed_class'] = 'Mid range weapon'
shot.loc[shot.armed.isin(vehicle), 'Armed_class'] = 'Unknown'
shot.loc[shot.armed.isin(unknown), 'Armed_class'] = 'Unknown'
shot.loc[shot.armed.isin(unarmed), 'Armed_class'] = 'Unarmed'

shot.Armed_class = shot.Armed_class.fillna('Unknown')

shot.loc[shot.flee == 'Not fleeing', 'Fleeing_class'] = 'Not fleeing'
shot.loc[shot.flee != 'Not fleeing', 'Fleeing_class'] = 'Fleeing'

shot.loc[shot.threat_level == 'attack', 'Threat_class'] = 'Attack'
shot.loc[shot.threat_level != 'attack', 'Threat_class'] = 'Not attack'


# # A first exploration of the shootings
# 
# The dataset contains the following:

# In[ ]:


shot.info()


# Quite a few **missing data** in the race feature, which will be of my primary interest later. Let's see when this feature is missing the most.

# In[ ]:


shot[shot.race.isnull()].year.value_counts()


# In[ ]:


shot[(shot.race.isnull()) & (shot.year == 2017)].month.value_counts()


# I decide to restrict the dataset to every entry **before June 1st 2017**.

# In[ ]:


shot = shot[shot.date < '2017-06-01']
shot.info()


# Let's see how the number of deadly shootings varies in these 2 and half years.

# In[ ]:


# Number of deadly shootings per month over time
shot[['year', 'month', 'name']].groupby(['year', 'month']).size().plot(figsize=(15,5))
plt.title("Number of deadly shootings over time")


# Anticipating what I am going to do next, I want to see the same graph segmented by race (white in blue, non-white collectively in orange)

# In[ ]:


shot[shot.race == "W"][['year', 
                        'month', 
                        'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), 
                                                                                         label="W")
shot[shot.race != "W"][['year', 
                        'month', 
                        'name']].groupby(['year', 'month']).size().plot(figsize=(15,5),
                                                                                        label="Non-W")
plt.title("Deadly shootings segmented by race over time")
plt.legend()


# Gender-wise, the situation is overwhelmingly skewed towards the men.

# In[ ]:


shot[shot.gender == "M"][['year', 
                          'month', 
                          'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="M")
shot[shot.gender == "F"][['year', 
                          'month', 
                          'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="F")
plt.title("Deadly shootings segmented by gender over time")
plt.legend()


# Also in terms of presence of a **body camera**, in most of the cases the camera was not used.

# In[ ]:


shot[shot.body_camera == True][['year', 
                                'month', 
                                'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="bodyCam")
shot[shot.body_camera != True][['year', 
                                'month', 
                                'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="NoCam")
plt.title("Deadly shootings segmented by body cam over time")
plt.legend()


# Looking at the **threat level** (if the police were attacked or not before or during the shooting), I observe a slight increase in non-threatening situations resulted in a deadly shooting.

# In[ ]:


shot[shot.Threat_class == "Attack"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Attack")
shot[shot.Threat_class != 'Attack'][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="NoAttack")
plt.title("Deadly shootings segmented by threat over time")

plt.legend()


# The next plot shows how the incidents involving armed people (blue), potentially armed people (orange), and unarmed people (green) evolved across time. Nothing to notice except that most of the victims were armed.

# In[ ]:


fil = (shot.Armed_class != "Unknown") & (shot.Armed_class != "Unarmed")
shot[fil][['year', 
           'month', 
           'name']].groupby(['year', 
                             'month']).size().plot(figsize=(15,5), label="Armed")
shot[shot.Armed_class == "Unknown"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Unknown")
shot[shot.Armed_class == "Unarmed"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Unarmed")
plt.title("Deadly shootings segmented by weapon over time")

plt.legend()


# On the same line, in most of the cases there were no **signs of mental illness**.

# In[ ]:


shot[shot.signs_of_mental_illness == True][['year', 
                                            'month', 
                                            'name']].groupby(['year',
                                                              'month']).size().plot(figsize=(15,5), label="Signs")
shot[shot.signs_of_mental_illness != True][['year', 
                                            'month', 
                                            'name']].groupby(['year',
                                                              'month']).size().plot(figsize=(15,5), label="NoSigns")
plt.title("Deadly shootings segmented by signs of mental illness over time")
plt.legend()


# Next, I want to focus on the **age** feature. We see that most of the shootings involved someone between the age of 20 and 40, with two extremes (6 and 88).

# In[ ]:


def segm_target(var, target):
    count = shot[[var, target]].groupby([var], as_index=True).count()
    count.columns = ['Count']
    mean = shot[[var, target]].groupby([var], as_index=True).mean()
    mean.columns = ['Mean']
    ma = shot[[var, target]].groupby([var], as_index=True).max()
    ma.columns = ['Max']
    mi = shot[[var, target]].groupby([var], as_index=True).min()
    mi.columns = ['Min']
    median = shot[[var, target]].groupby([var], as_index=True).median()
    median.columns = ['Median']
    std = shot[[var, target]].groupby([var], as_index=True).std()
    std.columns = ['Std']
    df = pd.concat([count, mean, ma, mi, median, std], axis=1)
    return df

def corr_2_cols(Col1, Col2):
    res = shot.groupby([Col1, Col2]).size().unstack()
    res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
    return res


# In[ ]:


shot.age.hist(bins=20)
plt.tight_layout()
plt.title("Age distribution")


# If we look at how the age is segmented by race, we notice that generally *black people involving in deadly shootings are younger than in other races*.

# In[ ]:


segm_target('race', 'age')


# In[ ]:


g = sns.FacetGrid(shot, hue='race', size = 7)
g.map(plt.hist, 'age', alpha = 0.3, bins= 20)
g.add_legend()


# Regarding mental illness, I can see that when true the victims are on average older.

# In[ ]:


segm_target('signs_of_mental_illness', 'age')


# In[ ]:


g = sns.FacetGrid(shot, hue='signs_of_mental_illness', size = 7)
g.map(plt.hist, 'age', alpha = 0.3, bins= 20)
g.add_legend()


# If we instead look at how race and gender are correlated, we notice that generally more thant 95% of the victims were males, with the exception of Native Americans, where we have a higher proportion of females (although the numbers are pretty low).

# In[ ]:


corr_2_cols('race', 'gender')


# On the same line, we notice that black and hispanic people were showing signs of mental illness less often when shot by the police. If we instead segment by gender, we observe this feature to be present more often if the victim is a woman.

# In[ ]:


corr_2_cols('race', 'signs_of_mental_illness')


# In[ ]:


corr_2_cols('gender', 'signs_of_mental_illness')


# # Does race matter that much?
# 
# Let's begin by looking at the general distribution by race.

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(data=shot, x="race")

plt.title("Total number of people killed, by race", fontsize=12)


# Most of the victims were white, followed by black, then hispanic. If we look at the percentage of people killed of every race, we get the following.

# In[ ]:


#shot.race.dropna(inplace = True)
labels = shot.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = shot.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed, by race',fontsize = 12)


# Which already tells us something, given that the distribution of the population is the following (https://www.census.gov/quickfacts/fact/table/US/PST045216):
# 
# * White: 61.3%
# * Black: 13.3%
# * Asian: 5.7%
# * Hispanic: 17.8%
# * Natives: 1.5%
# 
# Just for the sake of it, let's perform a standard test to check if the sample represented by the victims of police shootings is randomly estracted from the population or not, according to the race segmentation. In other words, a chi squared test with the null hypothesis that the sample is estracted randomly from the total population.

# In[ ]:


tot = shot.shape[0]
USA_demo = pd.DataFrame({'race' : ["W", "B", "H", "A", "N", "O"],
                        'Population' : [int(0.613*tot),
                                       int(0.133*tot),
                                       int(0.178*tot),
                                       int(0.057*tot),
                                       int(0.015*tot),
                                       int((1-0.613-0.178-0.133-0.057-0.015)*tot)]})

USA_demo = USA_demo.sort_values(by='race')
USA_demo = USA_demo.set_index('race')
expected = USA_demo.Population
expected


# In[ ]:


shotbyrace = pd.crosstab(index=shot.race, columns="count")
observed = shotbyrace['count']
observed


# In[ ]:


stats.chisquare(observed, expected)


# Therefore the test tells us that there is less than 5% of probability that our sample is estracted randomly from the US population. **Other factors are most likely in place**.
# 
# Let's see how armed were the people of every race.
# 
# * 10% of black people killed were unarmed, 16% had an unknown weapon
# * 5% of white people killed were unarmed, 16% had an unknown weapon
# * 8% of hispanic people killed were unarmed, 19% had an unknown weapon

# In[ ]:


shot[['Armed_class', 'race', 'name']].groupby(['race', 'Armed_class']).count()


# ## Fleeing unarmed vs race
# 
# First, I want to focus on the people that was fleeing. We can see that most of them were fleeing with a long range weapon (most commonly a gun) or an unknown weapon (potentially a gun).

# In[ ]:


fleeing = shot[(shot.Fleeing_class == "Fleeing")]
fleeing.Armed_class.value_counts()


# Segmenting by race, we get the following:
# 
# * 13% of black people were fleeing unarmed when killed, 22% had an unknown weapon.
# * 8% of hispanic people were fleeing unarmed when killed, 25% had an unknown weapon.
# * 7% of white people were fleeing unarmed when killed, 27% had an uknown weapon.

# In[ ]:


fleeing[['Armed_class', 'race', 'name']].groupby(['race', 'Armed_class']).count()


# Interestingly enough, most of the people were fleeing while attacking.

# In[ ]:


fleeing.Threat_class.value_counts()


# If we then look at the distribution by race, we see how the ratio of black and hispanic people are increasing if we just focus on people fleeing. This can mean that white people flee less often, or that non-white people are more likely to be killed when they are fleeing.

# In[ ]:


labels = fleeing.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = fleeing.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing, by race',fontsize = 12)


# Let's then look at those **fleeing unarmed**, which are not that many and still have some level of threat apparently.

# In[ ]:


fleeunarmed = fleeing[fleeing.Armed_class == 'Unarmed']
fleeunarmed.Threat_class.value_counts()


# If we then look at the race distribution, we see that **most of the people killed while fleeing unarmed were black** (even more staggering since this is only the third race in the US, numerically).

# In[ ]:


labels = fleeunarmed.race.value_counts().index
colors = ['red','orange','green','brown','purple']
explode = [0,0,0,0,0]
sizes = fleeunarmed.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing unarmed, by race',fontsize = 12)


# The ones fleeing not unarmed (armed or probably armed) are instead distributed in a way more similar to the full sample of deadly shootings.

# In[ ]:


fleearmed = fleeing[fleeing.Armed_class != 'Unarmed']
labels = fleearmed.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = fleearmed.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing unarmed, by race',fontsize = 12)


# # Differences between the States
# 
# First, I need some data about the full population, I used wikipedia (https://en.wikipedia.org/wiki/Demography_of_the_United_States#Race_and_ethnicity) and struggled with the absence of the race "hispanic". I promise myself to edit this part with the proper corrections but for now let's proceed without that race.

# In[ ]:


name = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
       'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
       'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
       'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
population = [4830620,733375,6641928,2958208,38421464,5278906,3596222,926454,647484,19645772,10006693,
              1406299,1616547,12873761,6568645,309526,2892987,4397353,4625253,1329100,5930538,6705586,
              9900571,5419171,2988081,6045448,1014699,1869365,2798636,1324201,8904413,2084117,19673174,
              9845333,721640,11575977,3849733,3939233,12779559,1053661,4777576,843190,6499615,26538614,
              2903379,626604,8256630,6985464,1851420,5742117,579679]
demoW = [68.8,66.0,78.4,78.0,61.8,84.2,77.3,69.4,40.2,76.0,60.2,25.4,91.7,72.3,84.2,91.2,85.2,87.6,62.8,95.0,57.6,
        79.6,79.0,84.8,59.2,82.6,89.2,88.1,69.0,93.7,68.3,73.2,64.6,69.5,88.7,82.4,73.1,85.1,81.6,81.1,67.2,
        85.0,77.8,74.9,87.6,94.9,69.0,77.8,93.6,86.5,91.0]
demoB = [26.4,3.4,4.2,15.5,5.9,4.0,10.3,21.6,48.9,16.1,30.9,2.0,0.6,14.3,9.2,3.2,5.8,7.9,32.1,1.1,29.5,7.1,14.0,
        5.5,37.4,11.5,0.5,4.7,8.4,1.3,13.5,2.1,15.6,21.5,1.6,12.2,7.2,1.8,11.0,6.5,27.5,1.6,16.8,11.9,1.1,1.1,
         19.2,3.6,3.3,6.3,1.1]
demoN = [0.5,13.8,4.4,0.6,0.7,0.9,0.2,0.3,0.3,0.3,0.3,0.2,1.3,0.2,0.2,0.3,0.8,0.2,0.6,0.6,0.3,0.2,0.5,1.0,0.4,
         0.4,6.5,0.9,1.1,0.2,0.2,9.1,0.4,1.2,5.3,0.2,7.3,1.2,0.2,0.5,0.3,8.6,0.3,0.5,1.1,0.3,0.3,1.3,0.2,0.9,
         2.2]
demoA = [1.3,7.1,3.2,1.6,14.1,3.0,4.2,3.6,3.7,2.7,3.6,47.8,1.4,5.0,1.9,2.1,2.7,1.3,1.7,1.1,6.0,6.0,2.7,4.4,1.0,
         1.8,0.8,2.1,8.3,2.4,9.0,1.5,8.0,2.6,1.2,1.9,2.0,4.4,3.1,3.2,1.5,1.2,1.7,4.3,3.1,1.4,6.1,8.3,0.7,2.5,1.0]
demoO = [3.0,9.7,9.7,4.2,17.4,7.8,7.9,5.0,6.9,4.9,4.9,24.8,5.0,8.0,4.5,3.3,5.5,3.0,2.8,2.2,6.6,7.1,3.7,4.2,
         2.1,3.5,3.0,4.1,13.2,2.3,8.9,14.2,11.5,5.4,5.4,3.0,3.3,10.4,7.5,4.1,8.6,3.5,3.5,8.5,7.1,2.2,
        5.4,9.0,2.2,3.8,4.8]

states = pd.DataFrame({'state' : name, 'Population': population, 'State_share_W': demoW,
                   'State_share_B': demoB, 'State_share_N': demoN, 'State_share_A': demoA,
                    'State_share_O': demoO})
names = states.state
states.drop(labels=['state'], axis = 1, inplace=True)
states.insert(0, 'state', names)

states.sample(10)


# Now let's add some info about the shootings in every state

# In[ ]:


kills = shot[['name', 'state']].groupby('state', as_index = True).count()
kills.rename(columns = {'name' : 'N_kills'}, inplace=True)

temp = shot[['state', 'race', 'name']].groupby(['state', 'race']).count().unstack().fillna(0)
kills['N_kills_A'] = temp['name']['A']
kills['N_kills_B'] = temp['name']['B']
kills['N_kills_H'] = temp['name']['H']
kills['N_kills_N'] = temp['name']['N']
kills['N_kills_W'] = temp['name']['W']
kills['N_kills_O'] = kills.N_kills - (kills.N_kills_A + kills.N_kills_B +
                                                     kills.N_kills_H + kills.N_kills_N +
                                                     kills.N_kills_W) #because some values are missing

temp = shot[['state', 'gender', 'name']].groupby(['state', 'gender']).count().unstack().fillna(0)
kills['N_kills_Fem'] = temp['name']['F']
kills['N_kills_Mal'] = temp['name']['M']

temp = shot[['state', 
             'signs_of_mental_illness', 
             'name']].groupby(['state', 'signs_of_mental_illness']).count().unstack().fillna(0)
kills['N_mental_illness'] = temp['name'][True]

temp = shot[['state', 
             'body_camera', 
             'name']].groupby(['state', 'body_camera']).count().unstack().fillna(0)
kills['N_body_camera'] = temp['name'][True]

temp = shot[['state', 
             'manner_of_death', 
             'name']].groupby(['state', 'manner_of_death']).count().unstack().fillna(0)
kills['N_Tasered'] = temp['name']['shot and Tasered']

kills['state'] = kills.index
kills.sample(10)


# At last, let's create some percentages for future reference.

# In[ ]:


states = pd.merge(states, kills, on ='state', how = 'left')
states['Kills_pp'] = states['N_kills'] / states.Population
states['Perc_Kills_A'] = states.N_kills_A / states.N_kills * 100
states['Perc_Kills_B'] = states.N_kills_B / states.N_kills * 100
states['Perc_Kills_H'] = states.N_kills_H / states.N_kills * 100
states['Perc_Kills_N'] = states.N_kills_N / states.N_kills * 100
states['Perc_Kills_W'] = states.N_kills_W / states.N_kills * 100
states['Perc_Male'] = states.N_kills_Mal / states.N_kills * 100
states['Perc_Female'] = states.N_kills_Fem / states.N_kills * 100
states['Perc_Mental'] = states.N_mental_illness / states.N_kills * 100
states['Perc_BodyCam'] = states.N_body_camera / states.N_kills * 100
states['Perc_Tasered'] = states.N_Tasered / states.N_kills * 100
states.sample(10)


# In[ ]:


states.describe()


# Let's begin by looking at the number of deadly shootings in every State. Here we find California, Texas, and Florida on top, but they are also the most populated States.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(data=states, x="state", y="N_kills")

plt.title("Total number of people killed, by state", fontsize=12)


# Indeed, pro capita we find Iowa, New Mexico, and Alaska. It would be interesting to see if those states have different rules of engagement or other peculiarities to justify this difference with, for example, New York that we find on the bottom.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Kills_pp", data=states)

plt.title("Number of killed pro capita, by state", fontsize=12)


# In[ ]:


states.sort_values(by='Kills_pp', ascending=False).head()


# In[ ]:


states.sort_values(by='Kills_pp').head()


# Gender wise, we can only oberve that Iowa is again on top as percentage of Female victims, together with DC (here considered a State), both followed by Wyoming.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Female", data=states)

plt.title("Percentage of females killed, by state", fontsize=12)


# In[ ]:


states.sort_values(by='Perc_Female', ascending=False).head()


# Regarding mental illness, the curious result is that all 5 men killed in New Hampshire were showing signs of mental illness. Not enough data to do a statistic but enough to raise the question if that states defines mental illness differently.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Mental", data=states)

plt.title("Percentage of killed with signs of mental illness, by state", fontsize=12)


# In[ ]:


states.sort_values(by='Perc_Mental', ascending=False).head()


# The body cams were used in all the 3 cases happened in Vermont, which is an anomaly with respect of the rest of the Country.

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_BodyCam", data=states)

plt.title("Percentage of killings with a body cam, by state", fontsize=12)


# In[ ]:


states.sort_values(by='Perc_BodyCam', ascending=False).head()


# The amount of people Tasered before being shot (I am assuming not after) can be a good indicator if in the future one wants to explore the mentioned rules of engagement. Moreover one my ask if the high result of Vermont is just a statistical fluctation (most likely) or a hint to explain why there we only have 3 deadly shootings (police trained to use the gun as final resource).

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Tasered", data=states)

plt.title("Percentage of killed and tasered, by state", fontsize=12)


# In[ ]:


states.sort_values(by='Perc_Tasered', ascending=False).head()


# At last, let's see how the distribution according to race changes between the full population and the one involved in deadly shootings, per state.

# In[ ]:


white = states.Perc_Kills_W
black = states.Perc_Kills_B
hispanic = states.Perc_Kills_H
native = states.Perc_Kills_N
asian = states.Perc_Kills_A

ind = states.state    
width = 0.75    
plt.figure(figsize=(16,5))

p1 = plt.bar(ind, white, width, color='orange', align='edge')
p2 = plt.bar(ind, black, width, bottom=white, color ='red', align='edge')
p3 = plt.bar(ind, hispanic, width, bottom=black+white, color ='green', align='edge')
p4 = plt.bar(ind, native, width, bottom = black+white+hispanic, color = 'brown', align='edge')
p5 = plt.bar(ind, asian, width, bottom = black+white+hispanic+native, color = 'blue', align='edge')

plt.ylabel('Percentage')
plt.title('Deadly killings segmented by race, per state')
plt.xticks(ind)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('W', 'B', 'H', 'N', 'A'))

plt.show()


# In[ ]:


white = states.State_share_W
black = states.State_share_B
hispanic = states.State_share_O
native = states.State_share_N
asian = states.State_share_A

ind = states.state    
width = 0.75    
plt.figure(figsize=(16,5))

p1 = plt.bar(ind, white, width, color='orange', align='edge')
p2 = plt.bar(ind, black, width, bottom=white, color ='red', align='edge')
p3 = plt.bar(ind, hispanic, width, bottom=black+white, color ='green', align='edge')
p4 = plt.bar(ind, native, width, bottom = black+white+hispanic, color = 'brown', align='edge')
p5 = plt.bar(ind, asian, width, bottom = black+white+hispanic+native, color = 'blue', align='edge')

plt.ylabel('Percentage')
plt.title('Population segmented by race, per state')
plt.xticks(ind)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('W', 'B', 'H', 'N', 'A'))

plt.show()


# Small errors on the side (propably due to approximation), we can see some anomalities. Let's then focus race by race on the most relevant states.
# 
# For Asian people, the result that hits me the most is the one of Luisiana, where 3.8% of people killings involved an Asian person and only 1.7% is Asian.

# In[ ]:


fil = ['state', 'State_share_A', 'Perc_Kills_A', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_A', ascending=False).head()[fil]


# Black people, we have seen it already, have a very though time in the US, in particular in the following states (Rode Island is less relevant due to the low number of shootings).

# In[ ]:


fil = ['state', 'State_share_B', 'Perc_Kills_B', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_B', ascending=False).head()[fil]


# As I said before, for Hispanic people this analysis is complicated so the following is very premature.

# In[ ]:


fil = ['state', 'State_share_W', 'Perc_Kills_H', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_H', ascending=False).head()[fil]


# For Native Americans it is also difficult to do analysis due to the low numbers, but the result of Alaska is interesting and could spark some further research.

# In[ ]:


fil = ['state', 'State_share_N', 'Perc_Kills_N', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_N', ascending=False).head()[fil]


# For white people the only result that partially surprises me is the one of Montana.

# In[ ]:


fil = ['state', 'State_share_W', 'Perc_Kills_W', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_W', ascending=False).head()[fil]


# # Have Body Cams any effect?
# 
# First I want to stress out that the effect I am looking for is *on the data* so "are the data different when there is a body cam?".
# 
# A more interesting question would be to see how many shootings didn't happen due to the presence of a body cam but we don't have data on that now.
# 
# We have only 254 entries, let's see if they differ somehow.

# In[ ]:


bodycam = shot[shot.body_camera == True]
bodycam.shape


# Signs of mental illness are slighly more common with respect to the full set and does not present any significant difference in how it is distributed race or gender wise.

# In[ ]:


res = bodycam.groupby(['gender', 'signs_of_mental_illness']).size().unstack()
res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
res


# In[ ]:


res = bodycam.groupby(['race', 'signs_of_mental_illness']).size().unstack()
res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
res


# The distribution by race is a little bit different, with less white people in proportion.

# In[ ]:


labels = bodycam.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = bodycam.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed with body camera, by race',fontsize = 12)


# At last, we can observe that there are no significant differences between these case and the full set of cases. Although I reserve myself to better elaborate in a future version of the notebook.
# 
# The only exception being that if before most of the people fleeing were also attacking, here we observe an even split.

# In[ ]:


bodycam.Armed_class.value_counts()


# In[ ]:


bodycam[bodycam.Fleeing_class == 'Fleeing'].Threat_class.value_counts()


# In[ ]:


bodycam[bodycam.Armed_class == 'Unarmed'].Threat_class.value_counts()


# In[ ]:


bodycam[bodycam.Armed_class == 'Unknown'].Threat_class.value_counts()


# # Conclusion
# 
# As mentioned before, the data are incomplete and thus any conclusion would be dangerously close to an opinion.
# 
# The only thing I can say is that the distribution of these cases across the races it is very unlikely to be related to the one we have in the full population and this result becomes more and more evident if we focus on people fleeing unarmed. Therefore these data tell us that there must be another explanation.
# 
# My knowledge of the US is very limited, thus please let me know if there are numbers that require further investigation (like "the state X is famous for never shooting, can we investigate that?"). Thus please let me know how can I expand this notebook further. Even better, let me know what did I do wrong.
# 
# Thanks for reading.
