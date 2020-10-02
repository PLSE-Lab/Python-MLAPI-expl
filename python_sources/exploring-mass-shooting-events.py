#!/usr/bin/env python
# coding: utf-8

# # Exploring Mass Shooting Events
# 
# The **Stanford Mass Shootings in America** is a record of standardized data on mass shooting events. It is not meant to be comprehensive, but does (hopefully) provide an unbiased estimate of the characteristics of a defined mass shooting event.
# 
# In this notebook we will explore mass shooting events. We will probe the basic dataset attributes and hopefully uncover some interesting effects from the data! This exploratory data analytics notebook is recommended for beginners and those interested in probing this dataset further. Feel free to fork this notebook and/or copy the code here and explore further on your own!

# ## Munging the data

# In[ ]:


import pandas as pd
pd.set_option('max_columns', None)
shootings = pd.read_csv('../input/stanford-msa/mass_shooting_events_stanford_msa_release_06142016.csv')
shootings.head(3)


# In[ ]:


shootings['Date'] = pd.to_datetime(shootings['Date'])


# ## Exploring shooting event characteristics
# 
# Let's grid out the characteristics of the dataset so that we know what we're working with. We'll start with the date and time characteristics, after all the dataset is hand-curated.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

f, axarr = plt.subplots(2, 2, figsize=(14, 8))
# plt.suptitle('Mass Shooting Events Breakdown', fontsize=18)
f.subplots_adjust(hspace=0.5)

kde_kwargs = {'color': 'crimson', 'shade': True}

sns.kdeplot(shootings['Number of Victim Fatalities'], ax=axarr[0][0], **kde_kwargs)
axarr[0][0].set_title("Victim Fatalities", fontsize=14)

sns.kdeplot(shootings['Number of Victims Injured'], ax=axarr[0][1], **kde_kwargs)
axarr[0][1].set_title("Victim Injuries", fontsize=14)

sns.countplot(shootings['Day of Week'], ax=axarr[1][0], color='salmon')
axarr[1][0].set_title("Day of Week of Attack", fontsize=14)

sns.kdeplot(shootings['Date'].dt.year, ax=axarr[1][1], **kde_kwargs)
axarr[1][1].set_title("Year of Attack", fontsize=14)

sns.despine()


# The number of victim fatalities and victim injuries in these events is very similar --- it is almost entirely squeezed between 1 and 10 fatalities. Recall also the definition of a mass shooting from the dataset description: "3 or more shooting victims (not necessarily fatalities), not including the shooter."
# 
# The distribution of days on which shooting occur is centered around Thursday and the middle-to-end of the week more generally. Overall however the differences are not very strong.
# 
# The dataset is *very* skewed towards recent events. In fact, the majority of the events in the dataset occured in 2010 or after! What this means is that this dataset is probably fairly representative of the *motivations* of mass shooting events, but not of the *volume*. I expect the data points earlier in the dataset time-wise are large-scale and/or especially well-known shootings (like Virginia Tech). I do not know why there is a dip around the year 2000.

# ## Shooter Characteristics
# 
# All that being said, let's examine the characteristics of the shooters themselves. Do they follow an archetype?

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

f, axarr = plt.subplots(3, 2, figsize=(14, 12))
# plt.suptitle('Mass Shooting Events Breakdown', fontsize=18)
f.subplots_adjust(hspace=0.5)

kde_kwargs = {'color': 'crimson', 'shade': True}

sns.countplot((shootings['Total Number of Fatalities'] - 
               shootings['Number of Victim Fatalities']) > 0, ax=axarr[0][0])
axarr[0][0].set_title("Shooter(s) Killed At Scene", fontsize=14)

sns.countplot(shootings['Shooter Sex'], ax=axarr[0][0])
axarr[0][0].set_title("Sex of Shooter(s)", fontsize=14)

sns.kdeplot(
    shootings[shootings['Average Shooter Age'].map(
        lambda v: pd.notnull(v) and "Unknown" not in str(v) and str(v).isdigit())
             ]['Average Shooter Age'], 
    ax=axarr[0][1], **kde_kwargs)
axarr[0][1].set_title("Average Age of Shooter(s)", fontsize=14)

sns.countplot(shootings['Fate of Shooter at the scene'], ax=axarr[1][0])
axarr[1][0].set_title("Fate of Shooter(s) at Scene", fontsize=14)

sns.countplot(shootings['Shooter\'s Cause of Death'], ax=axarr[1][1])
axarr[1][1].set_title("Shooter(s) Cause of Death", fontsize=14)

sns.countplot(shootings['History of Mental Illness - General'], ax=axarr[2][0])
axarr[2][0].set_title("History of Mental Illness?", fontsize=14)

sns.countplot(shootings['School Related'].map(lambda v: v if pd.isnull(v) else str(v).replace('no', 'No').replace('Killed', 'Unknown').replace('UnkNown', 'Unknown')), ax=axarr[2][1])
axarr[2][1].set_title("School Related?", fontsize=14)

sns.despine()


# The distribution of the age of shooters is bimodal, centered around 20 and 40 year olds.
# 
# That is an interesting result that I would not have necessarily expected.
# 
# Male shooters are twice as likely as female shooters. Only about a third of shooters have a known mental ilness history, which says something about the state of mental health case management today.
# 
# A bit less than half of shooters are dead on the scene; about two-thirds of those commit suicide, while the remaining third are killed by response forces. 
# 
# The media tends to focus on school-related shootings. While it's coloquially true that school-related shootings are the most lethal ones, the Stanford dataset shows that the majority (maybe three-quarters) of shootings are not school related at all.

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()

sns.countplot(
    shootings[shootings['Shooter Race'].isin(
            shootings['Shooter Race'].value_counts()[(shootings['Shooter Race'].value_counts() > 3)].index.values
    )]['Shooter Race'], ax=ax, color='salmon'
)
sns.despine()


# This is the relatively controversial part of this dataset, politically speaking: half of race-identifiable shooters are white. This fact ends up in the news media quite often because it is one of the leading arguments made by gun control advocates (and, in general, the political left wing) in the United States: that the dominant color of terrorism is white, and that current gun control laws are not strong enough to prevent this from happening.

# ## Geospatial distribution

# Finally, let's take a quick peek at the geospatial distribution of these shootings. This will  follow the distribution of population in the United States pretty closely: there's little reason to believe that, say, Midwesterners are more mentally unstable than folks on the Eastern Seaboard, or vice versa. Still, seeing the data on a map gives us an appreciation for the *volume* of what we're looking at.

# In[ ]:


import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
import matplotlib.pyplot as plt

us_states = gpd.read_file('../input/united-states-state-shapes/us-states.json')
shootings_by_state = shootings.groupby('State').count().join(us_states.set_index('NAME10'))
shootings_by_state = gpd.GeoDataFrame(shootings_by_state.loc[:, ['Location', 'geometry']])
shootings_by_state = shootings_by_state.rename(columns={'Location': 'Incidents'})
shootings_by_state = shootings_by_state.drop(['Alaska', 'Hawaii'])

gplt.choropleth(shootings_by_state, hue='Incidents', 
                projection=gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5),
                k=None, cmap='YlOrRd', figsize=(16, 10), legend=True)
plt.gca().set_ylim((-1647757.3894385984, 1457718.4893930717))
pass


# ## Further ideas
# 
# That's all here folks! There are a large number of fields in this dataset that I haven't even touched or haven't explored very deeply in this highly impressionistic EDA. There's lots more that can be done. For example, to trek onwards, try to look at the following things:
# 
# * Are designated "mass shooting events" getting more lethal over time?
# * What kind of summary information about the motives of shootings can you gather from the `Description` and other text fields?
# * What kinds of guns are used in shootings?
# * Who are the targetted victims?
# 
# To trek onwards, perhaps try out some of the following things:
# 
# Compare this dataset against the Boston AirBnB data. What significant differences are there between the US and Indian homesharing networks (besides the presence of hotels), and can you quantify them?
# Can you mine the textual description fields, omitted here, to see what kinds of words most commonly appear in listing descriptions? What can you learn from applying NLP to these entries?
