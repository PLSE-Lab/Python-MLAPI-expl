#!/usr/bin/env python
# coding: utf-8

# # Global Terrorism
# ## A Data Exploration in Proportions

# In order to get a better overview of terrorism, we first have to investigate how it is conducted. For this, I will analyze terrorist attacks between 1970 and 2016 and try to understand
# * **where** they take place (country, specific location),
# * **when** they occur (date, location change over time),
# * **how** they are conducted (bombing, assault, kidnapping, etc.),
# * **who** is responsible (e.g., the Islamic State, Boko Haram, PKK, etc.) and
# * **what** they target (e.g., police stations, civilians, soldiers, etc.).
# 
# While this topic and these questions have been dealt with many times in the past, most analyses focus on simple descriptions of absolute numbers, e.g., number terrorist attacks per country, but do not put these results into contrast with other measures, e.g., graveness of these terrorist attacks. I believe, that a thorough investigation of global terrorism can only be performed by combining the questions listed above. Furthermore, I try to gain new insights into the topic by comparing absolute numbers with proportions.

# # 1 Data Preparation

# ## 1.1 Load Dataset

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.basemap import Basemap

pd.set_option('display.max_columns', None)

path = '../input/globalterrorismdb_0617dist.csv'

dtype = {
    'approxdate': object,
    'resolution': object,
    'attacktype2_txt': object,
    'attacktype3_txt': object,
    'targsubtype3_txt': object,
    'gsubname2': object,
    'gname3': object,
    'gsubname3': object,
    'claimmode2_txt': object,
    'claimmode3_txt': object,
    'weaptype3_txt': object,
    'weapsubtype3_txt': object,
    'weaptype4_txt': object,
    'weapsubtype4_txt': object,
    'divert': object,
    'kidhijcountry': object,
    'ransomnote': object,
}

df = pd.read_csv(path, encoding='latin1', dtype=dtype)


# ## 1.2 Get an Overview of the Dataset

# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df[['nkill', 'nwound']].describe()


# In[ ]:


df[['country_txt', 'gname', 'attacktype1_txt', 'weaptype1_txt', 'targtype1_txt']].describe()


# The dataset is very well structured and maintained. There are no missing values in the primary columns eventid, iyear, imonth and iday. One row represents one terrorist attack.

# # 2 Data Exploration

# ## 2.1 Where and When: Terrorist Attacks By Country and Year

# In[ ]:


def country_year_heatmap(country_year, vmax, figsize=(15, 10)):
    # get region name for every country
    country_region = df[['country_txt', 'region_txt']].drop_duplicates()
    region_order = [
        'Middle East & North Africa',
        'Sub-Saharan Africa',
        'South Asia',
        'Southeast Asia',
        'Western Europe',
        'Eastern Europe',
        'North America',
        'Central America & Caribbean',
        'South America'
    ]

    # sort by number of kills and limit to 40
    country_year['total'] = country_year.sum(axis=1)
    country_year = country_year.sort_values(['total'], ascending=False)[0:40]

    # sort by region order
    country_year = country_year.join(country_region.set_index('country_txt'))
    country_year_sorted = pd.DataFrame()
    region_count = []
    for region in region_order:
        country_year_sorted = pd.concat([
            country_year_sorted,
            country_year[country_year['region_txt'] == region]
        ])
        if len(country_year[country_year['region_txt'] == region]) > 0:
            region_count.append(len(country_year_sorted['region_txt'].index))

    # remove columns used for sorting
    del country_year_sorted['total']
    del country_year_sorted['region_txt']

    # show heatmap
    country_year_sorted = country_year_sorted.fillna(0)
    f, ax = plt.subplots(figsize=figsize)
    if len(region_count) > 1:
        for count in region_count[0:-1]:
            ax.axhline(count)
    g = sns.heatmap(country_year_sorted, cmap='Reds', linewidths=1, vmax=vmax)
    ax.set_ylabel('')
    plt.show()


# ### 2.1.1 Number of Terrorist Attacks per Country per Year

# In[ ]:


country_year = df.groupby(['country_txt','iyear'])['eventid'].count().unstack()
country_year_heatmap(country_year, vmax=3000)


# This heatmap shows the number of terrorist attacks per country per year, limited to 40 countries with the highest number of terrorist attacks. Countries are grouped by region. On the first glance, we can see clusters for the past ten years in Iraq, Pakistan, Afghanistan, India, Phillipines and others. We would expect each of these countries to show up for a number of reasons. For instance, some countries are struck by extremist groups fighting over power, e.g., Iraq and Afghanistan, other countries are struck by civil war, e.g., Ukraine, and some countries are attacked by religious groups, e.g., the Philippines.
# 
# However, after careful investigation, one can notice, that some conflicts in the past seem underrepresented or are missing entirely, and several single events are missing. One example for a terrorist attack series, that is underrepresented in this visualization, are the attacks commited by the Red Army Faction in Germany between 1970 and 1990. An example for missing events are the September 11 attacks in the United States.
# 
# Simply presenting this visualization will, without doubt, suggest the user that terrorism is becoming worse. If only the number of terrorist attacks is taken into consideration, this is true. However, there are other metrics that have to considered in order to show a balanced picture. One possibility for this would be to factor in the number of casualties. This is done in the next step.
# 
# Furthermore, they might be other reasons for the increased number of terrorist attacks in the dataset. One explanation could be changes in data collection. For instance, it is possible, that in the past, several terrorist incidents in close proximity have been recorded as a single attack and today they are recorded individually.
# 
# Another problem with this visualization is that the high number of terrorist attacks in Iraq leads to many other countries disappearing in the background. To counter this issue, the maximum number of terrorist attacks per year has been limited to 3000.

# ### 2.1.2 Number of Casualties per Terrorist Attack per Country per Year

# In[ ]:


country_year = df.groupby(['country_txt','iyear'])['nkill'].sum().unstack() /                df.groupby(['country_txt','iyear'])['nkill'].count().unstack()
country_year_heatmap(country_year, vmax=200)


# Now we can clearly see especially where and when severe terrorist attack have occured. Severe in that context means the number of casualties per terrorist attack. For instance, we can see the September 11 Attacks in the United States. However, this visualization skews the picture into another direction. Now single severe events are overrepresented. Iraq is almost not visible in the past ten years. European countries are missing entirely.

# ### 2.1.3 Number of Terrorist Attacks per Country in Western Europe per Year

# In[ ]:


country_year = df[df['region_txt'].isin(['Western Europe'])].groupby(['country_txt','iyear'])['nkill'].count().unstack()
country_year_heatmap(country_year, vmax=250, figsize=(15, 6))


# This visualization shows, that a full picture cannot be provided without going into detail. The first heatmap suggests, that there is almost no terrorism in Western Europe. While there are much fewer incidents than, for instance, in the Middle East, there definitile is terrorism in Western Europe as well. However, in global comparison, it is reduced to background noise.
# 
# Nevertheless, Western Europe has been struck my local terrorism for many years. This visualization shows terrorist attacks by separatist movements, such as the IRA in the United Kingdom and the ETA in Spain and France.

# ## 2.2 How: Terrorist Attack Types, Weapons and Targets

# ### 2.2.1 Casualties per Terrorist Attack per Attack Type

# In[ ]:


attack_types = df.groupby(['attacktype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types.columns = ["_".join(x) for x in attack_types.columns.ravel()]
attack_types['killed per attack'] = attack_types['nkill_sum'] / attack_types['nkill_count']
attack_types['wounded per attack'] = attack_types['nwound_sum'] / attack_types['nkill_count']
attack_types = attack_types.sort_values(['killed per attack'], ascending=True)

# draw plot
ax = attack_types[['wounded per attack', 'killed per attack']].plot.barh(width=0.85, color=sns.color_palette("Paired"))
ax.set_ylabel('')
fig = plt.gcf()
fig.set_size_inches(8, 10)
plt.show()


# This visualization shows the casualties, killed and wounded, per terrorist attack per attack type. Hijacking seems to be the deadliest form of terrorist attack. Before drawing any conclusions, we want to further split the data into suicide attacks and non-suicide attacks.

# ### 2.2.2 Casualties per Terrorist Attack per Attack Type with and without Suicide Attacks

# In[ ]:


attack_types = df[df['suicide'] == 0].groupby(['attacktype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types.columns = ["_".join(x) for x in attack_types.columns.ravel()]
attack_types_suicide = df[df['suicide'] == 1].groupby(['attacktype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types_suicide.columns = ["_".join(x) for x in attack_types_suicide.columns.ravel()]
attack_types['killed per attack without suicide'] = attack_types['nkill_sum'] / attack_types['nkill_count']
attack_types['killed per attack with suicide'] = attack_types_suicide['nkill_sum'] / attack_types_suicide['nkill_count']
attack_types['wounded per attack without suicide'] = attack_types['nwound_sum'] / attack_types['nkill_count']
attack_types['wounded per attack with suicide'] = attack_types_suicide['nwound_sum'] / attack_types_suicide['nkill_count']
attack_types = attack_types.fillna(0)
attack_types = attack_types.sort_values(['killed per attack with suicide'], ascending=True)

# draw plots
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.55)
colors = ['#EDB120', '#D95319']
ax1 = attack_types[['wounded per attack without suicide', 'killed per attack without suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[0])
ax1.set_ylabel('')
ax1.yaxis.set_ticks_position('right')
ax1.set_yticklabels(labels=attack_types.index, ha='center')
ax1.tick_params(axis='y', direction='out', pad=95)
colors = ['#a987ff', '#7E2F8E']
ax2 = attack_types[['wounded per attack with suicide', 'killed per attack with suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[1])
ax2.set_ylabel('')
ax2.set_yticklabels('')
fig.set_size_inches(16, 10)
plt.show()


# Splitting the casualties per terrorist attack per attack type into suicide and non-suicide attacks highlights the severity of hijacking. However, we believe that this representation is distorted due to the high number of casualties in the September 11 attacks. In order to get a clearer picture, we will filter them out in a next step.

# ### 2.2.3 Casualties per Terrorist Attack per Attack Type with and without Suicide Attacks and September 11 Attacks Filtered Out

# In[ ]:


# filter out September 11 attacks
september_eleven = df[(df['iyear'] == 2001) & (df['imonth'] == 9) & (df['iday'] == 11) & (df['country_txt'] == 'United States')]
attack_types_filtered = pd.concat([df, september_eleven]).drop_duplicates(keep=False)

attack_types = attack_types_filtered[attack_types_filtered['suicide'] == 0].groupby(['attacktype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types.columns = ["_".join(x) for x in attack_types.columns.ravel()]
attack_types_suicide = attack_types_filtered[(attack_types_filtered['suicide'] == 1)].groupby(['attacktype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types_suicide.columns = ["_".join(x) for x in attack_types_suicide.columns.ravel()]
attack_types['killed per attack without suicide'] = attack_types['nkill_sum'] / attack_types['nkill_count']
attack_types['killed per attack with suicide'] = attack_types_suicide['nkill_sum'] / attack_types_suicide['nkill_count']
attack_types['wounded per attack without suicide'] = attack_types['nwound_sum'] / attack_types['nkill_count']
attack_types['wounded per attack with suicide'] = attack_types_suicide['nwound_sum'] / attack_types_suicide['nkill_count']
attack_types = attack_types.fillna(0)
attack_types = attack_types.sort_values(['killed per attack with suicide'], ascending=True)

# draw plots
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.55)
colors = ['#EDB120', '#D95319']
ax1 = attack_types[['wounded per attack without suicide', 'killed per attack without suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[0])
ax1.set_ylabel('')
ax1.yaxis.set_ticks_position('right')
ax1.set_yticklabels(labels=attack_types.index, ha='center')
ax1.tick_params(axis='y', direction='out', pad=95)
colors = ['#a987ff', '#7E2F8E']
ax2 = attack_types[['wounded per attack with suicide', 'killed per attack with suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[1])
ax2.set_ylabel('')
ax2.set_yticklabels('')
fig.set_size_inches(16, 10)
plt.show()


# Filtering out the September 11 attacks allows for an easier comparison of suicide attacks and non-suicide attacks. Here we can see that hostage taking incidents in connection with attacker suicides show a very high number of casualties per terrorist attack.

# ### 2.2.4 Casualties per Terrorist Attack per Weapon Type with and without Suicide Attacks and September 11 Attacks Filtered Out

# In[ ]:


# filter out September 11 attacks
september_eleven = df[(df['iyear'] == 2001) & (df['imonth'] == 9) & (df['iday'] == 11) & (df['country_txt'] == 'United States')]
attack_types_filtered = pd.concat([df, september_eleven]).drop_duplicates(keep=False)

attack_types = attack_types_filtered[attack_types_filtered['suicide'] == 0].groupby(['weaptype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types.columns = ["_".join(x) for x in attack_types.columns.ravel()]
attack_types_suicide = attack_types_filtered[(attack_types_filtered['suicide'] == 1)].groupby(['weaptype1_txt']).agg({'nkill': ['sum', 'count'], 'nwound': ['sum']})
attack_types_suicide.columns = ["_".join(x) for x in attack_types_suicide.columns.ravel()]
attack_types['killed per attack without suicide'] = attack_types['nkill_sum'] / attack_types['nkill_count']
attack_types['killed per attack with suicide'] = attack_types_suicide['nkill_sum'] / attack_types_suicide['nkill_count']
attack_types['wounded per attack without suicide'] = attack_types['nwound_sum'] / attack_types['nkill_count']
attack_types['wounded per attack with suicide'] = attack_types_suicide['nwound_sum'] / attack_types_suicide['nkill_count']
attack_types = attack_types.fillna(0)
attack_types = attack_types.sort_values(['killed per attack with suicide'], ascending=True)

# draw plots
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.55)
colors = ['#EDB120', '#D95319']
ax1 = attack_types[['wounded per attack without suicide', 'killed per attack without suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[0])
ax1.set_ylabel('')
ax1.yaxis.set_ticks_position('right')
labels = attack_types.index.tolist()
labels[8] = 'Vehicle'
ax1.set_yticklabels(labels=labels, ha='center')
ax1.tick_params(axis='y', direction='out', pad=95)
colors = ['#a987ff', '#7E2F8E']
ax2 = attack_types[['wounded per attack with suicide', 'killed per attack with suicide']].plot.barh(width=0.9, color=sns.color_palette(colors), ax=axes[1])
ax2.set_ylabel('')
ax2.set_yticklabels('')
fig.set_size_inches(16, 10)
plt.show()


# By comparing the number of casualties per terrorist attack per weapon type with and without suicide attacks and September 11 Attacks filtered out, we can see the impact of chemical weapons on the number of wounded. Another finding is the inverse proportion of killed and wounded by firearms and explosives. While terrorist attacks conducted with firearms result in a higher death toll than number of injuries, for terrorist attacks conducted with explosives this relation is inverted.

# ### 2.2.5 Weapon Type per Attack Type

# In[ ]:


weapontypes_attacktypes = pd.crosstab(df['attacktype1_txt'], df['weaptype1_txt'])
# for i in range(0, len(weapontypes_attacktypes.index)):
#     weapontypes_attacktypes.iloc[i] = weapontypes_attacktypes.iloc[i] / weapontypes_attacktypes.iloc[i].sum()

weapontypes_attacktypes['total'] = weapontypes_attacktypes.sum(axis=1)
weapontypes_attacktypes = weapontypes_attacktypes.sort_values(['total'], ascending=False)[1:41]
weapontypes_attacktypes = weapontypes_attacktypes.sort_values(['total'], ascending=True)
del weapontypes_attacktypes['total']

colors = sns.color_palette("Set1", 12)
colors[9] = (0.749, 0.961, 0.239)
colors[10] = (0.152, 0.937, 0.86)
colors[11] = (0.663, 0.586, 0.953)

ax = weapontypes_attacktypes.plot.barh(stacked=True, width=1, color=colors)
ax.set_ylabel('')
labels = weapontypes_attacktypes.columns.tolist()
labels[11] = 'Vehicle'
ax.legend(labels=labels)
fig = plt.gcf()
fig.set_size_inches(12, 4.5)
plt.show()


# Here we can see the total number of terrorist attacks by attack type and main weapon usage. While it clearly shows that firearms are the most used weapon type, we also want to have a clearer view on the proportion of weapon type used for each attack type. For this, we will normalize this data.

# ### 2.2.6 Proportion of Weapon Types Used per Attack Type

# In[ ]:


for i in range(0, len(weapontypes_attacktypes.index)):
    weapontypes_attacktypes.iloc[i] = weapontypes_attacktypes.iloc[i] / weapontypes_attacktypes.iloc[i].sum()

ax = weapontypes_attacktypes.plot.barh(stacked=True, width=1, color=colors)
ax.set_ylabel('')
labels = weapontypes_attacktypes.columns.tolist()
labels[11] = 'Vehicle'
ax.legend(labels=labels, loc=6, bbox_to_anchor=(1, 0.5))
fig = plt.gcf()
fig.set_size_inches(10, 4.5)
plt.show()


# From this visualization we can gather, that facility attacks are conducted with incendiaries, not with explosives. Another finding is that some hijackings have been executed with fake weapons. Regarding the structure of the dataset, we can see that biological and chemical weapons are subjected to unarmed assaults.

# ### 2.2.7 Proportion of Weapon Types used in Terrorist Attacks per Year

# In[ ]:


weapon_types = df.groupby(['weaptype1_txt', 'iyear'])['nwound'].count().unstack()
weapon_types = weapon_types.fillna(0)
for i in weapon_types.columns:
    weapon_types[i] = weapon_types[i] / weapon_types[i].sum()
f, ax = plt.subplots(figsize=(15, 3.5))
g = sns.heatmap(weapon_types, cmap='Reds', linewidths=1)
ax.set_xlabel('')
ax.set_ylabel('')
labels = weapon_types.index.tolist()
labels[11] = 'Vehicle'
ax.set_yticklabels(labels)
plt.show()


# Here we can see that terrorist attacks conducted with explosives have been prevalent most in most years between 1970 and 2016. In some years, however, firearms were seen more frequently than explosives. The trend in the past 20 years shows an increase in the use of explosives.

# ### 2.2.8 Proportion of Weapon Types used in Terrorist Attacks per Year in North America

# In[ ]:


weapon_types = df[df['region_txt'] == 'North America'].groupby(['weaptype1_txt', 'iyear'])['nwound'].count().unstack()
weapon_types = weapon_types.fillna(0)
for i in weapon_types.columns:
    weapon_types[i] = weapon_types[i] / weapon_types[i].sum()
f, ax = plt.subplots(figsize=(15, 3.5))
g = sns.heatmap(weapon_types, cmap='Reds', linewidths=1)
ax.set_xlabel('')
ax.set_ylabel('')
labels = weapon_types.index.tolist()
labels[11] = 'Vehicle'
ax.set_yticklabels(labels)
plt.show()


# This visualization shows that the use of explosives in North America has diminished since 1990. On the other side, terrorist attacks conducted with incendiaries have increased shortly after, becoming the most used weapon type in the early 2000s.

# ### 2.2.9 Proportion of Weapon Types used in Terrorist Attacks per Year in Western Europe

# In[ ]:


weapon_types = df[df['region_txt'] == 'Western Europe'].groupby(['weaptype1_txt', 'iyear'])['nwound'].count().unstack()
weapon_types = weapon_types.fillna(0)
for i in weapon_types.columns:
    weapon_types[i] = weapon_types[i] / weapon_types[i].sum()
f, ax = plt.subplots(figsize=(15, 3.5))
g = sns.heatmap(weapon_types, cmap='Reds', linewidths=1)
ax.set_xlabel('')
ax.set_ylabel('')
labels = weapon_types.index.tolist()
labels[11] = 'Vehicle'
ax.set_yticklabels(labels)
plt.show()


# Compared to North America, Western Europe shows less terrorist attacks conducted with firearms.

# ### 2.2.10 Casualties per Terrorist Attack per Weapon Type per Year in Western Europe

# In[ ]:


weapon_types = df[df['region_txt'] == 'Western Europe'].groupby(['weaptype1_txt', 'iyear'])['nwound'].sum().unstack() / df[df['region_txt'] == 'Western Europe'].groupby(['weaptype1_txt', 'iyear'])['nwound'].count().unstack()
weapon_types = weapon_types.fillna(0)
f, ax = plt.subplots(figsize=(15, 3.5))
g = sns.heatmap(weapon_types, cmap='Reds', linewidths=1)
ax.set_xlabel('')
ax.set_ylabel('')
labels = weapon_types.index.tolist()
labels[11] = 'Vehicle'
ax.set_yticklabels(labels)
plt.show()


# If we take a look at the number of casualties per terrorist attack per weapon type per year in Western Europe, we notice the rise in attacks conducted with vehicles. This becomes even more obvious, when we think of the recent vehicle rampage attacks in 2017 in Paris, London, Barcelona and Stockholm. Terrorist groups have shifted their methods. Building explosives requires knowledge and material and firearms are more difficult to acquire in Western Europe than in most other places. Vehicles, however, are easy to optain and can cause a lot of damage.

# ## 2.3 Who and How: Terrorist Groups and their Attack Types

# ### 2.3.1 Number of Terrorist Attacks per Terrorist Group per Year

# In[ ]:


# get 40 terrorist groups with highest number of terrorist attacks, filter out unknown
terrorist_groups = df.groupby(['gname']).agg({'nkill': ['sum', 'count']})
terrorist_groups.columns = ["_".join(x) for x in terrorist_groups.columns.ravel()]
terrorist_groups = terrorist_groups.sort_values(['nkill_count'], ascending=False)[1:41].index

# get number of attacks by group name by year
groups_year = df[df['gname'].isin(terrorist_groups)].groupby(['gname', 'iyear'])['nkill'].count().unstack()

# sort by total number of attacks and limit to 40
groups_year['total'] = groups_year.sum(axis=1)
groups_year = groups_year.sort_values(['total'], ascending=False)[0:40]

# remove sorting column
del groups_year['total']

groups_year = groups_year.fillna(0)
f, ax = plt.subplots(figsize=(15, 12))
g = sns.heatmap(groups_year, cmap='Reds', linewidths=1, vmax=500)
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()


# This heatmap shows the number of terrorist attacks per terrorist group per year, limited to 40 terrorist groups with the highest number of terrorist attacks. It also shows a terrorist group's years of activity. The number of terrorist attacks committed by unknown assailants far outweighs the number of terrorist attacks commited by known groups. However, they do not help in understanding terrorist group behaviour, so they have been excluded.

# ### 2.3.2 Attack Type per Terrorist Group

# In[ ]:


groups_attacktypes = pd.crosstab(df['gname'], df['attacktype1_txt'])

# order by total number of attacks, limit to 40 and exclude unknown (index 0)
groups_attacktypes['total'] = groups_attacktypes.sum(axis=1)
groups_attacktypes = groups_attacktypes.sort_values(['total'], ascending=False)[1:41]
groups_attacktypes = groups_attacktypes.sort_values(['total'], ascending=True)
del groups_attacktypes['total']
    
ax = groups_attacktypes.plot.barh(stacked=True, width=1, color=sns.color_palette("Set1", 9))
ax.set_ylabel('')
ax.legend()
fig = plt.gcf()
fig.set_size_inches(12,16)
plt.show()


# This visualization shows the type of attacks terrorist groups conduct, limited to 40 terrorist groups with the highest number of terrorist attacks. It can be used to distinguish between terrorist groups' method of operations. However, for this particular task, a normalized view is more suitable, as shown below.

# ### 2.3.3 Proportion of Attack Type per Terrorist Group

# In[ ]:


for i in range(0, len(groups_attacktypes.index)):
    groups_attacktypes.iloc[i] = groups_attacktypes.iloc[i] / groups_attacktypes.iloc[i].sum()
    
ax = groups_attacktypes.plot.barh(stacked=True, width=1, color=sns.color_palette("Set1", 9))
ax.set_ylabel('')
ax.legend(loc=9, bbox_to_anchor=(0.5, -0.025))
fig = plt.gcf()
fig.set_size_inches(12,16)
plt.show()


# Here we can see how the 40 most active terrorist groups operate. It is possible to presume a terrorist groups' goals from this visualization. For instance, the IRA and ETA, two separatist groups, both show a very high percentage of assassinations as compared to other terrorist groups. In their attempt to destabilize governments, they targeted politicians and policy makers. Another example, the Islamic State (ISIL) is using bombings to destroy government installations and disrupt military and community supply. An interesting observation is the realtively low number of hostage taking for most terrorist groups. An exception to this is M19, who were known for financing their operations with ransom money paid for hostages.

# ### 2.3.4 Proportion of Terrorist Groups' Target Types

# In[ ]:


groups_targettypes = pd.crosstab(df['gname'], df['targtype1_txt'])

# order by total number of attacks, limit to 40 and exclude unknown (index 0)
groups_targettypes['total'] = groups_targettypes.sum(axis=1)
groups_targettypes = groups_targettypes.sort_values(['total'], ascending=False)[1:40]
del groups_targettypes['total']

for i in range(0, len(groups_targettypes.index)):
    groups_targettypes.iloc[i] = groups_targettypes.iloc[i] / groups_targettypes.iloc[i].sum()

# order by proportionally most frequent target type
groups_targettypes = groups_targettypes.transpose()
groups_targettypes['total'] = groups_targettypes.sum(axis=1)
groups_targettypes = groups_targettypes.sort_values(['total'], ascending=False)
del groups_targettypes['total']
groups_targettypes = groups_targettypes.transpose()

groups_targettypes = groups_targettypes.fillna(0)
f, ax = plt.subplots(figsize=(8,11))
g = sns.heatmap(groups_targettypes, cmap='Reds', linewidths=1)
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()


# This heatmap shows the 40 most active terrorist groups' proportion of target types. The columns are ordered by overall proportionally most frequent target type. Private citizens, military and police are the top three targets across these 40 terrorist groups. However, there are some exceptions. For instance, the ETA mostly focused on police and business targets.

# ### 2.3.5 Weapon Type per Terrorist Group

# In[ ]:


groups_weapontypes = pd.crosstab(df['gname'], df['weaptype1_txt'])

# order by total number of attacks, limit to 40 and exclude unknown (index 0)
groups_weapontypes['total'] = groups_weapontypes.sum(axis=1)
groups_weapontypes = groups_weapontypes.sort_values(['total'], ascending=False)[1:41]
groups_weapontypes = groups_weapontypes.sort_values(['total'], ascending=True)
del groups_weapontypes['total']

colors = sns.color_palette("Set1", 12)
colors[9] = (0.749, 0.961, 0.239)
colors[10] = (0.152, 0.937, 0.86)
colors[11] = (0.663, 0.586, 0.953)
    
ax = groups_weapontypes.plot.barh(stacked=True, width=1, color=colors)
ax.set_ylabel('')
ax.legend()
fig = plt.gcf()
fig.set_size_inches(12,16)
plt.show()


# This visualization shows the main weapon type terrorist groups use for terrorist attacks, limited to 40 terrorist groups with the highest number of terrorist attacks. Again, a normalized representation of this data should help to make out differences between the groups' methods.

# ### 2.3.6 Proportion of Weapon Typ per Terrorist Group

# In[ ]:


for i in range(0, len(groups_weapontypes.index)):
    groups_weapontypes.iloc[i] = groups_weapontypes.iloc[i] / groups_weapontypes.iloc[i].sum()
    
colors = sns.color_palette("Set1", 12)
colors[9] = (0.749, 0.961, 0.239)
colors[10] = (0.152, 0.937, 0.86)
colors[11] = (0.663, 0.586, 0.953)    

ax = groups_weapontypes.plot.barh(stacked=True, width=1, color=colors)
ax.set_ylabel('')
ax.legend(loc=9, bbox_to_anchor=(0.5, -0.025))
fig = plt.gcf()
fig.set_size_inches(12,16)
plt.show()


# This visualization allows us to compare terrorist groups by their weapon type of choice. For instance, Palestinian extremists are known to resort to melee weapons, such as knives, since firearms are difficult to acquire in Israel. Another example are the Fulani extremists, who almost exclusively use firearms in their terrorist attacks.

# ## 2.4 Who and Where: Terrorist Groups' Area of Operation

# Some terrorist groups operate globally, others locally. In this section we will have a look at some examples for both types.

# In[ ]:


def draw_map(locations, llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, figsize=(16, 8), steps=[10, 50],
             resolution='c', marker_1=[2, 0.05], marker_2=[2, 0.05], marker_3=[10, 0.2], markerfacecolor=False, path=None):
    casualties_1 = locations[(locations['nkill'] >= 0) & (locations.nkill < steps[0])] 
    casualties_2 = locations[(locations['nkill'] >= steps[0]) & locations.nkill <= steps[1]]
    casualties_3 = locations[locations['nkill'] > steps[1]]
    casualties_1_color = '#ffa000'
    casualties_2_color = '#ff6000'
    casualties_3_color = '#c00000'

    # create map projection
    plt.figure(figsize=figsize)
    m = Basemap(projection='cyl', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawcoastlines(linewidth=1, color='#a0a0a0')
    m.drawcountries(linewidth=1, color='#a0a0a0')
    m.drawmapboundary(fill_color='#cdd2d4')
    m.fillcontinents(color='#f5f5f3', lake_color='#cdd2d4')

    # place markers
    if markerfacecolor:
        markerfacecolor_1 = casualties_1_color
        markerfacecolor_2 = casualties_2_color
        markerfacecolor_3 = casualties_3_color
    else:
        markerfacecolor_1 = 'none'
        markerfacecolor_2 = 'none'
        markerfacecolor_3 = 'none'
        
    x, y = m(list(casualties_1['longitude'].astype(float)), list(casualties_1['latitude'].astype(float)))
    m.plot(x, y, 'o', markersize=marker_1[0], alpha=marker_1[1], markeredgecolor=casualties_1_color, markerfacecolor=markerfacecolor_1)
    x, y = m(list(casualties_2['longitude'].astype(float)), list(casualties_2['latitude'].astype(float)))
    m.plot(x, y, 'o', markersize=marker_2[0], alpha=marker_2[1], markeredgecolor=casualties_2_color, markerfacecolor=markerfacecolor_2)
    x, y = m(list(casualties_3['longitude'].astype(float)), list(casualties_3['latitude'].astype(float)))
    m.plot(x, y, 'o', markersize=marker_3[0], alpha=marker_3[1], markeredgecolor=casualties_3_color, markerfacecolor=markerfacecolor_3)

    # add legend
    plt.legend(handles=[mpatches.Patch(color=casualties_1_color, label=' < ' + str(steps[0]) + ' casualties'),
                        mpatches.Patch(color=casualties_2_color, label=str(steps[0]) + ' - ' + str(steps[1]) + ' casualties'),
                        mpatches.Patch(color=casualties_3_color, label=' > ' + str(steps[1]) + ' casualties')],
                        fontsize=10, markerscale = 5)
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# ### 2.4.1 Terrorist Attacks Across the World

# In[ ]:


locations = df
draw_map(locations)


# First of all, we take a look at terrorist attacks across the world. This map shows all recorded terrorist attacks from 1970 to 2016, grouped by the number of casualties. We can see that terrorism happens almost in every populated area. However, there are obvious clusters, for instance, in the Middle East.

# ### 2.4.2 Terrorist Attacks in the Middle East

# In[ ]:


locations = df[df['iyear'] > 2015]
draw_map(locations, 10.5, 41, 22.5, 63.5, resolution='h', marker_1=[6, 0.1], marker_2=[12, 0.1], marker_3=[24, 0.5], markerfacecolor=True)


# Next, we take a close look at the Middle East. Iraq, Syria, Israel and Yemen are heavily affected by terrorist attacks. Oman, on the other hand, has not recorded a single terrorist attack since 1970.

# ### 2.4.3 Terrorist Attacks Conducted by the Taliban

# In[ ]:


locations = df[df['gname'] == 'Taliban']
draw_map(locations)


# The Taliban are an example for a terrorist group that almost exclusively operates withing a single country, i.e., Afghanistan.

# ### 2.4.4 Terrorist Attacks Conducted by the Taliban in and around Afghanistan

# In[ ]:


locations = df[df['gname'] == 'Taliban']
draw_map(locations, 28.5, 39, 58.5, 76, resolution='h', marker_1=[6, 0.1], marker_2=[12, 0.1], marker_3=[24, 0.5], markerfacecolor=True)


# Not until we take a close look we can see that there are also terrorist attacks conducted by the Taliban in Pakistan as well.

# ### 2.4.5 Terrorist Attacks Conducted by AL-Qaida

# In[ ]:


locations = df[df['gname'] == 'Al-Qaida']
draw_map(locations, marker_1=[3, 0.5], marker_2=[6, 0.5], marker_3=[12, 0.5], markerfacecolor=True)


# Al-Qaida, best-known for the September 11 attacks, is an example for a terrorist group that operated globally. Their last attack was recorded in 2011.

# ### 2.4.6 Terrorist Attacks Conducted by the Islamic State

# In[ ]:


locations = df[df['gname'] == 'Islamic State of Iraq and the Levant (ISIL)']
draw_map(locations, marker_1=[3, 0.5], marker_2=[6, 0.5], marker_3=[12, 0.5], markerfacecolor=True)


# The Islamic State is responsible for many terrorist attacks across the world. Unlike other terrorist groups, they have also conducted series of terrorist attacks in places far away from their main area of operation, e.g., in the Philippines.

# ### 2.4.7 Terrorist Attacks Conducted by ETA

# In[ ]:


locations = df[df['gname'] == 'Basque Fatherland and Freedom (ETA)']
draw_map(locations, marker_1=[3, 0.5], marker_2=[6, 0.5], marker_3=[12, 0.5], markerfacecolor=True)


# As a separatist group, ETA's focus was mostly centered around the Basque Country.

# ### 2.4.8 Terrorist Attacks Conducted by ETA in Western Europe

# In[ ]:


locations = df[df['gname'] == 'Basque Fatherland and Freedom (ETA)']
draw_map(locations, 33, 55, -12, 15, resolution='h', marker_1=[6, 0.1], marker_2=[12, 0.1], marker_3=[24, 0.5], markerfacecolor=True)


# In this map section we can clearly distinguish the accumulation of terrorist attacks conducted by ETA in the Basque Country.

# ### 2.4.9 Terrorist Attacks Conducted by Boko Haram

# In[ ]:


locations = df[df['gname'] == 'Boko Haram']
draw_map(locations, marker_1=[3, 0.5], marker_2=[6, 0.5], marker_3=[12, 0.5], markerfacecolor=True)


# Boko Haram is another example for a terrorist group that mainly operates locally.

# ### 2.4.10 Terrorist Attacks Conducted by Boko Haram in and around Nigeria

# In[ ]:


locations = df[df['gname'] == 'Boko Haram']
draw_map(locations, 2, 16, 0, 17, marker_1=[6, 0.1], marker_2=[12, 0.1], marker_3=[24, 0.25], markerfacecolor=True)


# However, a closer look shows, that their attacks are much more violent.
