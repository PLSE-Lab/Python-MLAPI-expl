#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import os
import folium
import pandas as pd
import numpy as np
import missingno as mn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ![](https://upload.wikimedia.org/wikipedia/commons/5/51/Fentanyl-2D-skeletal-A.png)
# 
# # Drug deaths in Connecticut: A story of growing opiod abuse
# 
# This kernel explores the Kaggle data set [Drug Related Deaths](https://www.kaggle.com/adarsh4u/drug-related-deaths), which appears to be for accidental deaths in the state of Connecticut over the last 6 years. 
# 
# Despite being a morbid topic, I think the trends and relationship's within its features are still very interesting. I attempt to look at the following visulisations:
# 
# 
# ## Contents
# 
# 1. [Load and check data](#load)
# 2. [Where do the most deaths occur](#where)
# 3. [Deaths by drug type](#type)
# 4. [Drug trends with time](#time)
# 5. [Demographic trends](#demo)
# 6. [Injury locations](#loc)
# 7. [Conclusions](#conclusions)

# ### Load and check data <a name="load"></a>
# 
# The data set is provided as a single .csv so straightforward to load.

# In[ ]:


data_path = os.path.join(os.path.pardir, 'input', 'Accidental_Drug_Related_Deaths__2012-2017 (1).csv')
df = pd.read_csv(data_path)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]  # standardise column names
df.head()


# Process the drug columns into binary flags (1, 0) to make them more readable and much more usable. Unfortunately this data is quite dirty and the positive flags seem to take a number of forms (e.g. 'Y', 'y', ' Y') or can inlcude some extra data. I will only include the obvious positive flags for now.
# 
# Also, create an extra single column (essentially un-pivot) of the drug type responsible for each death. This will make some visualisations much easier.

# In[ ]:


drug_types = ['heroin', 'cocaine', 'fentanyl', 'oxycodone', 'oxymorphone', 'etoh',
              'hydrocodone', 'benzodiazepine', 'methadone', 'amphet', 'tramad',
               'morphine_(not_heroin)', 'any_opioid']

positive_flag_types = ['Y', 'y', ' Y', '1']

df['drug_type'] = 'other'
for drug in drug_types:
    df.loc[~df[drug].isin(positive_flag_types), drug] = 0
    df.loc[df[drug].isin(positive_flag_types), drug] = 1
    df[drug] = df[drug].astype(np.int8)
    df.loc[df[drug] == 1, 'drug_type'] = drug


# Lets look at all the columns that are present, and at the same time how many null values they contain.

# In[ ]:


mn.matrix(df)


# After processing the drug columns above, there are few nulls there which is good as this will help with most of the analysis. Some of the less useful columns are *'amendedmannerofdeath'* and *'descriptionofinjury'*, so I may not be able to do much with them.

# ### Where do the deaths occur? <a name="where"></a>
# 
# Usefully, the *deathloc* columns contains latitude and longitude. This means we can use the excellent **folium** package to visualise where the deaths occur on a map.
# 
# **Note** I have coloured the markers by the drug type, and you can click on a marker to show the type.

# In[ ]:


df['deathloc_latitude'] = df['deathloc'].str.extract(r'(\d+\.\d+)', expand=True).values.astype(np.float32)
df['deathloc_longitude'] = -df['deathloc'].str.split(' -').str[1].str[:-1].astype(np.float32)


# In[ ]:


# Create map around the mean position
central_position = [df['deathloc_latitude'].mean(), df['deathloc_longitude'].mean()]
locations_map = folium.Map(location=central_position, zoom_start = 9)

# Colors for the different drug types
i = 0
pal = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 
       'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

# Add markers to the map according to their drug type
for drug in df['drug_type'].unique():
    
    drug_df = df[df['drug_type'] == drug]
    
    # Not interested in adding markers for 'other'
    if drug == 'other':  
        continue
        
    for case in drug_df.index[:30]:
        folium.Marker([drug_df.loc[case, 'deathloc_latitude'], drug_df.loc[case, 'deathloc_longitude']],
                       popup=drug_df.loc[case, 'drug_type'],
                       icon=folium.Icon(color=pal[i], icon='circle', prefix='fa')
                     ).add_to(locations_map)
    i += 1
locations_map


# The highest densities are unsurprisingly in the cities, and there doesn't seem to be any obvious trends in drug type by area.

# ### Deaths by drug type <a name="type"></a>
# 
# Which drugs are the most damaging? Do a simple count over the whole data set to investigate.

# In[ ]:


deaths_by_drug = df[drug_types].sum().sort_values(ascending=False)
fig, ax = plt.subplots(1, 1, figsize=[7, 5])
sns.barplot(x=deaths_by_drug, y=deaths_by_drug.index)
ax.set_xlabel('Total deaths over 6 years')
ax.set_title('Accidental deaths by drug type')
plt.show()


# Looks like heroin is responsible for the most accidental drug deaths in this data set, befitting its reputation as a deeply harmful substance. Fentanyl, another potent opiod, comes in second with cocaine coming in third presumably as a result of its widespread use.

# ### Drug trends with time <a name="time"></a>

# First, make the date feature into a pandas datetime and extract some extra detail.

# In[ ]:


df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['year'] = df['date'].dt.year
df['year_month'] = df['date'].dt.to_period('M')


# So what does the general trend with time look like?

# In[ ]:


annual_deaths = df.groupby('year_month')['date'].count()

fig, ax = plt.subplots(1, 1, figsize=[10, 5])
lr = LinearRegression().fit(pd.to_numeric(annual_deaths.index).values.reshape(-1, 1), 
                            annual_deaths.values.reshape(-1, 1))
trendline = lr.predict(pd.to_numeric(annual_deaths.index).values.reshape(-1, 1))
annual_deaths.plot(ax=ax, marker='o', ls='-', alpha=.9, markersize=5, color='r', label='Monthly deaths')
ax.plot(annual_deaths.index, trendline, ls=':', color='k', label='Trendline')
ax.set_ylabel('Total deaths')
ax.set_xlabel('Time')
ax.set_title('Annual accidental drug deaths')
ax.legend()
plt.show()


# So am alarming trend that indicates that accidental drug deaths doubled from 2013-2017. Lets dig a little deeper and see if there are any particular drug types driving this trend.

# In[ ]:


time_trends_by_drug = df.groupby(by=['year'])[drug_types].sum()

#TODO ROTATE PLOT
fig, ax = plt.subplots(1, 1, figsize=[17, 5])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data=time_trends_by_drug, square=True, cmap=cmap, center=0,
            linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
plt.show()


# This plot reveals that heroin and cocaine are the main death drivers and have generally been steadily getting worse with time. Fentanyl, which has been grabbing a lot of headlines recently, has clearly exploded from nowhere to become 2017's biggest cause of accidental death.

# ### Demographics of deceased <a name="demo"></a>
# 
# The data set contains some demographic information on the deceased, in particular:
# 
# - Age
# - Race
# - Gender
# 
# It will probably be interesting to see how these categories factor in to the number of accidental deaths, and if there are any leanings to particular drugs.
# 
# Let's start by looking at the age profiles of deceased users of the most frequently occuring drugs.

# In[ ]:


most_frequent_drugs = deaths_by_drug[:5].index
frequent_drugs_df = df[df['drug_type'].isin(most_frequent_drugs)]


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=[11, 6])
sns.violinplot(x='drug_type', y='age', hue='sex', data=frequent_drugs_df, ax=ax, split=True)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Drug type')
ax.set_title('Distribution of deceased\'s age by drug type and gender')
plt.show()


#  - Benzodiazepine shows quite a bit of different by age and sex - it appears to affect younger men and older women
#  - The most deaths from cocaine in both genders is by people aged around 50, despite it's reputation as a party drug
#  - On the other, hand heroin appears to strike early with bost genders showing a peak in the mid 20's.

# A caveat on the above data - about 75% of the dates are male. This is plotted below, and whilst we're at it lets do the same thing for race. This reveals that the vast majority of deaths are from white people, with the only other significant numbers present in mixed race white/hispanic and black people.

# In[ ]:


gender_counts = df['sex'].value_counts()
race_counts = df['race'].value_counts()
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])
sns.barplot(x=gender_counts, y=gender_counts.index, ax=ax)
sns.barplot(x=race_counts, y=race_counts.index, ax=ax1)
ax.set_title('Total deaths by gender')
ax.set_xlabel('Total deaths')
ax1.set_title('Total deaths by race')
ax1.set_xlabel('Total deaths')
plt.tight_layout()
plt.show()


# Despite the imbalance of the data, is there any variance in the age of death by race?

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=[8, 5])
for race in df['race'].unique():
    if sum(df['race'] == race) > 100:
        sns.distplot(df.loc[df['race'] == race, 'age'], hist=False, label=race, ax=ax)
ax.set_title('Age distributions of deceased by race')
ax.autoscale()
ax.legend()
plt.show()


# Interestingly, there is a large peak for white people at around 30 that isn't matched at all by the other races displayed here, who have peaks much later on in life between 40-60. Again though, there is not much data on the non-white races so can't be too conclusive.

# Are there any demographic trends in drug preference? For these plots I will stick to the 5 drugs responsible for the most deaths to simplify.

# In[ ]:


deaths_by_gender = deaths_by_drug.copy()
for gender in ['Male', 'Female']:
    temp_gender_df = df.loc[df['sex'] == gender, drug_types].sum().sort_values(ascending=False)
    temp_gender_df = 100 * temp_gender_df / sum(temp_gender_df)  # Change to a percentage
    deaths_by_gender = pd.DataFrame(deaths_by_gender).join(pd.DataFrame(temp_gender_df), rsuffix=gender)


# In[ ]:


deaths_by_race = deaths_by_drug.copy()
for race in ['White', 'Hispanic, White', 'Black']:
    temp_race_df = df.loc[df['race'] == race, drug_types].sum().sort_values(ascending=False)
    temp_race_df = 100 * temp_race_df / sum(temp_race_df)  # Change to a percentage
    deaths_by_race = pd.DataFrame(deaths_by_race).join(pd.DataFrame(temp_race_df), rsuffix=race)


# In[ ]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(14, 5))

N = len(deaths_by_gender.index[:5])
ind = np.arange(N)
width = 0.35
ax.bar(x=ind, 
       height=deaths_by_gender.iloc[:5, 1].values, 
       width=width/2, 
       label='Male')
ax.bar(x=ind + width/2, 
       height=deaths_by_gender.iloc[:5, 2].values, 
       width=width/2, 
       label='Female')
ax.set_xticks(ind)
ax.set_xticklabels(deaths_by_gender.index[:5])
ax.set_ylabel('Proportion of all deaths (%)')
ax.set_title('Deaths by drug split by gender')
ax.legend()

ax1.bar(x=ind, 
       height=deaths_by_race.iloc[:5, 1].values, 
       width=width/2, 
       label='White')
ax1.bar(x=ind + width/2, 
       height=deaths_by_race.iloc[:5, 2].values, 
       width=width/2, 
       label='White, Hispanic')
ax1.bar(x=ind + width, 
       height=deaths_by_race.iloc[:5, 3].values, 
       width=width/2, 
       label='Black')
ax1.set_xticks(ind)
ax1.set_xticklabels(deaths_by_race.index[:5])
ax1.set_ylabel('Proportion of all deaths (%)')
ax1.set_title('Deaths by drug split by race')
ax1.legend()


plt.show()


# So is **does** look like demographics influence the drug types used. Just a few observations I can make from these plots:
# 
# - Heroin and fentanyl are proportionally more responsible for death amongst men, whilst benzodiazepine is more responsible for women
# - Black people are more susceptible to cocaine than the other two races considered here, whilst white poeple are more susceptible to benzodiazepine

# ### Injury locations <a name="loc"></a>
# 
# Where do the injuries take place usually? This data has high cardinality so can be tricky to visualise, but is still iteresting in terms of the variety of different places these events can happen.
# 
# Lets plot the top 20:

# In[ ]:


injury_places = df['injuryplace'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=[8, 10])
sns.barplot(x=injury_places[:20], y=injury_places[:20].index)
plt.show()


# Vast majority occur at home or another residence, and the alternatives are quite spread out with hotes/motels coming in as the second definite location.

# ### Conclusions <a name="conclusions"></a>
# 
# - Opiod abuse is growing quickyly in Connecticut, driven by a rapid growth in Fentanyl and steady growth in heroin.
# - White males constitute the largest number of accidental drug deaths in general, and partuclarly in opiods.
# - Most deaths occur in urban, densely populated areas within residencies.
