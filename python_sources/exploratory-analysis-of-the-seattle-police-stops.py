#!/usr/bin/env python
# coding: utf-8

# # Seattle Terry Stops Analysis
# In this dataset, we'll analyse Seattle's Terry Stops.
# 
# This data represents records of police reported stops under Terry v. Ohio, 392 U.S. 1 (1968). Each row represents a unique stop. 
# 
# - Each record contains perceived demographics of the subject, as reported by the officer making the stop and officer demographics as reported to the Seattle Police Department, for employment purposes. 
# 
# - Where available, data elements from the associated Computer Aided Dispatch (CAD) event (e.g. Call Type, Initial Call Type, Final Call Type) are included.
# 
# This dataset contains the following data:
# - Subject Age Group: Subject Age Group (10 year increments) as reported by the officer.
# - Subject ID: Key, generated daily, identifying unique subjects in the dataset using a character to character match of first name and last name. "Null" values indicate an "anonymous" or "unidentified" subject. Subjects of a Terry Stop are not required to present identification.
# - GO / SC Num: General Offense or Street Check number, relating the Terry Stop to the parent report. This field may have a one to many relationship in the data.  
# - Terry Stop ID: Key identifying unique Terry Stop reports.
# - Stop Resolution: Resolution of the stop as reported by the officer.
# - Weapon Type: Type of weapon, if any, identified during a search or frisk of the subject. Indicates "None" if no weapons was found.
# - Officer ID: Key identifying unique officers in the dataset.
# - Officer YOB: Year of birth, as reported by the officer.
# - Officer Gender: Gender of the officer, as reported by the officer.
# - Officer Race: Race of the officer, as reported by the officer.
# - Subject Perceived Race: Perceived race of the subject, as reported by the officer.
# - Subject Perceived Gender: Perceived gender of the subject, as reported by the officer.
# - Reported Date: Date the report was filed in the Records Management System (RMS). Not necessarily the date the stop occurred but generally within 1 day.
# - Reported Time: Time the stop was reported in the Records Management System (RMS). Not the time the stop occurred but generally within 10 hours.
# - Initial Call Type: Initial classification of the call as assigned by 911.
# - Final Call Type: Final classification of the call as assigned by the primary officer closing the event.
# - Call Type: How the call was received by the communication center.
# - Officer Squad: Functional squad assignment (not budget) of the officer as reported by the Data Analytics Platform (DAP).
# - Arrest Flag: Indicator of whether a "physical arrest" was made, of the subject, during the Terry Stop. Does not necessarily reflect a report of an arrest in the Records Management System (RMS).
# - Frisk Flag: Indicator of whether a "frisk" was conducted, by the officer, of the subject, during the Terry Stop.
# - Precinct: Precinct of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.
# - Sector: Sector of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.
# - Beat: Beat of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.
# 
# Please bear in mind that not all the data was used, so there are some columns that were not used for this visualization.

# # Summary
# * [1. Data Cleaning](#1.-Initial-Data-Cleaning)
# * [2. Analysis](#2.-Analysis)
#     * [2.1. Age Comparison](#2.1.-Age-comparison)
#     * [2.2. Stop Resolution Comparison](#2.2.-Stop-Resolution-Comparison)
#     * [2.3. Stops bt Officer](#2.3.-Stops-by-officer)
#     * [2.4. Weapons by the first 10 Officers](#2.4.-Weapons-by-the-first-10-Officers)
#     * [2.5. The Top 5 Weapon Types of our Entire Dataframe](#2.5.-The-Top-5-Weapon-Types-of-our-Entire-Dataframe)
#     * [2.6. Weapons by Race](#2.6.-Weapons-by-Race)
#         * [2.6.1. Multi-Racial People Guns](#2.6.1.-Multi-Racial-People-Guns)
#         * [2.6.2. Black People Guns](#2.6.2.-Black-People-Guns)
#         * [2.6.3. White People Guns](#2.6.3.-White-People-Guns)
#         * [2.6.4. American Indian / Alaskan Native People Guns](#2.6.4.-American-Indian-/-Alaskan-Native-People-Guns)
#         * [2.6.5. Asias People Guns](#2.6.5.-Asias-People-Guns)
#         * [2.6.6. Hispanic People Guns](#2.6.6.-Hispanic-People-Guns)
#         * [2.6.7. Other People Guns](#2.6.7.-Other-People-Guns)
#     * [2.7. What Officer Gender is the most usual?](#2.7.-What-Officer-Gender-is-the-most-usual?)
#         * [2.7.1. Is there a relationship between Officer Gender and Weapons found?](#2.7.1.-Is-there-a-relationship-between-Officer-Gender-and-Weapons-found?)
#         * [2.7.2. Is there a relationship between Officer Gender and Stop Resolution?](#2.7.2.-Is-there-a-relationship-between-Officer-Gender-and-Stop-Resolution?)
#     * [2.8. What is the reprentation of the first 5 squads?](#2.8.-What-is-the-reprentation-of-the-first-5-squads?)
#         * [2.8.1. Stop occurrences by Officer Squad](#2.8.1.-Stop-occurrences-by-Officer-Squad)
#         * [2.8.2. Percentage of Stops by Seattle Precinct](#2.8.2.-Percentage-of-Stops-by-Seattle-Precinct)
#         * [2.8.3. Occurences by Sector](#2.8.3.-Occurences-by-Sector)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,10)


# In[ ]:


df = pd.read_csv('../input/terry-stops.csv')


# In[ ]:


df.head()


# # 1. Initial Data Cleaning
# Data cleaning is the name of the process that we manage to see what kind of data we should have or not on our dataframe. When we are working with datasets, it's important to clean the data we have to make our analysis more meaningful.
# We'll follow the following steps:
# * Check if we have null values or nonsense values and drop it
# * Remove columns that are not necessary for our analysis
# * Rename columns names

# In[ ]:


df.info()


# In[ ]:


df.isnull().any()


# As we can see, we have many columns that contains null values. Let's drop it.

# In[ ]:


df = df.dropna()
df.isnull().any()


# As we can see now, all the lines have no null value. We'll keep cleaning the data, but in a next moment.
# Now, let's rename our columns.

# In[ ]:


df.columns


# In[ ]:


df.columns = ['Subject_Age_Group', 'Subject_ID', 'GO_SC_Num', 'Terry_Stop_ID',
       'Stop_Resolution', 'Weapon_Type', 'Officer_ID', 'Officer_YOB',
       'Officer_Gender', 'Officer_Race', 'Subject_Perceived_Race',
       'Subject_Perceived_Gender', 'Reported_Date', 'Reported_Time',
       'Initial_Call_Type', 'Final_Call_Type', 'Call_Type', 'Officer_Squad',
       'Arrest_Flag', 'Frisk_Flag', 'Precinct', 'Sector', 'Beat']
df.columns


# Nice! The columns are renamed. Now, let's delete the columns that are not necessary for us.

# In[ ]:


df[:20]


# In[ ]:


del df['Subject_ID']
del df['GO_SC_Num']


# # 2. Analysis
# ## 2.1. Age comparison
# In the first comparison, we will understand how old are the people that are being stopped and how many people from each category.

# In[ ]:


filter_age = df['Subject_Age_Group'] != '-'
df_filter_age = df[filter_age]
x = df_filter_age['Subject_Age_Group'].value_counts().index
y = df_filter_age['Subject_Age_Group'].value_counts()

fig, ax = plt.subplots()
fig.set_size_inches(15, 7)

graph_age = sns.barplot(x=x, 
            y=y, 
            order=['1 - 17', '18 - 25', '26 - 35', '36 - 45', '46 - 55', '56 and Above'] )
graph_age.set(ylabel = 'Quantity of Stopped People', 
                          xlabel = 'Age Range', 
                          title = 'Stops by Age')
plt.show()


# ## 2.2. Stop Resolution Comparison

# In[ ]:


df.Stop_Resolution.unique()


# In[ ]:


# filter_stop_resolution = df['Stop_Resolution'] != '-'
# df_filter_stop_resolution = df[filter_stop_resolution]
# x_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts().index
# y_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts()

# fig, ax = plt.subplots()
# fig.set_size_inches(15, 7)
# graph_stop_resolution = sns.barplot(x=x_df_filter_stop_resolution, y=y_df_filter_stop_resolution)
# graph_stop_resolution.set(ylabel = 'Number of Stops', 
#                           xlabel = 'Resolution Type', 
#                           title = 'Seattle Terry Stops Resolution',)
# plt.show()


# In[ ]:


# I didn't remove the string value '-' (hifen) but for this analysis, I've suppressed it with this filter
filter_stop_resolution = df['Stop_Resolution'] != '-'
# Here I'm applying our dataframe using the filter to another variable, that will be a new dataframe
df_filter_stop_resolution = df[filter_stop_resolution]
# Here you can see that I'm retrieving the indexes of the Stop_Resolution column
y_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts().index
# Here we have the values for each Stop_Resolution
x_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts()

# Now, let's create a pie chart because I think it's easier for us to understand what is happening
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
graph_stop_resolution = ax.pie(x=x_df_filter_stop_resolution, 
                               labels=y_df_filter_stop_resolution,
                               autopct='%1.1f%%')

ax.set_title('Stop Resolution Comparison')

plt.show()


# ## 2.3. Stops by officer
# Now, let's see what are the first 10 officers that stopped more people.

# In[ ]:


len(df['Officer_ID'].unique()), len(df['Officer_ID'])


# In[ ]:


officer_counts = df['Officer_ID'].value_counts()
df_officer_counts = pd.DataFrame(officer_counts)

df_officer_counts_slice = df_officer_counts[:10]

x_counts = df_officer_counts_slice['Officer_ID'].index
y_counts = df_officer_counts_slice['Officer_ID']

fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
graph_officer_counts_ten = sns.barplot(x=x_counts, y=y_counts, data=df_officer_counts_slice, palette='winter_r')


# ## 2.4. Weapons by the first 10 Officers
# Now, from the officers we saw above, we'll see what are the weapons that they find.

# In[ ]:


officers_ids = officer_counts[:10].index
officers_ids


# In[ ]:


df_officer_ids_weapons = df.loc[df['Officer_ID'].isin(officers_ids)]
df_officer_ids_weapons.head()


# In[ ]:


filter_officer_ids_weapons = (df_officer_ids_weapons['Weapon_Type'] != '-') & (df_officer_ids_weapons['Weapon_Type'] != 'None')
filter_officer_ids_weapons.any()


# In[ ]:


df_officer_ids_weapons_filtered = df_officer_ids_weapons[filter_officer_ids_weapons]

sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_weapons = df_officer_ids_weapons_filtered['Weapon_Type'].value_counts()
y_weapons = df_officer_ids_weapons_filtered['Weapon_Type'].value_counts().index

graph_weapons_officers = ax.pie(x=x_weapons,
                                labels=y_weapons, 
                                autopct='%1.1f%%',
                                pctdistance=0.8)

for weapon in graph_weapons_officers[0]:
    weapon.set_edgecolor('black')
plt.show()


# ## 2.5. The Top 5 Weapon Types of our Entire Dataframe
# Above we saw the weapons that our first 10 officers found. Now, let's see what are the weapons for all our dataframe.

# In[ ]:


filter_total_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None')
filter_total_weapons.any()


# In[ ]:


df_total_weapons = df[filter_total_weapons]

# Before we go ahead, I'll fix some Weapon Types to make it easier for us.
df_total_weapons = df_total_weapons.replace({'Blackjack':'Club, Blackjack, Brass Knuckles', 
                                             'Brass Knuckle':'Club, Blackjack, Brass Knuckles',
                                             'Club':'Club, Blackjack, Brass Knuckles',
                                             'Firearm Other':'Firearm', 'Firearm (unk type)':'Firearm'})

max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(15,15)

x_total_weapons = df_total_weapons['Weapon_Type'].value_counts()[:max_weapon_on_chart]
y_total_weapons = df_total_weapons['Weapon_Type'].value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 15})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')


# As a result, we could see that Seattle Officers' findings are mainly Lethal Cutting Instruments - approximately 3/4.

# ## 2.6. Weapons by Race
# Let's see if we can found a pattern of guns by race

# In[ ]:


filter_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None') 
filter_race = (df.Subject_Perceived_Race != 'Unknown') & (df.Subject_Perceived_Race != '-')

df[filter_weapons & filter_race].Subject_Perceived_Race.unique()


# In[ ]:


df_weapons_race = df[filter_weapons & filter_race]

df_weapons_race.Subject_Perceived_Race.unique()


# In[ ]:


filter_Multi_Racial = df_weapons_race.Subject_Perceived_Race == 'Multi-Racial'
filter_Black = df_weapons_race.Subject_Perceived_Race == 'Black'
filter_White = df_weapons_race.Subject_Perceived_Race == 'White'
filter_AIAN = df_weapons_race.Subject_Perceived_Race == 'American Indian / Alaskan Native'
filter_Asian = df_weapons_race.Subject_Perceived_Race == 'Asian'
filter_Hispanic = df_weapons_race.Subject_Perceived_Race == 'Hispanic'
filter_Other = df_weapons_race.Subject_Perceived_Race == 'Other'


# ### 2.6.1. Multi-Racial People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Multi_Racial].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Multi_Racial].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ### 2.6.2. Black People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Black].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Black].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ### 2.6.3. White People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_White].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_White].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# In[ ]:


filter_AIAN = df_weapons_race.Subject_Perceived_Race == 'American Indian / Alaskan Native'
filter_Asian = df_weapons_race.Subject_Perceived_Race == 'Asian'
filter_Hispanic = df_weapons_race.Subject_Perceived_Race == 'Hispanic'
filter_Other = df_weapons_race.Subject_Perceived_Race == 'Other'


# ### 2.6.4. American Indian / Alaskan Native People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_AIAN].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_AIAN].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ### 2.6.5. Asias People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Asian].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Asian].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ### 2.6.6. Hispanic People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Hispanic].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Hispanic].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ### 2.6.7. Other People Guns

# In[ ]:


max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Other].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Other].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()


# ## 2.7. What Officer Gender is the most usual?

# In[ ]:


df.Officer_Gender.unique().tolist()


# In[ ]:


sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_officers_genders = df.Officer_Gender.value_counts()
y_officers_genders = df.Officer_Gender.value_counts().index

graph_officers_gender = ax.pie(x=x_officers_genders, labels=y_officers_genders, autopct='%1.2f%%')

plt.show()


# ### 2.7.1. Is there a relationship between Officer Gender and Weapons found?

# In[ ]:


filter_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None') 


# In[ ]:


filter_female = df['Officer_Gender'] == 'F'
filter_male = df['Officer_Gender'] == 'M'

df_female_weapons = df[(filter_female) & (filter_weapons)]
df_male_weapons = df[(filter_male) & (filter_weapons)]

sns.set_palette('Reds_r')
fig, ax = plt.subplots(1,2)
fig.set_size_inches(17,8)

x_female_weapons = df_female_weapons.Weapon_Type.value_counts()[:5]
y_female_weapons = df_female_weapons.Weapon_Type.value_counts().index[:5]

x_male_weapons = df_male_weapons.Weapon_Type.value_counts()[:5]
y_male_weapons = df_male_weapons.Weapon_Type.value_counts().index[:5]

graph_female_weapons = ax[0].pie(x=x_female_weapons, labels=y_female_weapons, autopct='%1.2f%%')
graph_male_weapons = ax[1].pie(x=x_male_weapons, labels=y_male_weapons, autopct='%1.2f%%')

ax[0].set_title('Female Officer Weapon Found')
ax[1].set_title('Male Officer Weapon Found')

plt.show()


# ### 2.7.2. Is there a relationship between Officer Gender and Stop Resolution?

# In[ ]:


filter_male = df['Officer_Gender'] == 'M'
filter_stop_resolutions = df.Stop_Resolution != '-'

filter_female = df['Officer_Gender'] == 'F'

df_male_weapons = df[(filter_male) & (filter_stop_resolutions)]

df_female_weapons = df[(filter_female) & (filter_stop_resolutions)]

sns.set_palette('Reds_r')
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(17,8)

x_male_weapons = df_male_weapons.Stop_Resolution.value_counts()[:5]
y_male_weapons = df_male_weapons.Stop_Resolution.value_counts().index[:5]

x_female_weapons = df_female_weapons.Stop_Resolution.value_counts()[:5]
y_female_weapons = df_female_weapons.Stop_Resolution.value_counts().index[:5]

graph_female_weapons = ax[1].pie(x=x_female_weapons, labels=y_female_weapons, autopct='%1.2f%%')
graph_male_weapons = ax[0].pie(x=x_male_weapons, labels=y_male_weapons, autopct='%1.2f%%')

ax[0].set_title('Female Officer Stop Resolution')
ax[1].set_title('Male Officer Stop Resolution')

plt.show()


# ## 2.8. What is the reprentation of the first 5 squads?

# In[ ]:


#chart config
sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_squads = df.Officer_Squad.value_counts()[:5]
labels_squads = df.Officer_Squad.value_counts().index[:5]

graph_squads = ax.pie(x=x_squads, labels=labels_squads, autopct='%1.2f%%')

plt.show()


# ### 2.8.1. Stop occurrences by Officer Squad

# In[ ]:


sns.set_palette('Blues')
fig, ax = plt.subplots()
fig.set_size_inches(15,12)

x_squads = df.Officer_Squad.value_counts().index[:20]
y_squads = df.Officer_Squad.value_counts()[:20]

graph_squads = sns.barplot(x=x_squads, y=y_squads, data=df )

for item in graph_squads.get_xticklabels():
    item.set_rotation(90)

plt.show()


# ### 2.8.2. Percentage of Stops by Seattle Precinct

# In[ ]:


filter_squad_precinct = (df.Officer_Squad.isin(df.Officer_Squad.value_counts()[:20].index.tolist())) & (df.Precinct != 'Unknown')


# In[ ]:


df_squads_precinct = df[filter_squad_precinct]

#chart config
sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_squads_precinct = df_squads_precinct['Precinct'].value_counts()
labels_squads_precinct = df_squads_precinct['Precinct'].value_counts().index

graph_squads_precinct = ax.pie(x=x_squads_precinct, labels=labels_squads_precinct, autopct='%1.2f%%')

for item in graph_squads_precinct[0]:
    item.set_edgecolor('white')

plt.show()


# In[ ]:


df.head(2)


# ### 2.8.3. Occurences by Sector

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15,10)

x_sector = df.Sector.value_counts().index
y_sector = df.Sector.value_counts()

graph_sectors = sns.barplot(x=x_sector, y=y_sector, data=df)

# for label in graph_sectors.get_xticklabels():
#     label.set_rotation(45)

plt.show()


# # Thank You!
# Hello guys! This is my **first data analysis**, so feel free to report errors, reviews or if you think I did something wrong, please do let me know.
# 
# If you have new ideas, do let me know and then we can create more visualizations on this kernel.
