#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load the Data

# In[ ]:


df = pd.read_csv('/kaggle/input/ufc-fights-2010-2020-with-betting-odds/data.csv')


# ### Let's take a quick look at the data and do some cleaning

# In[ ]:


df.info(verbose=True)


# date will be a lot more useful to us if it is of type datetime

# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# ***
# Let's remove any blank data

# In[ ]:


df = df.dropna()


# In[ ]:


df.info(verbose=True)


# We have removed about 1300 rows and converted date
# ***

# ### Let's take a peek at the features one-by-one

# In[ ]:


df[['R_fighter', 'B_fighter']].describe()


# 
# We can see that Donald Cerrone and Charles Oliveira have fought in a lot of fights, but we won't be sure of how many until we combine the R_fighter and B_fighter columns because fighters can appear in both columns
# ***

# In[ ]:


df[['date']].describe()


# 
# From the date column we can see that the first fight we are looking at occurred on March 21st, 2010. The most recent fight occurred on March 14, 2020.
# ***

# In[ ]:


df[['R_odds', 'B_odds']].describe()


# From the odds column we can see the largest favorite was -1700. The largest underdog was +1300
# ***

# In[ ]:


df[['location']].describe()


# Las Vegas has had the most fights, although fights have taken place in 144 different locations
# ***

# In[ ]:


#There is a blank space problem that causes 2 countries to be counted twice....
df['country'] = df['country'].str.strip()
display(df[['country']].describe())
display(df['country'].unique())


# 
# Fights have taken place in 25 different countries. Over half of the fights have occurred in the USA.
# ***

# In[ ]:


print(df['Winner'].describe())
print()
print(df['Winner'].unique())


# Red has won 2430 / 4240 fights. The three values of 'Winner' are 'Red', 'Blue', or 'Draw'
# ***

# In[ ]:


print(df['title_bout'].describe())


# 4026 out of 4240 fights have NOT been title fights
# ***

# In[ ]:


print(df['weight_class'].describe())
print()
print(df['weight_class'].unique())


# There are 13 weight classes covering the men's and women's divisions. 'Lightweight' has had the most fights
# ***
# 

# In[ ]:


print(df['gender'].describe())


# 3854 of the 4240 fights have been in the male divisions
# ***

# ### Fights by Year

# In[ ]:


year_labels = []
for z in range(2010, 2021):
    year_labels.append(z)
    
fight_counts = []
for z in (year_labels):
    fight_counts.append(len(df[df['date'].dt.year==z]))


# In[ ]:


plt.figure(figsize=(9,5))
plt.plot(year_labels, fight_counts)
plt.xlabel('Year', fontsize=16)
plt.ylabel('# of Fights', fontsize=16)
plt.title('Fights Per Year', fontweight='bold', fontsize=16)
plt.show()


# Since about 2014 there have been around 400 to 500 fights per year
# ***

# In[ ]:


female_fight_counts = []
for z in (year_labels):
    female_fight_counts.append(len(df[(df['date'].dt.year==z) & (df['gender']=='FEMALE')])) 
#print(female_fight_counts)

plt.figure(figsize=(9,5))
plt.plot(year_labels, female_fight_counts)
plt.xlabel('Year', fontsize=16)
plt.ylabel('# of Fights', fontsize=16)
plt.title('Female Fights Per Year', fontweight='bold', fontsize=16)
plt.show()


# The first female fight occurred in 2013 and the number of female fights has consistently risen year-by-year 
# ***

# ### Let's add an underdog column to the original dataframe. This will be helpful going forward
# 

# In[ ]:


df['underdog'] = ''

red_underdog_mask = df['R_odds'] > df['B_odds']
#print(red_underdog_mask)
#print()

blue_underdog_mask = df['B_odds'] > df['R_odds']
#print(blue_underdog_mask)
#print()

even_mask = (df['B_odds'] == df['R_odds'])
#print(even_mask)
#print()

df['underdog'][red_underdog_mask] = 'Red'
df['underdog'][blue_underdog_mask] = 'Blue'
df['underdog'][even_mask] = 'Even'


# ***
# ### How common are upsets?

# Let's explore how common upsets are.  First let's remove fights where the fighters are even from the dataframe.  We will also remove fights that end in a draw.  This should not be a very large number.

# In[ ]:


df_no_even = df[df['underdog'] != 'Even']
df_no_even = df_no_even[df_no_even['Winner'] != 'Draw']
print(f"Number of fights including even fights and draws: {len(df)}")
print(f"Number of fights with even fights and draws removed: {len(df_no_even)}")


# In[ ]:


number_of_fights = len(df_no_even)
number_of_upsets = len(df_no_even[df_no_even['Winner'] == df_no_even['underdog']])
number_of_favorites = len(df_no_even[df_no_even['Winner'] != df_no_even['underdog']])
#print(number_of_upsets)
#print(number_of_fights)
#print(number_of_favorites)
upset_percent = (number_of_upsets / number_of_fights) * 100
favorite_percent = (number_of_favorites / number_of_fights) * 100
#print(upset_percent)
#print(favorite_percent)
labels = 'Favorites', 'Underdogs'
sizes = [favorite_percent, upset_percent]
fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14})


# Favorites win about 65.5% of the time.  Let's take a deeper dive into this information.
# ***

# ### How much does the upset rate vary year-to-year?

# In[ ]:


year_labels
year_fight_counts = []
year_upset_counts = []
year_upset_percent = []

for y in year_labels:
    temp_fights = df_no_even[df_no_even['date'].dt.year==y]
    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    year_fight_counts.append(len(temp_fights))
    year_upset_counts.append(len(temp_upsets))
    year_upset_percent.append(len(temp_upsets)/len(temp_fights))
    
#print(year_fight_counts)
#print()
#print(year_upset_counts)
#print()
#print(year_upset_percent)

year_upset_percent = [x*100 for x in year_upset_percent]

plt.figure(figsize=(9,5))
barlist = plt.bar(year_labels, year_upset_percent)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Percent of Upset Winners", fontsize=16)
plt.xticks(year_labels, rotation=90)
plt.title('Upset Percentage By Year', fontweight='bold', fontsize=16)
barlist[10].set_color('black')
barlist[3].set_color('grey')


# In[ ]:


temp_df = pd.DataFrame({"Percent of Underdog Winners": year_upset_percent},
                      index=year_labels)

fig, ax = plt.subplots(figsize=(4,8))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0)
plt.title("Upset Percentage by Year", fontsize=16, fontweight='bold')


# Underdogs win between 30.9% and 39.8%. The best year for underdogs is 2020 so far. The worst year was 2013.
# ***

# ### How do upsets vary by weight class?

# In[ ]:


#weight_class_list = df['weight_class'].unique()
#We are manually going to enter the weight class list so we can enter it in order of lightest to heaviest.
weight_class_list = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 
                     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", 
                    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight", "Catch Weight"]
wc_fight_counts = []
wc_upset_counts = []
wc_upset_percent = []

for wc in weight_class_list:
    temp_fights = df_no_even[df_no_even['weight_class']==wc]
    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    wc_fight_counts.append(len(temp_fights))
    wc_upset_counts.append(len(temp_upsets))
    wc_upset_percent.append(len(temp_upsets)/len(temp_fights))

#print(weight_class_list)
#print()
#print(wc_fight_counts)
#print()
#print(wc_upset_counts)
#print()
wc_upset_percent = [x*100 for x in wc_upset_percent]    
#print(wc_upset_percent)
plt.figure(figsize=(9,5))
barlist = plt.bar(weight_class_list, wc_upset_percent)
plt.xlabel("Weight Class", fontsize=16)
plt.ylabel("Percent of Upset Winners", fontsize=16)
plt.xticks(weight_class_list, rotation=90)
plt.title('Upset Percentage By Weight Class', fontweight='bold', fontsize=16)
barlist[9].set_color('black')
barlist[11].set_color('grey')


# In[ ]:


temp_df = pd.DataFrame({"Percent of Underdog Winners": wc_upset_percent},
                      index=weight_class_list)

fig, ax = plt.subplots(figsize=(4,8))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0)
plt.title("Upset Percentage by Weight Class", fontsize=16, fontweight='bold')


# 
# Upset Percentage varies from 28.6% for Women's Featherweight to 39.5% for Women's Flyweight.
# ***

# ### How do Upsets vary by Gender?

# In[ ]:


gender_list = df['gender'].unique()
gender_fight_counts = []
gender_upset_counts = []
gender_upset_percent = []

for g in gender_list:
    temp_fights = df_no_even[df_no_even['gender']==g]
    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    gender_fight_counts.append(len(temp_fights))
    gender_upset_counts.append(len(temp_upsets))
    gender_upset_percent.append(len(temp_upsets)/len(temp_fights))
    
plt.figure(figsize=(9,5))
barlist = plt.bar(gender_list, gender_upset_percent)
plt.xlabel("Gender", fontsize=16)
plt.ylabel("Percent of Upset Winners", fontsize=16)
plt.xticks(gender_list, rotation=90)
plt.title('Upset Percentage By Gender', fontweight='bold', fontsize=16)


# 
# 1. The upset percentage for male and females is almost identical. Male underdogs win 34.46% of the time. Female underdogs win 34.50% of the time.
# ***

# ### Are upsets more common in title fights?
# 

# In[ ]:


title_list = df['title_bout'].unique()
title_fight_counts = []
title_upset_counts = []
title_upset_percent = []

for t in title_list:
    temp_fights = df_no_even[df_no_even['title_bout']==t]
    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    title_fight_counts.append(len(temp_fights))
    title_upset_counts.append(len(temp_upsets))
    title_upset_percent.append(len(temp_upsets)/len(temp_fights))
    
#print(title_list)
#print()
#print(title_fight_counts)
#print()
#print(title_upset_counts)
#print()
#title_upset_percent = [x*100 for x in title_upset_percent]    
#print(title_upset_percent)    

plt.figure(figsize=(9,5))
barlist = plt.bar(['Non-Title', 'Title'], title_upset_percent)
plt.xlabel("Bout Status", fontsize=16)
plt.ylabel("Percent of Upset Winners", fontsize=16)
plt.xticks(['Non-Title', 'Title'])
plt.title('Upset Percentage By Title Bout', fontweight='bold', fontsize=16)


# Upsets are slightly more likely in non-title bouts. They occur 34.6% of the time compared to 32.2% of the time in title fights
# ***

# ### Are upsets more likely in certain weight class title bouts?
# 
# Here we are starting to stray into an area where sample sizes may be too small for some weight classes. I still think this is worth exploring though to see if any anomalies stand out.
# 

# In[ ]:


df_title = df_no_even[df_no_even['title_bout']==True]
weight_class_list = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 
                     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", 
                    "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"]

wc_fight_counts = []
wc_upset_counts = []
wc_upset_percent = []

for wc in weight_class_list:
    temp_fights = df_title[df_title['weight_class']==wc]
    temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    wc_fight_counts.append(len(temp_fights))
    wc_upset_counts.append(len(temp_upsets))
    wc_upset_percent.append(len(temp_upsets)/len(temp_fights))

#print(weight_class_list)
#print()
#print(wc_fight_counts)
#print()
#print(wc_upset_counts)
#print()
wc_upset_percent = [x*100 for x in wc_upset_percent]    
#print(wc_upset_percent)


# In[ ]:


plt.figure(figsize=(9,5))
barlist = plt.bar(weight_class_list, wc_upset_percent)
plt.xlabel("Weight Class", fontsize=16)
plt.ylabel("Percent of Upset Winners", fontsize=16)
plt.xticks(weight_class_list, rotation=90)
plt.title('Title Fight Upset Percentage By Weight Class', fontweight='bold', fontsize=16)
barlist[7].set_color('black')


# In[ ]:


temp_df = pd.DataFrame({"Percent of Underdog Winners": wc_upset_percent, 
                        "Number of Fights": wc_fight_counts},
                      index=weight_class_list)

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0, fontsize=12)
plt.title("Title Fight Upset Percentage by Weight Class", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)


# 
# In the five title fights in UFC Women's Flyweight history there has never been an upset. Out of 19 title fights in heavyweight history 47.37% of them have been won by the underdog.
# ***

# ### Who has the most upsets since 2010?
# 

# In[ ]:


red_fighter_list = df_no_even['R_fighter'].unique()
blue_fighter_list = df_no_even['B_fighter'].unique()
fighter_list = list(set(red_fighter_list) | set(blue_fighter_list))
upset_list = []

for f in fighter_list:
    temp_fights = df_no_even[(df_no_even['R_fighter']==f) | (df_no_even["B_fighter"]==f)]

    #Filter out fights where the fighter is not the winner.
    temp_fights = temp_fights[((temp_fights['R_fighter']==f) & (temp_fights['Winner']=='Red')) |
                             ((temp_fights['B_fighter']==f) & (temp_fights['Winner']=='Blue'))]
    
    
    #Filter out the fights where our hero is not the underdog.
    temp_fights = temp_fights[((temp_fights['R_fighter']==f) & (temp_fights['underdog']=='Red')) |
                             ((temp_fights['B_fighter']==f) & (temp_fights['underdog']=='Blue'))]
    
    
    upset_list.append(len(temp_fights)) 
    
    #print(temp_upset_count)
    #print(temp_fights)
    #print(f"{f}: {len(temp_fights)}")
    #temp_upsets = temp_fights[temp_fights['Winner'] == temp_fights['underdog']]
    #wc_fight_counts.append(len(temp_fights))
    #wc_upset_counts.append(len(temp_upsets))
    #wc_upset_percent.append(len(temp_upsets)/len(temp_fights))

#Zip the two lists into a dataframe
upset_tuples = list(zip(fighter_list, upset_list))
upset_df = pd.DataFrame(upset_tuples, columns=['fighter', 'upset_count'])
upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)
display(upset_df.head(8))


# Eight fighters have seven or more upsets since 2010
# ***

# ### What events have had the most upsets since 2010?

# In[ ]:


#It is possible that there are 2 events that occur in the same day, but there would not be two events on the same day 
#in the same location.

#event_list = df_no_even['date'].unique()

event_df = df_no_even[['date', 'location']]
event_df = event_df.drop_duplicates()

event_array = event_df.values
upset_list = []
date_list = []
location_list = []
for e in event_array:
    temp_event = df_no_even[(df_no_even['date']==e[0]) & (df_no_even["location"]==e[1])]
    #Temp event now has all of the fights in the array
    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]
    #print(len(temp_fights))
    upset_list.append(len(underdog_df)) 
    date_list.append(e[0])
    location_list.append(e[1])
    
#print(len(upset_list))
#print(len(event_array))
upset_tuples = list(zip(location_list, date_list, upset_list))
upset_df = pd.DataFrame(upset_tuples, columns = ['location', 'date', 'upset_count'])
upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)
display(upset_df.head(9))


# 9 Events have had 8 or more upsets. It is worth noting that they all take place between May 2012 and October 2016. 
# ***

# ### Let's do a deeper dive into locations and how often upsets occur there. Let's start on the country level and then break it down into cities.

# In[ ]:


country_list = df_no_even['country'].unique()
#print(country_list)
upset_list = []
upset_per_list = []
for c in country_list:
    temp_event = df_no_even[(df_no_even['country']==c)]
    #Temp event now has all of the fights in the array
    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]
    #print(len(temp_fights))
    underdog_count = len(underdog_df)
    fight_count = len(temp_event)
    upset_per_list.append((underdog_count) / (fight_count) * 100)     
    upset_list.append(underdog_count) 
upset_tuples = list(zip(country_list, upset_list, upset_per_list))
upset_df = pd.DataFrame(upset_tuples, columns=['country', 'upset_count', 'upset_per'])
upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)
#print(upset_list)
#print(country_list)


# In[ ]:


plt.figure(figsize=(15,6))
barlist = plt.bar(upset_df['country'], upset_df['upset_count'])
plt.xlabel("Country", fontsize=16)
plt.ylabel("Number of Upsets", fontsize=16)
plt.xticks(upset_df['country'], rotation=90)
plt.title('Upset Count by Country', fontweight='bold', fontsize=16)
barlist[0].set_color('black')
barlist[24].set_color('grey')


# In[ ]:


temp_df = upset_df.set_index('country')

fig, ax = plt.subplots(figsize=(4,8))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0)
plt.title("Upset Count By Country", fontsize=16, fontweight='bold')


# Not super informative by itself.  Let's take a look at the upset percentage by country.

# In[ ]:


plt.figure(figsize=(15,6))
barlist = plt.bar(upset_df['country'], upset_df['upset_per'])
plt.xlabel("Country", fontsize=16)
plt.ylabel("Upset Percentage", fontsize=16)
plt.xticks(upset_df['country'], rotation=90)
plt.title('Upset Percentage by Country', fontweight='bold', fontsize=16)
barlist[16].set_color('black')
barlist[23].set_color('grey')


# In[ ]:


temp_df = upset_df[['country', 'upset_per']]
temp_df = temp_df.set_index('country')

fig, ax = plt.subplots(figsize=(4,8))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0)
plt.title("Upset Percentage By Country", fontsize=16, fontweight='bold')


# 
# This is more interesting. The countries are still arranged buy upset count, but here you can see the upset percentages. As there are less fights the percentages start to vary by more. This makes sense. There is still considerable amount of variance at the top of the list. The United Kingdom shows a significant lack of upsets. Australia is the only country in the top 12 by upset count to have an upset percentage over 40%.

# ***
# 

# ### Let's Look at it City-by_City

# In[ ]:


location_list = df_no_even['location'].unique()
#print(len(location_list))
upset_list = []
upset_per_list = []
for l in location_list:
    temp_event = df_no_even[(df_no_even['location']==l)]
    #Temp event now has all of the fights in the array
    underdog_df = temp_event[((temp_event['Winner'] == temp_event['underdog']))]
    #print(len(temp_fights))
    underdog_count = len(underdog_df)
    fight_count = len(temp_event)
    upset_per_list.append((underdog_count) / (fight_count) * 100)     
    upset_list.append(underdog_count) 
upset_tuples = list(zip(location_list, upset_list, upset_per_list))
upset_df = pd.DataFrame(upset_tuples, columns=['location', 'upset_count', 'upset_per'])
upset_df = upset_df.sort_values(by=['upset_count'], ascending=False)
#print(upset_list)
#print(country_list)
#display(upset_df)


# In[ ]:


plt.figure(figsize=(10,30))
plt.grid(axis='x')
barlist = plt.barh(upset_df['location'], upset_df['upset_count'])
plt.xlabel("Number of Upsets", fontsize=16)
plt.ylabel("Location", fontsize=16)
plt.yticks(upset_df['location'])
plt.title('Upset Count by Location', fontweight='bold', fontsize=16)
barlist[0].set_color('black')
barlist[-1].set_color('grey')


# In[ ]:


plt.figure(figsize=(10,30))
plt.grid(axis='x')
barlist = plt.barh(upset_df['location'], upset_df['upset_per'])
plt.xlabel("Upset Percentage", fontsize=16)
plt.ylabel("Location", fontsize=16)
plt.yticks(upset_df['location'])
plt.title('Upset Count by Location', fontweight='bold', fontsize=16)
barlist[33].set_color('black')
barlist[-1].set_color('grey')


# In[ ]:


temp_df = upset_df[['location', 'upset_per']]
temp_df = temp_df.set_index('location')

fig, ax = plt.subplots(figsize=(8,30))
sns.heatmap(temp_df, annot=True, fmt=".4g", cmap='binary', ax=ax)
plt.yticks(rotation=0)
plt.title("Upset Percentage By Location", fontsize=16, fontweight='bold')


# Brazil has two locations that have had over 90% upsets: Porto Alegre, Rio Grande do Sul and Natal, Rio Grande do Norte.  Albany, New York, USA is the only location to never have an upset.

# ***
# ### Largest Upsets since 2010
# 

# In[ ]:


underdog_win_df = df_no_even[(df_no_even['Winner'] == df_no_even['underdog'])].copy()
underdog_win_df['winner_odds'] = underdog_win_df[['B_odds', 'R_odds']].values.max(1)
underdog_win_df = underdog_win_df.sort_values(by=['winner_odds'], ascending=False)
underdog_display = underdog_win_df[['R_fighter', 'B_fighter', 'weight_class', 'date', 'Winner', 'winner_odds']]

display(underdog_display.head(10))


# The three largest upsets of the last 10 years all were in the Bantamweight division, and they all occurred over a period of 9 months.

# In[ ]:




