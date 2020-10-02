#!/usr/bin/env python
# coding: utf-8

# Terrorism. Most of us know what this word means. When we hear this word, we imagine attacks in some third world country beset by bombings and killing. We tell ourselves, "What a tragedy" and go about our day. We don't think about these things because the very notion of terrorism in the Western "civilized" world ("the West") seems nearly impossible. Sure, there are some outliers that happen every now and then, like the 9/11 attack in New York. For the most part however, is terrorism even a problem that affects all parts of the world? How come we don't hear about it if it does? 

# # Is it because terrorism is mainly isolated to certain parts of the world?

# Before we can get started, we should first define what terrorism is as well as the time period for this analysis.  
# 
# For the definition of terrorism, I looked at the definiton linked by the dataset that I am analyzing (the Global Terrorism Database (GTD)), which is:
# > - It must be intentional
# > - It must have some level of violence or immediate threat of violence
# > - It must be done by non-state organizations  
# 
# In addition, it has to include two of the following:  
# > - Must be done to attain a political, economic, religious, or social goal
# > - Must have evidence of intention to convey to others some type of message besides to the immediate victims
# > - Action must be done outside the context of legitimate war activities
# 
# The time period that I am analyzing is from 1996 to 2016. Since we are comparing locations instead of change over time, it is more important to prioritize recent data over a data set collected over a longer period of time for relevancy issues.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


terror = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', low_memory=False,
                    usecols=[0, 1, 2, 8, 10, 11, 12, 26, 27, 29, 35, 41, 82, 98])
terror = terror.rename(columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'country_txt':'country', 'region_txt':'region', 'provstate':'province_or_state', 'attacktype1_txt':'attack_type', 'natlty1_txt':'nationality', 'targtype1_txt':'target', 'weaptype1_txt':'weapon_type', 'nkill':'killed'})
terror['killed'] = terror['killed'].fillna(0).astype(int)
terror = terror[terror['year'] > 1995]

terror.head(3)


# # The Dataset
# I chose to modify the dataset so I am only using the columns that are useful and/or relavant to my analysis. Below are the columns that are being used in my analysis as well as a short description of the columns.
# - The *id* is used to identify individual elements, mostly in aggregating information.
# - The *year* and *month* columns are used to identify the trends that may appear to occur over time or in seasons.
# - The *country*, *region*, *province_or_state*, and *city* columns are used to identify and compare different areas where terrorist attacks have occured.
# - The *success* and *suicide* columns are either 0 or 1, with 0 being false and 1 being true (ie 0 in success is a failed attack and 1 in suicide meaning the attack was a suicide attack)
# - *attack_type* and *weapon_type* describe the type of attacks as well as weapons used in terrorist attacks.
# - *target* and *nationality* describe the objective of the terrorist attacks. For example, a target with business and nationality of France, means that the terrorist attack was directed towards a French business.

# With that out of the way, lets explore the dataset and see if we can find something interesting or trends that appear.

# In[ ]:


x = terror['month']
bins = 12
months = range(1,13)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.grid(linestyle='dashed')
plt.hist(x, bins=bins, normed=True, color='brown')
plt.xlabel('Month of Attack')
plt.xticks(months, months)
plt.ylabel('Probability')
plt.title('Probability of Attack in a Given Month')
plt.show()


# It seems that terrorist attacks aren't affected by the time of the year. Most of the attacks appear to be spread evenly throughout the months. How about if we compare the years? Would there also be a relatively even split of successful terrorist attacks from 1996-2016?

# In[ ]:


x = terror.groupby('year').count()['success']
years = terror['year'].unique()

plt.plot(x)
plt.grid(linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Successful Terrorist Attacks')
plt.title('Successful Terrorist Attacks from 1996-2016')
plt.xticks(years, years, rotation=90)
plt.show()


# Interesting! It appears that the number of terrorist attacks remained relatively flat until 2007. From 2007 onwards, the number of attacks went up significantly, and started declining in 2015.

# Let's pivot here and look at the number of terrorist attacks by region. There is a popular assumption that most of all the world's terrorist attacks occur in the Middle East region, due to the number of popular terrorist organizations (ISIS, Al Qaeda, and Taliban) located there, but is this true?

# In[ ]:


ax = terror['region'].value_counts().plot(kind='barh', color='red')
ax.set_xlabel('Regions')
ax.set_ylabel('Number of Terrorist Attacks')
ax.set_title('Terrorist Attacks in Regions')


# The Middle East has experienced the most terrorist attacks, but what's interesting is that South Asia also appears to ha an issue with terrorist attacks! Why is this the case? **Are there any differences in terrorist attacks in South Asia and the Middle East?**

# In[ ]:


middle_east_region = terror[terror['region'] == 'Middle East & North Africa']
middle_east_y = middle_east_region['country'].value_counts()
middle_east_x = middle_east_y.keys()

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.barh(middle_east_x, middle_east_y)
plt.title('Terrorist Attacks in Middle East and North Africa')
plt.xlabel('Number of Attacks')
plt.ylabel('Countries in Middle East & North Africa Region')

south_asia_region = terror[terror['region'] == 'South Asia']
south_asia_y = south_asia_region['country'].value_counts()
south_asia_x = south_asia_y.keys()
plt.subplot(1,2,2)
plt.barh(south_asia_x, south_asia_y, color='green')
plt.title('Terrorist Attacks in South Asia')
plt.xlabel('Number of Attacks')
plt.ylabel('Countries in South Asia')
plt.tight_layout()
plt.show()
print('The average number of terrorist attacks per country in the Middle East is {0:.0f} while in South Asia the average number of terrorist attacks are {1:.0f}.'.format(middle_east_y.mean(), south_asia_y.mean()))


# As we can see above, the number of terrorist attacks in the Middle East appears to be skewed due to Iraq. In South Asia however, it appears to be split between countries that have low and high terrorist activties. One interesting thing to note here for South Asia is that the countries that experience low terrorist activities are relatively smaller nations compared to the countries that experience a greater amount of terrorist attacks.

# Since the biggest outlying country in the data appears to be Iraq, lets see if we can get even more specific about where the terrorist attacks are occuring.

# In[ ]:


iraq_y = terror[terror['country'] == 'Iraq']['city'].value_counts()
iraq_x = range(iraq_y.count())
plt.scatter(iraq_y, iraq_x)
plt.xlabel('Number of Attacks on Iraqian Cities')
plt.ylabel('Cities')
plt.title('Attacks on Iraqian Cities')
plt.show()
print('In Iraq, {0} has experienced {1} terrorist attacks.'.format(iraq_y.index[0], iraq_y.values[0]))


# There are many cities in Iraq that experience terrorist attacks, but the number of attacks those cities experience appear to be around 1000 or less. Out of all the cities in Iraq, two cities have experienced the brunt of the attacks, with one city having experienced 2000 terrorist attacks and around 7000.

# Just as an interesting tidbit, how would the Middle East region compare to the other regions if we did not consider Iraq in the dataset? Would the Middle East be comparable in terms of the number of terrorist attacks that region experiences?

# In[ ]:


no_iraq = terror[terror['country'] != 'Iraq']
y = no_iraq['region'].value_counts()
x = y.keys()

plt.barh(x, y, color='green')
plt.xlabel('Regions')
plt.ylabel('Number of Terrorist Attacks')
plt.title('Terrorist Attacks in Regions')
plt.show()


# While the number of terrorist attacks in the Middle East is still higher than the other regions sans South Asia, it has dropped dramatically as terrorist attacks in Iraq basically accounts for about half of all the attacks in the Middle East!

# Next, let's check out the weapon and attack types that terrorist use.

# In[ ]:


plt.figure(figsize=(15, 5))
y = terror['weapon_type'].value_counts()
x = y.keys()
plt.subplot(1, 2, 1)
plt.bar(x, y, color='pink')
plt.xlabel('Weapon Types')
plt.ylabel('Number of Attacks')
plt.xticks(rotation = 90)
plt.title('Weapon Types Used by Terrorists')

y = terror['attack_type'].value_counts()
x = y.keys()
plt.subplot(1, 2, 2)
plt.bar(x, y, color='teal')
plt.xlabel('Attack Type')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=90)
plt.title('Attack Types Used by Terrorists')
plt.show()


# Based on the above visuals, it appears that terrorist are most fond of using bombs and dynamites in their attacks. Obviously enough, the use of bombs as the weapon of choice seems to lead to more bombing attacks. **Do firearms also only directly correlate mainly with one attack type?** For example, is there a correlation between terrorists using firearms as their weapon of choice and armed assault attacks?

# In[ ]:


y = terror[terror['weapon_type'] == 'Firearms'].groupby('attack_type').count()['id']
x = y.index
explode=(0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
percent = 100.*y/y.sum()
patches, texts = plt.pie(y, explode=explode, startangle=90, radius=1.1)
labels = ['{0} - {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]

plt.legend(patches, labels, bbox_to_anchor=(-0.2, 1), fontsize=10)
plt.axis('equal')
plt.title('Distribution of Attack Types for Firearms')
plt.show()


# While a majority of the attacks caused by firearms are armed assaults, not all of them fall under this category. It seems terrorists also use firearms for assassinations attacks and kidnapping attacks.

# So who do terrorists try to target with all these attacks? Governments? Individuals?

# In[ ]:


y = terror['target'].value_counts()
x = y.keys()

plt.figure(figsize=(10,5))
plt.barh(x, y)
plt.ylabel('Target Type')
plt.xlabel('Number of Attacks')
plt.title('Terrorist Attacks Against Targets')
plt.show()


# More than anything, it appears that terrorists like to target private citizens and property. Previously, when we were looking at the visuals for Iraq, we noticed that a majority of the attacks in the Middle East occured in Iraq. **Is it possible that most attacks in countries on private citizens are overwhelmingly low, with only a few outliers skewing the data?**

# In[ ]:


y = terror[terror['target'] == 'Private Citizens & Property']['country'].value_counts()

plt.hist(y, bins=20, color='orange')
plt.xlabel('Number of Attacks on Private Citizens & Property')
plt.ylabel('Number of Countries')
plt.title('Distribution of Terrorist Attacks on Private Citizens & Property')
plt.show()
print('Most of the terrorist attacks on Private Citizens & Property occur in {}.'.format(y.index[0]))


# # Conclusion  
# It appears that most of the terrorist attacks occurs in very specific locations, specifically Iraq in the Middle East region and the South Asia region. 

# One of the things I've seen while researching for my Capstone Analytic Report is that there are a vast amount of visualizations available to use, both from matplotlib and other packages. I would like to learn more about these packages as well as go more indepth with the plots that were already covered in the prep course. While the examples provided in the prep course showed me the basics, some of the more advanced ways to use these graphs weren't covered. This made it challenging for me to create interesting graphs for this analytic report.
# 
# One thing that I wish I could have done more with this dataset is be more indepth in my analysis. There were instances where I wanted to go more indepth, but I felt at times that I was straying too far from the initial prose of my analysis.
# 
# Another thing I want to improve upon is to ask better questions. Initially, I felt confident in my questions, but after reviewing the questions with my mentor, I realized the questions I prosed were not solid questions at all! Unfortunately, I think the only way to improve this is by constantly asking questions and trying to see if I can ask different types of questions beyond just looking at the surface of the issues.
