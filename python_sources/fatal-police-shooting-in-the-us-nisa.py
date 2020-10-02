#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Our dataset gives some information about the people who were killed in the US.

# ![](https://www.tampabay.com/resizer//TlVjhZYHH67eiYz-zwqr0bHDmBE=/800x450/smart/cloudfront-us-east-1.images.arcpublishing.com/tbt/XACMIVYVFZBFDOALGH2GHFJBYM.jpg)

# I am going to visualize data with using:
# * Bar Plot
# * Point Plot
# * Joint Plot
# * Count Plot
# * Pis Chart
# * Lm Plot
# * Kde Plot
# * Box Plot
# * Swarm Plot
# * Pair Plot

# I am going explain each following parts and also visualize them:

# 1. [Poverty Level of Each State](#1) 
# 1. [Most Common 15 Names that Tere Killed](#2)
# 1. [High School Graduation Rate Of The Population That Is Older Than 25 In The State](#3)
# 1. [Percentage Of State's Population According To Races That Are Black,White,Native American,Asian And Hispanic](#4)
# 1. [High School Graduation Rate vs Poverty Rate of Each State](#5)
# 1. [Race Rate According to Kill Data](#6)
# 1. [High School Graduation Rate vs Poverty Rate of Each State With Different Style of Seaborn Code](#7)
# 1. [Classifiying People According to Their Manner of Death, Their Genders and Ages](#8)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# data read:

# In[ ]:


below_poverty = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding = "windows-1252")
police_killings = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "windows-1252")
share_race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding = "windows-1252")
completed_high_school_over25 = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding = "windows-1252")
house_income_median = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding = "windows-1252")


# **encoding = "windows-1252"** was used for handling utf-8 error.

#  <a id = "1"></a>
# # Poverty level of each state:

# In[ ]:


below_poverty.head(10)


# To get general idea about our poverty data we use "**info()**"method.

# In[ ]:


below_poverty.info()


# In[ ]:


below_poverty.isnull().any()


# In[ ]:


below_poverty.poverty_rate.value_counts()


# We don't know the meaning of some rates because they are represented "-" in the data. So, we can ignore these 201 data or replace with 0.

# In[ ]:


below_poverty.poverty_rate.replace("-",0, inplace = True)


# After replating "-" values, we can check using "**value_counts()**" method.

# In[ ]:


below_poverty.poverty_rate.value_counts()


# Then, as you can see, type of poverty rate is object. We need to convert it to numerical value.

# In[ ]:


below_poverty.poverty_rate = below_poverty.poverty_rate.astype(float)


# In[ ]:


below_poverty.poverty_rate.value_counts()


# I want to find powerty rate of each state. So, we need to find the each unique state.

# In[ ]:


below_poverty["Geographic Area"].value_counts().index


# In[ ]:


below_poverty["Geographic Area"].value_counts().values


# or we can use "unique()" method to show each unique state.

# In[ ]:


below_poverty["Geographic Area"].unique()


# Converting to a list:

# In[ ]:


unique_area_list = list(below_poverty["Geographic Area"].unique())


# In[ ]:


unique_area_list


# ## Using matplotlib library, we can visualize poverty level of each state like this:

# In[ ]:


plt.figure(figsize = (15,10))
plt.title("Poverty Level of Each State")
plt.bar(unique_area_list, below_poverty["Geographic Area"].value_counts().values) 
plt.xlabel("States")
plt.ylabel("Poverty Level")
plt.show()


# ## Using seaborn library, we can visualize poverty level of each state like this:

# In[ ]:


plt.figure(figsize = (15,10))
plt.title("Poverty Level of Each State")
sns.barplot(unique_area_list, below_poverty["Geographic Area"].value_counts().values) 
plt.xticks(rotation = 45)   # xticks method is used to determinethe location of variable names on the x axis.
plt.xlabel("States")
plt.ylabel("Poverty Level")
plt.show()


# <a id = "2"></a>
# # most common 15 names that were killed:

# In[ ]:


police_killings.head(10)


# In[ ]:


police_killings.name.value_counts()


# zipping study:

# In[ ]:


a = "nisa soylu"
a.split()


# In[ ]:


b = ["nisa soylu", "mine soylu","zeynep bumin soylu"]
c = []
for i in b:
    c.append(i.split())
    unzipped_version = zip(*c)
    print(list(unzipped_version))


# In[ ]:


first_part, second_part = zip(*(police_killings.name[police_killings.name != "TK TK"].str.split()))


# In[ ]:


print(first_part)


# In[ ]:


len(first_part)


# In[ ]:


print(second_part)


# In[ ]:


len(second_part)


# In[ ]:


name_list = first_part + second_part


# In[ ]:


print(name_list)


# In[ ]:


len(name_list)


# As you understand, firstly I unzipped names and surnames. (I made two seperated list). Then I concatenated these two list.

# In order to calculate the repetition of names, I used "Counter" method.

# In[ ]:


name_count = Counter(name_list)


# In[ ]:


print(name_count)


# In[ ]:


most_used_15_names = name_count.most_common(15)


# In[ ]:


most_used_15_names


# In[ ]:


x,y = zip(*most_used_15_names)


# In[ ]:


x,y = list(x), list(y)


# In[ ]:


print(x)


# In[ ]:


print(y)


# ## Using matplotlib library, we can visualize poverty level of each state like this:

# In[ ]:


plt.figure(figsize = (15,10))
plt.bar(x,y)
plt.title("Most Common 15 Names or Surnames of Killed People")
plt.xlabel("Name or surname of killed people")
plt.ylabel("Frequency")
plt.show()


# ## Using seaborn library, we can visualize poverty level of each state like this:

# In[ ]:


plt.figure(figsize = (15,10))
sns.barplot(x, y)
plt.xticks(rotation = 45) 
plt.xlabel("Name or surname of killed people")
plt.ylabel("Frequency")
plt.title("Most Common 15 Names or Surnames of Killed People")
plt.show()


#  <a id = "3"></a>
# # Let's find the high school graduation rate of the population that is older than 25 in the state.

# These are our data, so lets find which data has graduation information and which one has age information.

# * below_poverty 
# * police_killings
# * city_data
# * completed_high_school_over25
# * house_income_median

# In[ ]:


completed_high_school_over25.head(10)


# We can use completed high school data. But firstly, we should check the variables if there is a missing data or not.

# In[ ]:


completed_high_school_over25.percent_completed_hs.value_counts()


# As you can see, 197 of data has missing values. We should get rid of them in order to do our visualization correctly.

# I am going to change "-" values into 0.

# In[ ]:


completed_high_school_over25.percent_completed_hs.replace("-",0, inplace = True)


# Let's check any missing variable occurs or not:

# In[ ]:


completed_high_school_over25.percent_completed_hs.value_counts()


# We replaced correctly.

# Now we should check the data type of the percentage, who completed the highschool. They should be numerical.

# In[ ]:


completed_high_school_over25.info()


# But, as you can see, our data type is object. We should change it to float with using "astype()" method.

# In[ ]:


completed_high_school_over25.percent_completed_hs = completed_high_school_over25.percent_completed_hs.astype(float)


# In[ ]:


completed_high_school_over25.percent_completed_hs.value_counts()


# In[ ]:


list_of_states = completed_high_school_over25["Geographic Area"].unique()


# In[ ]:


type(list_of_states)


# In[ ]:


state_list = list(list_of_states)
state_list


# In[ ]:


#completed_high_school_over25["Geographic Area"].value_counts().index.sort_values()
completed_high_school_over25["Geographic Area"].sort_values().value_counts()
    


# In[ ]:


area_highschool = []
for i in state_list:
    x = completed_high_school_over25[completed_high_school_over25['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)


# Sorting data:

# In[ ]:


data = pd.DataFrame({'area_list': state_list,'area_highschool_ratio':area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)


# Visualization using seaborn:

# In[ ]:



# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
plt.show()


#  <a id = "4"></a>
# # Percentage Of State's Population According To Races That Are Black,White,Native American,Asian And Hispanic

# In[ ]:


share_race_city.head()


# In[ ]:


share_race_city.info()


# Firstly, we need to check if there is a missing value or not(for each column). We have 7 different columns, which are:
# * Geographic area        
# * City                   
# * share_white            
# * share_black            
# * share_native_american  
# * share_asian            
# * share_hispanic

# In[ ]:


share_race_city["Geographic area"].value_counts()


# Values are good for Geographic Area column

# In[ ]:


share_race_city.City.value_counts()


# In[ ]:


share_race_city.share_white.value_counts()


# In[ ]:


share_race_city.share_black.value_counts()


# In[ ]:


share_race_city.share_native_american.value_counts()


# In[ ]:


share_race_city.share_asian.value_counts()


# In[ ]:


share_race_city.share_hispanic.value_counts()


# * We have some corruptions in our data("(X)" and "0"). I replaced them to 0.
# * Then, as you can see, type of share_white, share_black, share_native_american, share_asian and share_hispanic data are object. We need to change them to float64. (using "astype()" method)

# In[ ]:


share_race_city.replace("-",0, inplace = True)
share_race_city.replace("(X)",0, inplace = True)

share_race_city.share_white = share_race_city.share_white.astype(float)
share_race_city.share_black = share_race_city.share_black.astype(float)
share_race_city.share_native_american = share_race_city.share_native_american.astype(float)
share_race_city.share_asian = share_race_city.share_asian.astype(float)
share_race_city.share_hispanic = share_race_city.share_hispanic.astype(float)


# In[ ]:


share_race_city.info()


# In[ ]:


area_list = list(share_race_city["Geographic area"].unique())

share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for i in area_list:
    x = share_race_city[share_race_city['Geographic area']==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))


# Visualization part using Seaborn Library:

# In[ ]:


plt.figure(figsize = (9,15))
plt.yticks(rotation = 0)
sns.barplot(share_white, area_list, color = "orange", label = "White")
sns.barplot(share_black, area_list, color = "brown", label = "African American")
sns.barplot(share_native_american, area_list, color = "red", label ="Native American")
sns.barplot(share_asian, area_list, color = "yellow", label ="Asian")
sns.barplot(share_hispanic, area_list, color = "purple", label ="Hispanic")
plt.xlabel("Percentage of Races")
plt.ylabel("States")
plt.title("Percentage Of State's Population According To Races")
plt.legend()
plt.show()


#  <a id = "5"></a>
# # High School Graduation Rate vs Poverty Rate of Each State

# ## Let's calculate poverty rate of each state:

# In[ ]:


below_poverty.head()


# In[ ]:


below_poverty["Geographic Area"].value_counts()


# In[ ]:


below_poverty["poverty_rate"].value_counts()


# The following code, I prepared visualization graph's x and y axis.

# In[ ]:


city_names = list(below_poverty["Geographic Area"].unique())

poverty_rates_each_city = []

for i in city_names:
    x = below_poverty[below_poverty["Geographic Area"] == i]
    poverty_rates_each_city.append((sum(x.poverty_rate))/len(x))


# In[ ]:


poverty_rates_each_city


# ## Let's calculate high school graduation rate of each state:

# In[ ]:


completed_high_school_over25.head()


# In[ ]:


graduation_rates_each_city = []

for i in city_names:
    x = completed_high_school_over25[completed_high_school_over25["Geographic Area"] == i]
    graduation_rates_each_city.append(sum(x.percent_completed_hs)/len(x))


# In[ ]:


graduation_rates_each_city


# ## Point Plot

# In[ ]:


plt.figure(figsize = (20,10))
sns.pointplot(city_names, graduation_rates_each_city, color ="lime", label = "graduation rate")
sns.pointplot(city_names, poverty_rates_each_city,color = "brown", label = "poverty rate")

plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Percentage of High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.text(40,30.60,'high school graduate ratio',color='lime',fontsize = 17,style = 'italic')
plt.text(40,25.55,'poverty ratio',color='brown',fontsize = 18,style = 'italic')
plt.legend()
plt.grid()
plt.show()


# As you can see, when we try to find relation between high school graduate and poverty rate using point plot didn't help us so much. We can maybe say looking at this graph, when the graduation increases, poverty rate decreases.

# In the following part, I am going to try visualize this relation, using "Joint Plot".

# ## Joint plot

# * pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# * If it is zero, there is no correlation between variables

# In[ ]:


sns.jointplot(poverty_rates_each_city, graduation_rates_each_city,ratio=3, color="purple", kind="kde", height=7)
plt.xlabel("area poverty percentage")
plt.ylabel("area highschool ratio")
plt.savefig('graph.png')
plt.show()


# As you can understand by looking at the graph, there is a negative correlation between high school percentage and poverty percentage.

# In[ ]:


g = sns.jointplot(poverty_rates_each_city, graduation_rates_each_city, height=5, ratio=3, color="lime")
plt.xlabel("area poverty percentage")
plt.ylabel("area highschool ratio")
plt.show()


#  <a id = "6"></a>
#  # Race Rate According to Kill Data 

# Following is the kill data:

# In[ ]:


police_killings.head()


# In[ ]:


police_killings.race


# As you can see, we have some missing values. We should get rid of them.

# In[ ]:


police_killings.race.dropna(inplace = True)


# Let's check if we could delete the missing variables from our data.

# In[ ]:


police_killings.race


# Data is good now.

# In[ ]:


police_killings.race.value_counts()


# In[ ]:


list(police_killings.race.value_counts().index)


# ## Pie Plot

# When we give the values, pie plot automatically calculates the rates and we can easily visualize it.

# In[ ]:


list(police_killings.race.value_counts().values)


# In[ ]:


plt.figure(figsize = (10,10))
colors = ["orange","purple","yellow","brown","lime","pink"]
explode = [0,0.1,0,0,0,0]  # only "explode" the 2nd slice (i.e. "B") 
plt.pie(list(police_killings.race.value_counts().values),explode,list(police_killings.race.value_counts().index),colors, autopct ="%1.1f%%")
plt.title("Killed People According to Races")
plt.show()


#  <a id = "7"></a>
#  # High School Graduation Rate vs Poverty Rate of Each State With Different Style of Seaborn Code

# ## KDE(Kernel Density Estimation) Plot

# In[ ]:


sns.kdeplot(poverty_rates_each_city, graduation_rates_each_city, color ="orange", shade = True) # shade => empty or filled shape
plt.xlabel("area poverty percentage")
plt.ylabel("area highschool ratio")
plt.show()


# ## Violin Plot

# Violin plot, rather than looking at the relation between the two data, looks at the distribution of different properties.

# In[ ]:


import numpy
n_poverty_rates_each_city = numpy.array(poverty_rates_each_city)
n_graduation_rates_each_city = numpy.array(graduation_rates_each_city)

plt.title("Poverty Percentage                 Graduation Percentage", color = "blue")

sns.violinplot(n_poverty_rates_each_city, color = "orange", inner = "points", label ="poverty percentage")
sns.violinplot(n_graduation_rates_each_city, color = "lime", inner = "points", label = "graduation percentage")
plt.legend()

plt.show()


# In[ ]:


n_poverty_rates_each_city/100


# In[ ]:


n_graduation_rates_each_city/100


#  <a id = "8"></a>
# # Classifiying People According to Their Manner of Death, Their Genders and Ages

# ## Box Plot

# Let's classify people according to their manner of death, their genders and ages. Using box plot will be beneficial for us for the visualization part.

# In[ ]:


sns.boxplot(police_killings.gender, police_killings. age, hue = police_killings.manner_of_death,data = police_killings)
plt.show()


# * The blue and orange rectangles in the left part represent men who was dead because of shoting(blue rectangle) or both shooting and terasing(orange rectangle). 
# * The blue and orange rectangles in the right part represent women who was dead because of shoting(blue rectangle) or both shooting and terasing(orange rectangle). 

# * Boxplot has a feature: Even if we don't write the code that explains what the x and y axis means, barplot automatically determines the names of the axes the chart by taking the qunique values in the selected columns.
# * I mean :

# In[ ]:


police_killings.gender.unique()


# barplot function automatically uses this "unique()" function to give variables for x axis.

# In[ ]:


police_killings.age.unique()


# barplot function automatically uses this "unique()" function to give variables for y axis.

# ## Swarm Plot

# In[ ]:


sns.swarmplot(police_killings.gender, police_killings.age, hue = police_killings.manner_of_death, data = police_killings)
plt.show()


# ## Count Plot

# In[ ]:


police_killings.manner_of_death.value_counts()


# Countplot is used for actually visualizing for "value_counts()" method.

# In[ ]:


sns.countplot(police_killings.gender)
plt.title("Gender", color = "orange")
plt.show()


# In[ ]:


police_killings.armed.value_counts().index


# In[ ]:


police_killings.armed.value_counts().values


# In[ ]:


plt.figure(figsize = (25,10))
sns.barplot(police_killings.armed.value_counts().index[:7], police_killings.armed.value_counts().values[:7])
plt.title("Gender", color = "lime")
plt.show()


# In[ ]:


sns.countplot(police_killings.manner_of_death)
plt.title("Gender", color = "red")
plt.show()


# # Conclusion

# As a result, I tried to explain and visualize Fatal Police Shooting in the US. If you have any questions or comments, I will be happy to answer them. If you think that this notebook is beneficial for you please up vote. Thanks.
