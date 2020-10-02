#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


script_lines = pd.read_csv("../input/simpsons_script_lines.csv", error_bad_lines=False)


# In[ ]:


script_lines.head()


# In[ ]:


#check that characters names are consistent within database
characters = pd.read_csv("../input/simpsons_characters.csv")
inconsistent = script_lines[script_lines["raw_character_text"].isin(characters["name"]) == False]
print(inconsistent["raw_character_text"].value_counts())


# It appears that certain characters have modified names in the script data set. These modifications identify different ages or thoughts. Will need to consider changing these to the original characters names for future analyses. This has been ignored for now.

# In[ ]:


#Top 5 characters with most lines
print(script_lines.groupby("raw_character_text").count()["id"].sort_values(ascending=False)[0:5])


# It's not surprising that the four characters with the most lines are also the four main characters who can speak. The next analysis will be to look at the locations where Homer speaks the most lines.

# In[ ]:


homer = script_lines[script_lines["raw_character_text"] == "Homer Simpson"]
homer_loc = homer.groupby("raw_location_text").count()["id"].sort_values(ascending=False)
print(homer_loc[0:10])


# Some of these locations are fairly similar and can be collapsed into the same category. For example, "Simpson Home" and "Simpson Living Room". About to collapse categories below.

# In[ ]:


#Set of functions for exploring location datasets

#Search locations for specific word and produce list, function defaults to homer location list
def list_search (word, search_list=homer_loc.index):
    return_list = []
    for item in search_list:
        if word.lower() in item.lower():
            return_list.append(item)
            
    return return_list

#print list with item number
def list_print (to_print):
    for x,item in enumerate(to_print):
        print(x,item)

#create new list from subsection of an old list
def list_select (original_list, keep_list):
    new_list = []
    for x in keep_list:
        new_list.append(original_list[x])
    return new_list

#take list of locations and print number of lines from each location
def lines_in_location(location_list, df=homer):
    for location in location_list:
        lines = df[df["raw_location_text"] == location].shape[0]
        print(location, lines)

#print lines of dialog from listed locations
def display_dialog(location_list, df=homer):
    for location in location_list:
        print(location)
        print(df[df["raw_location_text"] == location]["spoken_words"])
        print("\n")
    


# In[ ]:


h_simpson_list = list_search("simpson")
list_print(h_simpson_list) #trying to identify which locations are inside the house.


# Some of the location names above are vague and will need investigation to determine if they are within the Simpson's house. Below I have selected the locations that are definitely within the Simpson's house

# In[ ]:


simpsons_house_numbers = [0,2,3,5,6,7,11,20,21,29,30,33]
house_list = list_select(h_simpson_list, simpsons_house_numbers)
list_print(house_list)


# In[ ]:


possible_house_numbers = [10,44,47,48,52]
possible_house = list_select(h_simpson_list, possible_house_numbers)

display_dialog(possible_house)


# Comparing the dialog above with information from [Friniac][1] shows that lines from "Simpsons", "SIMPSON AND FLANDERS HOUSES", and "Simpson" took place within the Simpson's house. The line from "SIMPSON AND FLANDERS HOUSES" took place outside. The location of "SIMPSON'S" could not be determined
# 
# 
#   [1]: https://frinkiac.com

# In[ ]:


#Add locations from above to house_list
house_list.extend(["Simpsons","SIMPSON AND FLANDERS HOUSES","Simpson"])

#find number of lines for homer per location in house_list
lines_in_location(house_list)


# In[ ]:


#identifying list of locations within Moe's tavern. Include search terms pub and bar

h_tavern_list = list_search("tavern")
list_print(h_tavern_list)
print("\n")
pub_list = list_search("pub")
list_print(pub_list)
print("\n")
bar_list = list_search("bar")
list_print(bar_list)


# In[ ]:


possible_tavern = []
possible_tavern.extend(list_select(pub_list,[6]))
possible_tavern.extend(list_select(bar_list,[3]))

display_dialog(possible_tavern)


# It appears that only some of the dialog from "Bar" takes place in Moe's Tavern and none from "Pub". These locations will be excluded from the Moe's Tavern list

# In[ ]:


tavern_list = h_tavern_list


# In[ ]:


power_plant_list = list_search("power")
list_print(power_plant_list)


# In[ ]:


power_plant_list = list_select(power_plant_list,[0,2,5,6])


# In[ ]:


h_car_list = list_search("car")
list_print(h_car_list)


# In[ ]:


car_list = list_select(h_car_list,[0,1])


# In[ ]:


#display list of locations for cleaning homer df
list_print(house_list)
print("\n")
list_print(tavern_list)
print("\n")
list_print(power_plant_list)
print("\n")
list_print(car_list)


# In[ ]:


def simple_locations(row):
    location = "Other"
    if row["raw_location_text"] in (house_list):
        location = "Simpson's House"
    elif row["raw_location_text"] in (tavern_list):
        location = "Moe's Tavern"
    elif row["raw_location_text"] in (power_plant_list):
        location = "Nuclear Power Plant"
    elif row["raw_location_text"] in (car_list):
        location = "Simpson's Car"
    return location

homer["simple_locations"] = homer.apply(simple_locations, axis = 1)
no_other = homer[homer["simple_locations"] != "Other"]

Homers_lines_by_location = no_other.groupby("simple_locations").count()["id"].sort_values(ascending=False)
Homers_lines_by_location


# In[ ]:


Homers_lines_by_location.plot.bar()


# In[ ]:


#graph mean words per location for homer
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=homer,x = "simple_locations",y = "word_count")


# In[ ]:


sns.boxplot(data=homer,x = "simple_locations",y = "word_count")


# It appears that the amount of words that Homer speaks is similar in each location. Next set of analyses will focus on the other family members.

# In[ ]:


marge = script_lines[script_lines["raw_character_text"] == "Marge Simpson"]
marge_loc = marge.groupby("raw_location_text").count()["id"].sort_values(ascending=False)
print(marge_loc[0:10])


# With the exception of the church, Marge's top locations are similar to Homer's. Next will check to see if the simplified location lists for Homer work for Marge.

# In[ ]:


m_simpson_list = list_search("simpson", search_list=marge_loc.index)
m_car_list = list_search("car", search_list=marge_loc.index)
m_tavern_list = list_search("tavern", search_list=marge_loc.index)

def compare_lists(original, compare):
    for x, item in enumerate(compare):
        if item not in original:
            print(x,item)

print("House:")
compare_lists(h_simpson_list, m_simpson_list)
print("Car:")
compare_lists(h_car_list, m_car_list)
print("Tavern:")
compare_lists(h_tavern_list, m_tavern_list)


# Marge has a couple of new locations on her search list that are not present on Homer's list. They will be added to the general lists below.

# In[ ]:


house_list.extend(list_select(m_simpson_list,[9]))
car_list.extend(list_select(m_car_list,[1]))


# In[ ]:


m_church_list = list_search("church", search_list=marge_loc.index)
list_print(m_church_list)


# In[ ]:


church_list = list_select(m_church_list, [0,2,3,8,9,10])
list_print(church_list)


# In[ ]:


lisa = script_lines[script_lines["raw_character_text"] == "Lisa Simpson"]
lisa_loc = lisa.groupby("raw_location_text").count()["id"].sort_values(ascending=False)
print(lisa_loc[0:10])

