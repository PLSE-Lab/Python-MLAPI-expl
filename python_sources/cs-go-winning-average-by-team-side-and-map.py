#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hi fellow player or ML dev!
# Since i'm a complete novice on machine learning, but a genuine lover of the game Counter Strike, as soon as I found this dataset with over 1400 CS:GO matches I felt the need to use it to get some game info. This is not a machine learning algorithm tho, its what me, as a ML begginer, am still getting used to, the data management and preparation, which is a very important step on machine learning algorithms.
# 
# # Goal
# 
# In this simple script, my goal is to calculate the average winning rate of certain side (such as CT or TR), on certain map (of all the maps available on the dataset).

# # Managing the csv
# 
# The first step, as shown below, was to read the csv file with *"pandas"* and then defining the columns which will be needed for the test (the *"map"* and the *"winner_side"*)

# In[ ]:


#Getting the dataset
cs_file_path = '../input/mm_master_demos.csv'
cs_data = pd.read_csv(cs_file_path)
#Defining the columns wich will be used
cols = ['map','round','winner_side']


# # Filtering by map
# 
# Now that we have our dataset with only the columns we need, we must filter the rows by the map we want to test. In order to do this the .loc can be used. *".loc"* is a purely label-location based indexer for selection by label.

# In[ ]:


#Selecting only the rows that contains the "map_to_test"
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]


# # Defining the test function
# 
# Now that we only have the data we want, comes the fun part. It can be done sequentially, but I find it more useful to create a function that can be called several times whenever I want to do a test. The function will be called *"get_winners_percentage"* in this case and will be responsible to sum all *terror* and *ct* side winnings over all the dataset on the selected map and then calculate what was the average.
# 
# To iterate through each row, the *.iterrows* can be used on the loop, then its just a matter of string comparison.
# 
# Since the dataset has data from each player in each round and we only want 1 row of each round, a variable *prevRound* is defined to test if we already checked that round's winner.

# In[ ]:


#Test function
def get_winners_percentage(map_to_test, cs_data):
    ct = 0.0
    terror = 0.0
    prevRound = 0
    
    #Using .iterrows to access each row
    for index, row in cs_data.iterrows():
        if (row['round'] != prevRound):
            prevRound = row['round']
            if (row['winner_side'] == 'Terrorist'):
               terror+=1
            else:
                ct+=1

    total = int(terror + ct)

    terror = float((terror/total) * 100)
    ct = float((ct/total) * 100)


# # Final result
# 
# Feel free to leave any comments or corrections.

# In[1]:


import pandas as pd

#Test function
def get_winners_percentage(map_to_test, cs_data):
    ct = 0.0
    terror = 0.0
    prevRound = 0
    
    #Using .iterrows to access each row
    for index, row in cs_data.iterrows():
        if (row['round'] != prevRound):
            prevRound = row['round']
            if (row['winner_side'] == 'Terrorist'):
               terror+=1
            else:
                ct+=1

    total = int(terror + ct)

    terror = float((terror/total) * 100)
    ct = float((ct/total) * 100)

    print ("from "+ str(total) +" rounds on map '"+ map_to_test +"', the winning percentage was:")
    print ("Terrorist: %.2f" %terror,"%")
    print ("Counter Terrorist: %.2f" %ct,"%")
    print ("")


#Getting the dataset
cs_file_path = '../input/mm_master_demos.csv'
cs_data = pd.read_csv(cs_file_path)
#Defining the columns wich will be used
cols = ['map','round','winner_side']


#Tests
#Defining the map to analyze
map_to_test = 'de_dust2'
#Selecting only the rows that contains the "map_to_test"
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
#Calling the test function
get_winners_percentage(map_to_test, cs_data1)

#Test 2 with a different map
map_to_test = 'de_cache'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 3 with a different map
map_to_test = 'de_inferno'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 4 with a different map
map_to_test = 'de_mirage'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 5 with a different map
map_to_test = 'de_overpass'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 6 with a different map
map_to_test = 'de_train'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 7 with a different map
map_to_test = 'de_dust'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

#Test 8 with a different map
map_to_test = 'de_cbble'
cs_data1 = cs_data[cols].loc[cs_data['map'] == map_to_test]
get_winners_percentage(map_to_test, cs_data1)

