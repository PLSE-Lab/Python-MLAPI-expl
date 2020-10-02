#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from matplotlib import pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


import os
print(os.listdir("../input"))
print(os.listdir("../input/superheroes-characters-and-comics/"))
print(os.listdir("../input/superheroes-info-and-stats/"))


# # Read data from files

# In[ ]:


characters_stats = pd.read_csv("../input/superheroes-info-and-stats/superheroes_stats.csv")
characters_stats.sample(10)


# In[ ]:


superheroes_power_matrix = pd.read_csv("../input/superheroes-info-and-stats/superheroes_power_matrix.csv",index_col="Name")
superheroes_power_matrix.sample(10)


# In[ ]:


characters_info = pd.read_csv("../input/superheroes-info-and-stats/superheroes_info.csv",index_col=0,parse_dates=["FirstAppearance","Year"])
characters_info.sample(10)


# In[ ]:


marvel_comics = pd.read_csv("../input/superheroes-characters-and-comics/comics.csv")
marvel_characters = pd.read_csv("../input/superheroes-characters-and-comics/characters.csv")
marvel_characters_to_comics = pd.read_csv("../input/superheroes-characters-and-comics/charactersToComics.csv")


# ## Easy

# In[ ]:


# e-01
# question: Good versus Evil - Which group has more combined power?


# In[ ]:


# e-02
# question:   Which alignment (good\bad) has higher avg speed?


# In[ ]:


# e-03
# question: How many superheros have more Intelligence then strength?


# In[ ]:


# e-04
# question: Show the distribution of Total (all the powers combined) for the good and the evil.


# In[ ]:


# e-05
# question: How has more characters DC or Marvel?


# In[ ]:


# e-06
# question: Who has higher representation of female heros DC or Marvel?


# In[ ]:


# e-07
# question: Show the hight distribution for the characters of 'Marvel Comics'.


# In[ ]:


# e-08
# question: How many characters apperred only once?


# In[ ]:


# e-09
# question: How many characters died in their first apperance (have one apperance and are deceased)?


# In[ ]:


# e-10
# question:   Display a pie chart of the 10 most common hair styles


# In[ ]:


# e-11
# we want to build the master group to fight evil, kind of an avengers 2.0, but only better,
# lets select the captain, the one with the most total stats  (obviously his Alignment must be good to fight evil)


# In[ ]:


# e-12
# On the other hand, long lasting series are great as well :), retrive the comic book with the biggest issue number


# In[ ]:


# e-13
# What's the mean intelligence of the superheroes who have a 'True' value in the power matrix and the same for the superheroes who have a 'False' value?


# ## Medium

# In[ ]:


# m-01
# question: Show 5 top comics with the largest amount of participants on a plot bar.


# In[ ]:


# m-02
# question: Unmatched rivals - show for each super hero the number of vilans that stronger then him/her


# In[ ]:


# m-03
# question: Weak point - for each vilan, show his weakest characteristic.


# In[ ]:


# m-04
# find the comics with most participants and display all of the participants


# In[ ]:


# m-05
# A great team needs great diversity, and to be great at everything, get the best hero at each statistical category


# In[ ]:


# m-06
# Is your strength and intelligence related?.
# Show a scatter chart where the x axis is stength, and the y axis is intelligence, scatter heros and villans as two different color dots


# In[ ]:


# m-07
# To truly be a great superhero, you need to possess multipule abilities. Create a series of every superhero and how many different abilities they possess, in descending order


# In[ ]:


# m-08
# Create a serires that counts the number of comic book appeerences for each hero
# Bonus: show the top 10 heros in a pie chart


# In[ ]:


# m-09
# Pick any hero from the previous question and list all the comic book titles that he appeared in


# In[ ]:


# m-10
# It's the holiday season!
# retrive all heros who have participated in a holiday comic book (With "Holiday" in their title)


# In[ ]:


# m-11
#We saw that the characters with the 'False' intelligence do have a lower intelligence score than the 'True' ones. That means that the 2 different datasets we based our analysis on have a similar evalutaion of characters intelligence in general.

#* Can you find characters that one dataset classifies them as intelligent and the other one classifies them as not intelligent?

#* How many characters like that are there? (since 1 dataset is boolean and the other is numeric, assume that 100 score equals 'True' and 0 score equals 'False')

#* What that might say on those 2 datasets?


# 

# In[ ]:


# m-12
# show the distribution of BMI for all characters with height and weight data. show the distribution according to BMI categories


# ## Hard

# In[ ]:


# h-01
# Two of the most iconic marvel superheros, Iron Man and Captain America, appeer together quite offten. 
# see if you can get the ammount of comic books they both appear in together in, order by that ammount in a descending order


# In[ ]:


# h-02
# Now that we know how many comic books both of those guys have appeared together at, are they the best power duo in the marvel universe?.
# create a series with a multi index of 2 superheros(name1,name2) and count for each of them the ammount of comic books they have been in 


# In[ ]:


# h-03
# show 5 top rare and common abilites


# In[ ]:


# h-04
# question: Unmatched rivals - show for each super hero , all the names of the  vilans that stronger then him/her
# BONUS
# question: Unmatched rivals - find an informative way to visualize the results you got


# In[ ]:


# h-05
# question: Show pairs of characters that always appear together. rank them by number of appearances


# # The End
