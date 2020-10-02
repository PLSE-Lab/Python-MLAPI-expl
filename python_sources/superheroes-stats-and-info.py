#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


charcters_stats = pd.read_csv("../input/charcters_stats.csv")
charcters_stats.head()


# In[ ]:


superheroes_power_matrix = pd.read_csv("../input/superheroes_power_matrix.csv")
superheroes_power_matrix.head(1)


# In[ ]:


marvel_dc_characters = pd.read_excel("../input/marvel_dc_characters.xlsx")
marvel_dc_characters.head(1)


# In[ ]:


marvel_characters_info = pd.read_csv("../input/marvel_characters_info.csv")
marvel_characters_info.head(1)


# **Questions**
# 1. Show 5 top comics with top participants on a plot bar.
# 2. Good versus Evil - Which group has more combined power?
# 3. Show pairs of characters that always appear together. rank them by number of appearances
# 4. Unmatched rivals - show for each super hero , all the vilans that stronger then him
# 5. Weak point - for each vilan, show his weakest characteristic.
# 6. Who can beat me? - for each vilan, show how many superheros can defeat them (compare by total score)
# 7. 
# 
# 

# **Questions by Omer**
# 
# **Easy**
# 
# Who is the most powerfull hero and villain (by total stats)?
# 
# Which alignment (good\bad) has higher avg speed?
# 
# How many superheros have more Intelligence then strength?
# 
# Show the distrebution of Total (all the powers combined) for the good and the evil.
# 
# How many comics contain only one good character and one evil one?
# 
# How many comics with 7 or more character published each year? (show on graph)
# 
# How has more characters DC or Marvel?
# 
# How has higher representation of female heros DC or Marvel?
# 
# How has higher representation of black skined heros DC or Marvel?
# 
# Show how common is each trait in 'superheroes_power_matrix.csv'.
# 
# Show the hight distrebution for the characters of 'Marvel Comics' (from 'marvel_characters_info.csv').
# 
# Show the distrebution of apperences (from 'marvel_dc_characters.csv').
# 
# Show the distrebution of eye colors (from 'marvel_dc_characters.csv').
# 
# How many characters apperred only once (from 'marvel_dc_characters.csv')?
# 
# How many characters died in thair first apperance (have one apperance and are deceased) (from 'marvel_dc_characters.csv')?

# In[ ]:


marvel_comics = pd.read_csv("../input/comics.csv")
# print (len(marvel_comics))
marvel_characters = pd.read_csv("../input/characters.csv")
# print (len(marvel_characters))
marvel_characters_in_comics = pd.read_csv("../input/charactersToComics.csv")
# print (len(marvel_characters_in_comics))
# marvel_comics.head
# marvel_characters.head
# marvel_characters_in_comics.head

comics_and_characters = marvel_comics.merge(marvel_characters_in_comics).merge(marvel_characters)
comics_and_characters = comics_and_characters.sort_values("comicID",ascending="True")
# print (len(comics_and_characters))
comics_and_characters = comics_and_characters.drop_duplicates()
# print (len(comics_and_characters))
comics_and_characters.head(1)


# In[ ]:


#Display a pie chart of the 10 most common hair styles
marvel_chars_df = marvel_characters_info.replace("No Hair", "Bald") #this transformation needs to be added to the data
marvel_chars_df = marvel_characters_info.replace("-", "Unknown") #this transformation needs to be added to the data
most_common_hair_styles = pd.value_counts(marvel_chars_df["HairColor"]).head(10)
most_common_hair_styles.plot.pie()


# In[ ]:


# Display the average height
marvel_chars_df
pd.DataFrame.mean(marvel_chars_df["Height"])
# plt.hist(heights)
# plt.xlabel('height (cm)')
# plt.ylabel('number');


# In[ ]:


#find the comic with the most participants
pd.value_counts(comics_and_characters["comicID"])
# after you found it display the characters the participate it in
comics_and_characters[comics_and_characters["comicID"]==4241]


# In[ ]:


# the oldest character of both universes
pd.DataFrame.min(marvel_dc_characters["Year"])

# marvel_dc_characters


# In[ ]:


#find the heroes that has the maximum amount of abilities(more True values than anyone)


# In[ ]:




