#!/usr/bin/env python
# coding: utf-8

# # Find Best Rated Book For Every Year
# 
# ## Rating System
# 
# A Book is rated highly if more number of people give it a higher rating.
# i.e. If Book A is rated 5* by 100 people and Book B is rated '5' by 90 people
#     then Book A is rated higher, 
#     If Both A and B was rated by 100 people as '5' then the rating '4' is taken into account 
# 
# ### Importing Data

# In[ ]:


import pandas as pd
import numpy as np
import glob


# In[ ]:


all_files = glob.glob(r'../input/goodreads-book-datasets-10m/CSV datasets/CSV datasets' + "/*.csv")
l = []
for filename in all_files:
    df = pd.read_csv(filename, sep=',', usecols=['Name','RatingDist5','RatingDist4','RatingDist3','RatingDist2','RatingDist1','PublishYear'])
    l.append(df)

df = pd.concat(l, axis=0)
dataset = df.values


# ### Cleaning Data

# In[ ]:


# Changing ratings form '5:7' -> '7'
for column in range(2,7):
    new = []
    for row in range(dataset.shape[0]):
        new.append(int(dataset[row,column].split(':')[1]))
    dataset[:,column] = new 


# ## Finding Top 5 Books from 1995 to 2015

# In[ ]:


for year in range(1995,2016):
    this_years_books = dataset[dataset[:,1]==year,:] # Books published in given year
    # Sorting in order 5*, 4*, 3*, 2*, 1*
    for i in [6,5,4,3,2]:
        this_years_books = this_years_books[np.argsort(this_years_books[:, i])] 
        
    print("\nYear: ",year)
    for book in this_years_books[-5:,0]:
        print("\t",book)

