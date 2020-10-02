#!/usr/bin/env python
# coding: utf-8

# **Spoiler alert:**  This won't be like a  data visualization kernel.This is small dataset so i decided to  find the insights using python basic syntax.( no visualiztion).I am just making this for change.Lets see how much diffcult to understand the data without visualization.

# SpaceX
# 
# SpaceX designs, manufactures and launches advanced rockets and spacecraft. The company was founded in 2002 to revolutionize space technology, with the ultimate goal of enabling people to live on other planets.It was founded in 2002 by entrepreneur Elon Musk with the goal of reducing space transportation costs and enabling the colonization of Mars. SpaceX has since developed the Falcon launch vehicle family and the Dragon spacecraft family, which both currently deliver payloads into Earth orbit.
# 

# In[ ]:



import numpy as np 
import pandas as pd
import os


# In[ ]:


#Import csv file
df=pd.read_csv("../input/spacex_launch_data.csv")
df.head()


# In[ ]:


#Lets find the dimension of the dataframe
df.shape


# In[ ]:


# There is 57 rows and 11 cloumns
#Lets print the column name
df.columns


# In[ ]:


# Lets find the no of unique element in each column

#First lets store the column name into a list variable
column=list(df.columns)

#Lets print unqiue elemnt in each column
for i in column:
    print("Column name: "+str(i))
    print("Total unique Elements: "+str(len(df[i].unique())))
    print("Unique List are: ")
    print(df[i].unique())
    print("*************************************************************************************************")

# You can feel why I am doing like this.Insted of this,
# we can put in  beautiful data visualization.For a chance I want to see like this.


# Lets compare customer over the time period
# 
# There are 30 unique customer and 27 customer are repeated.

# In[ ]:


# Lets comapre with year.Now lets extract year from the date.

year=[]
for i in df["Date"]:
    year.append(i.split("-")[0])
print("year")
print(year)
print("Unique Year")
print(set(year))


# **Did you notice it.... The difference between year and unique year.**
# .
# .
# .
# .
# .
# .
# WAIT FOR IT
# .
# .
# .
# .
# .
# .
# 2011 !!!!!
# There is a break for one year in spacex.
# 

# In[ ]:


#Customer diversity

#Top repeated customer for spaceX and these are the only customer who came more than one time 
df["Customer"].value_counts()[:9]


#  **Did you notice it........ In customer table above**
#  .
#  .
#  .
#  .
#  .
#  WAIT FOR IT
#  .
#  .
#  .
#  .
#  .
#  .
#  SpaceX is been customer for Spacex for two times 
#  

# Thank you for reading.This is version1 of my work.I will keep on updating it. Support me by upvote and comment
