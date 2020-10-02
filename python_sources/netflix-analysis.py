#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS
# # DATA CLEANING
# # DATA MAPPING
# # DATA VISUALIZATION

# We import pandas library to do exploratory data analysis and visualization

# In[ ]:


import pandas as pd


# Since the file "movie_titles.csv" does not contain any column names i had to give it manually. and we should use encoding = "ISO-8859-1" to read the csv file or else we encounter error.

# In[ ]:


columnNames = ['YEAR','MOVIE']
MT = pd.read_csv("../input/netflix-prize-data/movie_titles.csv",names= columnNames,encoding = "ISO-8859-1")


# In order to observe the first 20 records of the file "movie_titles.csv" we use head() function

# In[ ]:


MT.head(20)


# As we can observe that YEAR column has values in FLOAT , lets try to convert it into integer.
# Now,find the number of null values present in each column.

# In[ ]:


MT.isnull().sum()


# Now, we found that column YEAR has 7 null values and now we have to find those movies which has null years and fill them

# In[ ]:


MT[pd.isnull(MT).any(axis=1)]


# 1.   Ancient Civilizations: Rome and Pompeii - 2001
# 2.   Ancient Civilizations: Land of the Pharaohs - 2001
# 3.   Ancient Civilizations: Athens and Greece - 2001
# 4.   Roti Kapada Aur Makaan - 1974
# 5.   Hote Hote Pyaar Ho Gaya - 1999
# 6.   Jimmy Hollywood -1994   
# 7.   Eros Dance Dhamaka -1999
# 
# Now, fill accordingly in the year column to their movies.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


MT.at[4388,'YEAR'] = 2001
MT.at[4794,'YEAR'] = 2001
MT.at[7241,'YEAR'] = 2001
MT.at[10782,'YEAR'] = 1974
MT.at[15918,'YEAR'] = 1999
MT.at[16678,'YEAR'] = 1994
MT.at[17667,'YEAR'] = 1999


# Now, lets find if there are any null values left in the data frame.

# In[ ]:


MT.isnull().sum()


# Yay!! No null values were found . 
# 
# 1.   lets convert the YEAR column to Integer from FLOAT.
# 
# 
# 

# In[ ]:


MT['YEAR'] = MT['YEAR'].astype(int)


# YAY!! we successfully converted the YEAR column to integers.
# 
# 1.   Lets verify by displaying the dataframe .
# 
# 
# 

# In[ ]:


MT.head(5)


# 1. Read the combined_data_1.txt , combined_data_2.txt, combined_data_3.txt, combined_data_4.txt . As it is in raw form we    might encounter with errors . so better we use seperator and also give column names as given in the ReadMe File.
# 
# 2. output the first 5 rows in the combined_data_1.txt file.
#    we see that it has null first row 
#    
# 3. Remove all the rows having the null values .
# 
# 4. find out how many null values are present in the data set and particularly in each column.
# 
# 5. Now verify if the dataset has null values are not.
# 
# 6. Find the shape of the data set and then remove all the null valued rows present in the data set.
#    Here we see that it has 3 columns and 19 lakh rows.
#    

# In[ ]:


Cd1 = pd.read_csv("../input/netflix-prize-data/combined_data_1.txt",sep = ',',names= ['CustomerID','Rating','Date'] )


# In[ ]:


Cd2 = pd.read_csv("../input/netflix-prize-data/combined_data_2.txt",sep = ',',names= ['CustomerID','Rating','Date'] )


# In[ ]:


Cd3 = pd.read_csv("../input/netflix-prize-data/combined_data_3.txt",sep = ',',names= ['CustomerID','Rating','Date'] )


# In[ ]:


Cd4 = pd.read_csv("../input/netflix-prize-data/combined_data_4.txt",sep = ',',names= ['CustomerID','Rating','Date'] )


# In[ ]:


Cd1.head(5)


# In[ ]:


Cd2.head(5)


# In[ ]:


Cd3.head(5)


# In[ ]:


Cd4.head(5)


# In[ ]:


Cd1 = Cd1.dropna(how='any')


# In[ ]:


Cd2 = Cd2.dropna(how='any')


# In[ ]:


Cd3 = Cd3.dropna(how='any')


# In[ ]:


Cd4 = Cd4.dropna(how='any')


# In[ ]:


Cd1.isnull().sum()


# In[ ]:


Cd2.isnull().sum()


# In[ ]:


Cd3.isnull().sum()


# In[ ]:


Cd4.isnull().sum()


# In[ ]:


Cd1.shape


# In[ ]:


Cd2.shape


# In[ ]:


Cd3.shape


# In[ ]:


Cd4.shape 


# Data Visualization

# In[ ]:


MT['YEAR'].max()


# In[ ]:


MT['YEAR'].min()


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt


# In order to find the number of movies released on a particular year in the given movie_titles.csv dataset. i did the below code and gave the figure size or other wise the file size becomes too small to analyse.

# In[ ]:


MT['YEAR'].value_counts().plot(kind='bar', figsize=(20,5) )


# In[ ]:




