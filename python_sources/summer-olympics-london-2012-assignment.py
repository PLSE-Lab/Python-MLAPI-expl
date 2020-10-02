#!/usr/bin/env python
# coding: utf-8

# # Assignment 02: Evaluate the Summer Olympics, London 2012 dataset
# 
# *The comments/sections provided are your cues to perform the assignment. You don't need to limit yourself to the number of rows/cells provided. You can add additional rows in each section to add more lines of code.*
# 
# *If at any point in time you need help on solving this assignment, view our demo video to understand the different steps of the code.*
# 
# **Happy coding!**
# 
# * * *

# #### 1: View and add the dataset

# In[ ]:


#Import the necessary library
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# In[ ]:


#Manually add the Summer Olympics, London 2012 dataset as arrays
df = pd.read_excel("../input/Olympic 2012 Medal Tally.xlsx",sheet_name='Sheet1',skiprows = [1,2])
arrCountries = np.array(df['Unnamed: 1'])
arrCountriesCode = np.array(df['Unnamed: 2'])
arrCountriesWonYear = np.array(df['Unnamed: 3'])
arrCountriesWonTotalGold = np.array(df['Unnamed: 4'])
arrCountriesWonTotalSilver = np.array(df['Unnamed: 5'])
arrCountriesWonTotalBronze = np.array(df['Unnamed: 6'])


# #### Find the country with maximum gold medals

# In[ ]:


#Use the argmax() method to find the highest number of gold medals
highestCountryGoldIndex = arrCountriesWonTotalGold.argmax()
arrCountries[highestCountryGoldIndex]


# In[ ]:


#Print the name of the country
for countryName in arrCountries:
    print(countryName)


# #### Find the countries with more than 20 gold medals

# In[ ]:


#Use Boolean indexing technique to find the required output
arrCountriesMoreThan20Gold = arrCountries[arrCountriesWonTotalGold > 20]
for countryName in arrCountriesMoreThan20Gold:
    print(countryName)


# #### Evaluate the dataset and print the name of each country with its gold medals and total number of medals

# In[ ]:


#Use a for loop to create the required output
print ("{:<20} {:<10} {:<10}".format('Country','Golds','Total Medals')) # formatting tables
for index,country in enumerate(arrCountries):
    totalMedals = (arrCountriesWonTotalGold[index]+arrCountriesWonTotalSilver[index]+arrCountriesWonTotalBronze[index]);
    print ("{:<20} {:<10} {:<10}".format(country,arrCountriesWonTotalGold[index],totalMedals))

