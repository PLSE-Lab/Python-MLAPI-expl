#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

names = pd.read_csv("../input/NationalNames.csv")


# In[ ]:


# Computing the average name lengths as a function 
# of the years - to check if names got shorter or 
# longer as time went on. 


def compute_average(year):
    # Computes the average length of the names for each year
    x = (names.Name[names.Year==year])
    y = x.str.len()
    return np.average(y)

# Use two lists to store simultaneously the average length 
# and the year; I could use maps - but well! 
averages = []
years = []

# Loop over the years 
for year in names.Year.unique():
    averages.append(compute_average(year))
    years.append(year)
    
plt.plot(years, averages)
plt.xlabel('Years')
plt.ylabel('Average length of names')


# The average lenght of baby names seems to be 
# between 5-6 letters through time. There is a 
# rise in the length of names until ~1990 after 
# which we see a sharp decline up to the present 
# times.

# In[ ]:


# To compute the number of NEW names as a function 
# of time. This helps interpret the times of 'high
# immigration' and their 'flux'.

# Simply, the names that are present in one time period 
# compared to its immediate previous time period; 
# the newer names are most likely to be attributed by 
# the new populus

def names_in_years(year, threshold=5):
    # Returns a series of names for a stipulated period
    x = names.Name[names.Year>=year][names.Year<year+threshold]
    return x

# We finally have a map this time! 
yeared_names = {}
# Set the time period over which to evaluate the new 
# names
n_years = 1

for year in names.Year.unique():
    if year%n_years == 0:
        listnames = names_in_years(year, threshold=n_years)
        yeared_names.update({year:listnames})
        
newnames_years = {}
for years in sorted(yeared_names.keys(), reverse=True):
    if years != 1880:
        no_newnames = len(yeared_names[years-n_years][yeared_names[years-n_years].isin(yeared_names[years])==False])
        newnames_years.update({years:no_newnames})


plt.plot([Year for Year in (newnames_years.keys())],[Count for Count in (newnames_years.values())])
plt.xlabel('Years')
plt.ylabel('Number of new names')


# From 1880 to ~1910, the number of new names 
# is fairly low and increases gradually - possibly 
# attributed to variations of existing names and 
# small immigrations.
# 
# ~1910 a sharp rise in new names is observed - 
# this is attritbuted to the European immigration 
# which was at its peak in 1907
# 
# An immigration act in 1924 restricted the influx
# of immigrants - and enforced a quota system for 
# immigrants for different countries. 
# 
# The global depression combined with WW2 saw a 
# decline in immigration and hence the number of 
# new names.
# 
# In 1965 however, Congress passed the Immigration 
# and Nationality Act and since then we see a rapid
# rise in the number of new names year-on-year! 
# Source: http://www.history.com/topics/u-s-immigration-before-1965
# 
# It will be interesting to see this graph in the next
# 10 years, depending on who the next US president is! :) 
