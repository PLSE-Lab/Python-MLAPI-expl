#!/usr/bin/env python
# coding: utf-8

# second data science

# Notas:
# <ul>
# <li>puedes mejorar buscando en el box plot los datos que no son outliers</li>
# </ul>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mpl
import seaborn as sb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print ("key questiom:")
print("1.-Which are cities in each states?")
print("2.-How many people on average are literate in each state by gender?")
print("3.-How stable are the states with respect to their populations of cities?")
print("4.-How many literates are graduate within each states?")
print("5.-Which is the most/least literates state?")
print("6.-Which is the most/least graduates state?")
print("7.-Which is the most child rate state on average ? and the least child rate state on average?")
print("8.-Which do city have most child rate? most graduates?  most literates?")
print("9.-Which is the most population state?")
print("10.-is there men are graduater than women?")
print ("try it!!!!" )
print ("Post_data : sorry fo my english i'm a new with it! , and i enjoy my work")
print("if you have a improved note , tell me please!!!")


# Any results you write to the current directory are saved as output.


# <h1>Explore and analyze data: structure, typedata and integrity </h1>

# In[ ]:


Data = pd.read_csv("../input/cities_r2.csv") # take data 
Data.info()
for x in Data.columns: # see the integrity of data: do those have a NaN??
    if Data[x].isnull().values.any():
        print( x,' has a value NaN.')
else:
    print ("all are cleaned apparently")


# In[ ]:


Data.head(5) # see the structure of datas


# This is to see the summary of data using of size of city within each states. We can use to bound state by size to analyze them.

# In[ ]:


DataN = Data.groupby("state_name").filter( lambda x : len(x) > 4 ) # select states with size > 4
Data = DataN
print( Data.groupby("state_name").size() )


# <h1>Question 1: Which cities are on each states? </h1>

# In[ ]:


#NameState = { x:[] for x in DataN["state_name"].unique() } # first version
from collections import defaultdict
NameState = defaultdict(list)
for x in Data.itertuples():
    NameState[ x.state_name ].append( x.name_of_city)
print( NameState["TAMIL NADU"]) # put the name of state you want


# <h1>Question 2: How many people on average are literate in each state by gender?</h1>

# In[ ]:


Q2 = Data.groupby("state_name").agg({"literates_male": np.mean , "literates_female": np.mean }).sort(["literates_male"], ascending = True )
Q2.plot(kind = "barh")


# We can see that Maharashtra , NCT of delhi and Gujarat are first top 3 on literates male while Uttarakhand, bihar and haryana are last.

# <h1>Question 3: How stable are the states with respect to their populations of cities?</h1>

# In[ ]:


DataN = Data[[ "state_name", "population_total"] ].groupby("state_name").plot(kind="box")


# <h1>Question 4 : How many literates are graduate within each states? </h1>

# In[ ]:


DataN = Data[["state_name", "literates_total","total_graduates"]].groupby("state_name").agg({"literates_total":np.sum , "total_graduates":np.sum})
DataN["graduates_by_literates_rates"] = DataN["total_graduates"]/DataN["literates_total"]
DataN[["graduates_by_literates_rates"]].sort(["graduates_by_literates_rates"], ascending= True).plot(kind="barh")


# Uttarakhand have greater probability of graduates from his literates while Gujarat have lower probability.

# <h1>Question 5: Which is the most literate state? Which is the least alphabet?</h1>

# We take the literate of the population of each state as the solved

# In[ ]:


DataN = Data[["state_name", "population_total", "literates_total"]].groupby("state_name").agg({"population_total":np.sum,"literates_total":np.sum})
DataN["literates_rate"] = DataN["literates_total"]/DataN["population_total"]
DataN[["literates_rate"]].sort(["literates_rate"], ascending = True ).plot(kind = "barh")


# Kerala, Tamil Nadu and Orissa are top on the most literates states while Rajasthan, Bihar and Uttar Pradesh are the least literates states. The value of probalility of literates states' at least 68%

# <h1>Question 6: Which is the most graduate state? and the least graduate state?</h1>

# We take the graduate of the population of each state as solved

# In[ ]:


DataN = Data[["state_name", "population_total", "total_graduates"]].groupby("state_name").agg({"population_total":np.sum,"total_graduates":np.sum})
DataN["graduates_rate"] = DataN["total_graduates"]/DataN["population_total"]
#DataN[["graduates_rate"]].sort(["graduates_rate"], ascending = True ).plot(kind="barh")


# The minimum value probability at least 11%. Gujarat, Bihar and Rajasthan are the least graduates state while Uttarakhand, NCT of delhi and Haryana.

# <h1>Question 7: Which is the most child rate state on average ? and the least child rate state on average ? </h1>

# child rate  is 0 - 6 old year child

# In[ ]:


#DataN = Data[["state_name","0-6_population_total"]].groupby("state_name").agg({ "0-6_population_total" : np.mean })
#DataN.sort_values(by="0-6_population_total", ascending = True).plot(kind = "barh")


# NCT of delhi, Maharashtra and Gujarat are top on the most child rate states while Orissa , Uttarakhand and West Bendal are last.

# <h1>Question 8:Which do city have most child rate? most graduates?  most literates?</h1>

# We only show 10 first cities

# In[ ]:


Data1 = pd.read_csv("../input/cities_r2.csv")
Data1 = Data1[["name_of_city","literates_total","total_graduates","0-6_population_total"]].set_index("name_of_city")
Data0 = Data1[["0-6_population_total"]].sort_values(by = "0-6_population_total", ascending= True).nlargest(10,"0-6_population_total")
Data0.plot(kind= "barh")


# In[ ]:


Data0 = Data1[["total_graduates"]].sort_values(by="total_graduates",ascending = True ).nlargest(10,"total_graduates")
#Data0.plot(kind="barh")


# In[ ]:


Data0 = Data1[["literates_total"]].sort_values(by="literates_total",ascending = True ).nlargest(10,"literates_total")
Data0.plot(kind = "barh")


# <h1>Question 9:  Which is the most population state?</h1>

# In[ ]:


Data0 = Data[["state_name","population_total"]].groupby("state_name").agg({"population_total":np.sum})
Data0.sort_values(by="population_total").plot(kind = "barh")


# the most population is Maharashtra and the least population is Uttarakhand

# <h1>Question 10: is there men are graduater than women?</h1>

# In[ ]:


print( Data[["male_graduates","female_graduates"]].sum(axis = 0 ) )


# there are more graduated men than women
