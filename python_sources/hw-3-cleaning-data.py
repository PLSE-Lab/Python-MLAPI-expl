#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Hi, in this kernel I will try to improve my Data Cleaning skills and get used to play with Datasets
# 
# this kernel will include most of the knowledge I get from te course so far such as plots etc.
# 

# 

# In[ ]:


data = pd.read_csv("../input/countries of the world.csv")  #getting the data with pandas


# In[ ]:


data.info()  # learning more about the data


# Our very first problem seems to be data types of most columns. 
# As seen above, numerical values doesn't have a numerical data types such as int or float. But we can work on it
# 
# Let's continue to explore

# In[ ]:


data.head(10)


# In[ ]:


data.describe()


# above we see only 3 columns, which is absolutely not what we desire.
# 
# As seen above with " data.head() " our main problem seems to be the comma( , ) instad of dot( . ) if we can replace, then we can transform our data types into float

# In[ ]:


def comma_to_dot(data):
    """This function is created for the replacement of comma with dot and changing the data type to float"""
    data = str(data);   # for any type of unexpected data types
    data = data.replace(",",".");
    data = float(data);
    return data;


# In[ ]:


print(data.columns)    # checking the full names of each column


# Now we can apply our function on each column we need below.
# 
# Unfortunately, the data has column name inconsistency but we can use as " data[' GDP ($ per capita ) ' ] "

# In[ ]:


data['Pop. Density (per sq. mi.)'] = list(map(comma_to_dot,data['Pop. Density (per sq. mi.)']))
data['Coastline (coast/area ratio)'] = list(map(comma_to_dot,data['Coastline (coast/area ratio)']))
data['Net migration'] = list(map(comma_to_dot,data['Net migration']))
data['Infant mortality (per 1000 births)'] = list(map(comma_to_dot,data['Infant mortality (per 1000 births)']))
data['GDP ($ per capita)'] = list(map(comma_to_dot,data['GDP ($ per capita)']))
data['Literacy (%)'] = list(map(comma_to_dot,data['Literacy (%)']))
data['Phones (per 1000)'] = list(map(comma_to_dot,data['Phones (per 1000)']))
data['Arable (%)'] = list(map(comma_to_dot,data['Arable (%)']))
data['Crops (%)'] = list(map(comma_to_dot,data['Crops (%)']))
data['Other (%)'] = list(map(comma_to_dot,data['Other (%)']))
data['Climate'] = list(map(comma_to_dot,data['Climate']))
data['Birthrate'] = list(map(comma_to_dot,data['Birthrate']))
data['Deathrate'] = list(map(comma_to_dot,data['Deathrate']))
data['Agriculture'] = list(map(comma_to_dot,data['Agriculture']))
data['Industry'] = list(map(comma_to_dot,data['Industry']))
data['Service'] = list(map(comma_to_dot,data['Service']))


# In[ ]:


data.head(10)   # to see results on dataset


# In[ ]:


data.describe()


# In[ ]:


data.info();


# Now we have what we wanted, our data is much more easy to work with and data types are just as we would wish.
# 

# In[ ]:


data.Climate.value_counts(dropna=False)   
# it is interesting that 2.0 and 3.0 is most common but the 2.5 is least common one even that it's just between most common values.
# also we have a lot of NaN values.


# Let's drop the all null values since this notebook is for experimenting

# In[ ]:


data = data.dropna();

data.info();   # we still have 180 countries that we can work on easily.  


# In[ ]:


data.boxplot(figsize=(20,8), column='Literacy (%)', by='GDP ($ per capita)', grid=False)


# ## The BoxPlot above is interesting but not surprising.
# As seen, Literacy has very big effect on the income of people and government which is not a surprise. The surprise is that there is not many outliner values on this plot which means there is almost no exceptional cases for it.  

# Let's check for correlation of our dataset.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.clf()
plt.figure(figsize=(20,12))
sns.heatmap(data.corr(),annot=True,fmt='1.1f')


# I see that Service is very correlated with other columns such as "Birthrate" , "Agriculture" , "Phones" , "Infant Mortality".
# 
# Also, GDP per capita and Birtrate is very correlated with many different Columns too.

# I want to fix our Region values too, they have too much space which makes it hard to read.

# In[ ]:


b = set(data.Region);
print(b)   # seems like we have a Lot of spaces, considering that they use storage unnecesarily: I want to delete whole spaces.


# In[ ]:


def SpaceRemover(data):
    """This function is created for the replacement of Space with empty"""
    data = str(data);   # for any type of unexpected data types
    data = data.replace(" ","");
    return data;


# In[ ]:


data['Region'] = list(map(SpaceRemover,data['Region']))


# In[ ]:


print(set(data.Region))  # now not super clear but still better.


# In[ ]:


plt.clf()
plt.figure(figsize=(15,10));
plt.scatter(data.Climate , data.Region, alpha=0.1, s=200, c="blue")  

# I made 0.1 opacity to see how often it repeats easily
# darker the blue, often the appearence of that climate.

plt.show()


# I will try to improve my data manipualtion skills such as melting or pivoting the data on the dataset.

# In[ ]:


melted_data = pd.melt(frame=data, id_vars="Country", value_vars=['Literacy (%)','Birthrate','Phones (per 1000)'])

melted_data  # the picked variables are all correlated, so when I pivot it will has more sense.


# In[ ]:


pivotted_melt = melted_data.pivot(index='Country',columns="variable",values='value')

pivotted_melt  # we can see that how dependently numbers change in each column easily.


# Example usage of assert on my dataset can be like:

# In[ ]:


# assert pivotted_melt.Birthrate.dtype == np.dtype(int)   ## would return an error.

assert pivotted_melt.Birthrate.dtype == np.dtype(float)   # returns no error because data type is really float


# In[ ]:


# assert data.columns[1] == "Efe"  ## would return an error 

assert data.columns[1] != "Efe" # in here I assert negatively. (not "Efe" is true)


# Example of Concatenaing data could be like this:

# In[ ]:


dataHead = data.head(10);               dataTail = data.tail(10);

concated_data = pd.concat([dataHead,dataTail],axis=0,ignore_index=1)


# In[ ]:


concated_data   # Rigth Below we see that by "concat" operation of numpy we easliy created new data from 2 data(s). 


# In[ ]:


# this notebook is created for the 3rd homework of Data Science by DATAI on Udemy.


# In[ ]:




