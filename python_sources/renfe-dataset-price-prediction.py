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


# In this notebook we will try to build a model to predict the price of tickets.
# 
# First of all we will see the type of data and we will perform a bit of explorations through data visualization. 

# In[ ]:


data = pd.read_csv("../input/renfe.csv")
""" WE WILL CHECK VALUES AT THE BEGINNING AND AT THE END OF THE DATASET"""
data.head(10) # USED TO CHECK THE FIRST 10 ROWS OF THE DATASET


# > 

# In[ ]:


data.tail(10) # USED TO CHECK THE LAST 10 ROWS OF THE DATASET


# As we can see above every data unless price are categorical values. So, before to study these is interesting to see what is happening with prices to know how they works.
# 
# Also, it would be very interesting to know the distances between the different origins and destination to compare between the next 3 different variables: Price, start date, distances. With these relationships we could improve the problem of chosing the cheapest way to arrive a place.
# 
# Thus, next exercise is to undestand which type of values, how are these values (null or whatever), etc.

# In[ ]:


# Lets see the price distribution
data.price.describe()


# In[ ]:


data.count()


# As it can be seen we have lost around 200K prices in the ticket sales. So, now we should take the decission about what to do with these loss data. Then, my proposal is the next one:
# 
# 1.- To check if these loss data is random along the whole dataset or not.
# 
# 2.- If it is random or if it is a set or family (in example one kind of train ha most of null values), due to the size of the dataset, I will remove these records, so that, we have enough information (as fitst estimation) to perform the pridciton. Why I am also taking this decission? Because it would not be very accurate to predict over predictions, the result could be not satisfactory
# 
# 3.- If it is not random and it is centered over a range or over a kind of data, I will remove these categories (even these with price data), why? because under my point of view it would be more efficient to focus over the rest of the categories to predict.
# 
# Let's work:

# First of all, lets see which are most popular origins, destionations and trains that are taken by people:

# In[ ]:


import matplotlib.pyplot as plt

def routes(df, group):
    for i in group:
        seti = np.unique(df[i])
        Q_i = list()
        for orig in seti:
            aux = len(df.index[data[i]==orig])
            Q_i.append(aux)
        if len(seti) >5:
            plt.barh(seti, Q_i)
            plt.show()
        else:
            plt.bar(seti, Q_i)
            plt.show()
        
routes(data, ["origin", "destination", "train_type"])


# Now that we know which are the most famous places (Madrid, Barcelona & Sevilla) in Spain (at least to take trains), lets see what is happening with prices. To do this, we will see whihc trians has not too many information about prices:

# In[ ]:


data_nan = data[data["price"].isnull()]
data_nan.head(10)


# In[ ]:


data_nan.tail(10)


# As we can see there most travels without price information are these done with ALVIA and between MADRID-SEVILLA. So, now we are going to compare these both fields to know how many tickets are with price and which are not.

# In[ ]:


from matplotlib import pyplot as plt
trainstypes = data.train_type.unique()
traincounter = list()
for trains in trainstypes:
    totaldata = len(data.index[data["train_type"]== trains])
    datanoprice = data_nan[data["train_type"]== trains]
    datanoprice = len(datanoprice.index)
    no_prices_rate = datanoprice / totaldata
    traincounter.append(no_prices_rate)
    #print ("Type of train", trains, "No price rate", no_prices_rate)

plt.barh(trainstypes, traincounter)
plt.show()


# As it is seen in the graph above, LD-AVE and ALVIA has not too many information about the price and rest of trains are more or less distributed along the data set, lets drop nan values in the datasert and we will comppare again how the origin and destination is.

# In[ ]:


data_clean = data.dropna()
routes(data_clean, ["origin", "destination", "train_type"])


# We can see that many routes has been removed over all between Madrid & Sevilla, whereas Ponferrada is quite similar to the previous state.
