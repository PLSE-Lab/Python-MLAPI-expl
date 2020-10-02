#!/usr/bin/env python
# coding: utf-8

# #  Time of accidents

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/us-accidents/US_Accidents_May19.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Imports

# To start off we do the usual import like pandas etc. and as extra we need(just because I used it in this solution) datetime

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from datetime import datetime
import dateutil.parser


# ### Reading data and change format

# In[ ]:


dataset = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")


# Because we want to know about the time of Accidents we need to find a variable that indicates this. I choose the variable Start_Time, which tells us the start time of the accident. Before we can use it we need to find the type of this column.

# In[ ]:


type(dataset["Start_Time"][1])


# In[ ]:


dt_object = dataset["Start_Time"][1:30].apply(lambda x: dateutil.parser.parse(x))


# In[ ]:


dt_object.head()


# As string we can not work very well with this data. It will be a lot easier, if we can convert it into a datetime object, which supports different methods like day() and hour(). We therefore add a new Time Variable called "Time_added".
# 
# If Kaggle would have a higher Python version, than one could use the method fromisoformat() like this:
# dataset["Time_added"] = dataset["Start_Time"].apply(lambda x: datetime.fromisoformat(x))
# This new method is way faster in calculation than this dateutil.parser
# 
# If your run it it is going to need some minutes to add the column. 

# In[ ]:


dataset["Time_added"] = dataset["Start_Time"].apply(lambda x: dateutil.parser.parse(x))


# Now lets look if we can work with this new datetime object and start with a method called hour which returns the hour of the accident.

# In[ ]:


dataset["Time_added"][0:4].apply(lambda timestamp: timestamp.hour)


# We can now for example get the hour at which the accident happend with relative ease.

# ### Visualization and results

# We now want to see at which hours do the most accidents happen. For this we need a seaborn countplot(or any other plot) and apply the hour method to our new column. I made the plot a little bigger so it is easier to diffrentiate between single hours.

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(x=dataset["Time_added"].apply(lambda timestamp: timestamp.hour))


# Now as we can see the most accidents happen around in the hours from 7 to 9 in the morning. Btw if we want a countplot for a different category of time our work now pays off because we just have to change the method which is called at the end. I give an example below for month.

# In[ ]:


a4_dims = (10, 7)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(x=dataset["Time_added"].apply(lambda timestamp: timestamp.month))


# At last we now want a list of hours with the corresponding number of accidents. Again our work from before pays off. We can easily use the apply function to generate such a list. If we want it for days we can get it, months no problem, year easy.

# In[ ]:


time_list = dataset["Time_added"].apply(lambda timestamp: timestamp.hour).value_counts()


# In[ ]:


time_list


# This is my solution, I am happy if you got any suggestions, comments or questions.

# In[ ]:




