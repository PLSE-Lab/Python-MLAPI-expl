#!/usr/bin/env python
# coding: utf-8

# # Cleaning data with Python - Challenge day 3 - Parsing dates
# 
# Rather than forking, I prefer to start with a clean workbook and make it my own, so let's get started on [day 3 of Rachael's cleaning data in Python][1] 5-dayer.
# 
# Today, it's about parsing dates, something I'm pretty happy with in both base R and using lubridate, but Python? Not so much. Time to being working towards changing that...
# 
# [1]: https://www.kaggle.com/rtatman/data-cleaning-challenge-parsing-dates

# In[24]:


# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)


# ### Your turn! Check the data type of the Date column in the earthquakes dataframe (note the capital 'D' in date!)

# In[25]:


# take a look at the variable names
print(earthquakes.head())


# In[26]:


# note the capital 'D' indeed, let's answer the question...
earthquakes['Date'].dtype


# ### Your turn! Create a new column, date_parsed, in the earthquakes dataset that has correctly parsed dates in it. (Don't forget to double-check that the dtype is correct!)

# In[27]:


earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")


# Hmmm, an error. It's just a simple bit of code, could the issue be that this simple bit of code is a bit too rigid to deal with the dataset?
# 
# Let's try this code ([thanks Jamie!][1]) to see if we can see if there are any rows causing our error:
# [1]: https://www.kaggle.com/jsteckel

# In[28]:


print (pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y"))
mask = pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y").isnull()
print (earthquakes['Date'][mask])


# And there we have it: looks like we have three rows with our date in a different format. Let's get around that with the addition of an argument to our original, inflexible, code:

# In[29]:


earthquakes['dateParsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)


# In[30]:


earthquakes.dateParsed.head()


# dtype: datetime64. Nice.
# 
# ### Your turn! get the day of the month from the date_parsed column

# In[31]:


earthquakes['dayOfMonth'] = earthquakes['dateParsed'].dt.day
earthquakes.dayOfMonth.head()


# ### Your turn! Plot the days of the month from your earthquake dataset and make sure they make sense.

# In[33]:


# remove na's
earthquakes.dayOfMonth = earthquakes.dayOfMonth.dropna()

# plot the day of the month
sns.distplot(earthquakes.dayOfMonth, kde = False, bins = 31)


# "And that", as Jim Lovell (well, it was Tom Hanks I suppose) said, "is how we do that".
# 
# Aquarius. Signing off.
