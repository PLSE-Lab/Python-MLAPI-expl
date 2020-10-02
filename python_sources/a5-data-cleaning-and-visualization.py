#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# The smiley face-based rating system that Seattle uses for restaurant inspections has always fascinated me. I've also been particularly intrigued by this system as there have been a few restaurants that I've liked that have received very poor marks in the past.
# 
# I'd love to dig into this data and see what it's all about!
# 
# First, let's import the data.

# In[ ]:


import pandas as pd

# Request functionalities from a bunch of different services
import pandas_datareader.data as web

# Plot library... "as" creates an alias
import matplotlib.pyplot as plt

# Helps with dates and times
import datetime as dt

# Some modules from Google that were recommended
import math
import seaborn as sns

df = Food_Establishment_Inspection_Data = pd.read_csv("../input/king-county-food-inspection/Food_Establishment_Inspection_Data.csv")


# Let's see what this data looks like. I'll print out the first few rows to get a good idea of what's there.

# In[ ]:


df.head(5)


# This is more data than I need. Let's consolidate the rows.

# In[ ]:


four_col = df[['Program Identifier', 'Inspection Date', 'Inspection Result', 'Violation Description']]
four_col


# Great! Now I want this format but also look up the specific business name.

# In[ ]:


# From above, this is the code to consolidate the table into three rows
# three_col = df[['Program Identifier', 'Inspection Result', 'Violation Description']]
# three_col

# And this is the code to identify rows with "Fremont Bowl". 
# df_filter = df['Program Identifier'].isin(['FREMONT BOWL'])
# df[df_filter]

df_filter = four_col['Program Identifier'].isin(['CAFE SELAM'])
four_col[df_filter]


# Gosh, this looks like a huge dataset! I'll use ".tail()" to see where it ends.

# In[ ]:


four_col.tail(5)


# Now let's do some data visualization!
# 
# I'd like to see what the inspection results in 2020 are. I'll break this down into smaller chunks.

# In[ ]:


# Let's drill this down to two columns
two_col = df[['Inspection Date', 'Inspection Result']]
two_col

# And let's see what the other results could be
two_col.head(20)
# Seems like there are only four kinds of inspection results


# In[ ]:


def inspections(results):
    if (results == "Complete"):
        return "Completed"
    elif (results == "Satisfactory"):
        return "Passed"
    elif (results == "Incomplete"):
        return "Incomplete"
    elif (results == "Unsatisfactory"):
        return "Unsatisfactory"
    else:
        return "Unknown"

df['Inspection Result'].apply(inspections).value_counts().plot(kind='bar',rot=0)


# Good news is that this worked! Bad news is that... yikes, there are a LOT of unsatisfactory inspection results. Now how can I do this with the addition of the year?
# 
# First, let's convert the inspection date to a more consumable format. Currently, it reads as "NaN".

# In[ ]:


# Let's get that date converted to date-time
df['Inspection Date'] = pd.to_datetime(df['Inspection Date'])

# And this will just keep the date and get rid of the rest
# Thanks for the help, Hannah! 
df['Inspection Date'] = df['Inspection Date'].map(lambda x: x.year).fillna("")

df['Inspection Date']


# Great, it looks like that worked. Now I'll see if I can plot these two data points together.
# 
# In plain English, I want a bar chart that will show the Inspection Rating for each Inspection Date (or, in this case, Year).
# 
# So first, let me plot the years out.

# In[ ]:


df['Inspection Date'].value_counts().plot(kind='bar', figsize=(10,10), rot = 0)


# So it's kind of odd for the years to be floats - especially since I converted the floats into integers (well, objects) previously.
# 
# Let's try this as a line graph.

# In[ ]:


df['Inspection Date'].value_counts().plot(kind='line', figsize=(10,10))


# Well, that looks good. But the dates are still saved as floats. Out of curiosity, what are the number of inspections per year?

# In[ ]:


# This should do the trick...
c = df['Inspection Date'].value_counts()
print (c)


# Now I want to do the following...
# 
# * Have a chart for each year
# * Graph the inspection result as the x axis and the # of inspection for said year
# 
# In other words, how many results were "Complete" in 2019? How many results were "Unsatisfactory" in 2018? For the total inspections in 2019, how many went into which categories?

# In[ ]:


a = df['Inspection Result'].apply(inspections).value_counts()
b = df['Inspection Date'].value_counts()

df.plot(x=[a, b], kind="bar")


# This probably should've done the trick. Unfortunately, I'm getting this cryptic error that I'm getting blocked on. I believe this still has something to do with the "Inspection Date" :(
# 
# What happens when I do this manually? And yes - I know, this defeats the whole purpose of this exercise. But I have to see what this would look like...

# In[ ]:


a = df['Inspection Result'].apply(inspections).value_counts()
c = df['Inspection Date'].value_counts()

df2 = pd.DataFrame({
    'year':['2006','2007','2008','2009','2010','2011','2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'],
    'inspections':[12245,12901,13339,13447,15583,17044,19165, 19849, 22665, 22769, 23579, 25709, 27578, 27752, 3191],
})

df2.plot(kind='bar',x='year',y='inspections',rot=0, figsize=(10,10))
plt.show()


# This is exactly what I want (and - aside from the floats - is identical to what I did earlier), but I'd like to have the breakdown of inspection results as well per year. However, due to the incompatility of the dates as floats, I don't think this is possible (or, at least, is limited by my foundational knowledge of pandas).
