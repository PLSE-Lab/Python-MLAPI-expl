#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from IPython.display import Image


# Let's import data from a website this time. Many websites will have an API, or an application programming interface, that allows users to obtain data from a webpage. Github is the standard for open source code, almost like a refined social media for coders. Let's call a github url for a csv.

# In[ ]:


Image('/kaggle/input/github/github.png')


# The url is the page full of data that comes up when you click the 'raw' version of the data.

# In[ ]:


url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/06-20-2020.csv'
covid=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/06-20-2020.csv')
covid.head()


# This is a dataset for covid-19 data by U.S. state on 6/21. The .head() tag just means that only the first five rows will be displayed. I will use this on a lot of printed outputs to avoid dataframes taking up the whole page.

# In[ ]:


#directory of all methods that may be called on a dataframe
dir(pd.DataFrame())


# One of the most important methods that I've used for almost any dataframe is .isin(). This checks to see if a specified item or list of items are present in dataframe or subsection of dataframe.

# In[ ]:


#search the entire dataframe for elements that contain the phrase 'US'

#covid is the dataframe we are searching
#'US' is the phrase we are searching for
    #'US' is in brackets because the .isin() method is bulit to accept a list of items, so even if you're searching for one
    #item, it still needs to be in brackets
covid.isin(['US']).head()


# Notice that all of the cells are filled with true or false. This one call only checks to see whether the element is present in the frame. Looking at the output, it makes sense that only the elements in the Country_Region are true.
# 
# What if we want to see the actual elements?
# * we can embed the above code in the dataframe, and it will only return the True elements.

# In[ ]:


covid[covid.isin(['US'])].head()


# This isn't very helpful because we only see one column of data. The way that the .isin() method is useful is when you search a specific column.
# 
# Rather than searching the whole dataframe, let's search the only the Country_Region column.

# In[ ]:


covid['Country_Region'].isin(['US']).head()


# This gives a much more concise output. Again, all of the columns return true, but we're narrowing our search to the data that's relevant.
# 
# Now, embedding this code in the covid dataframe will yield a much better result than last time.

# In[ ]:


covid[covid['Country_Region'].isin(['US'])].head()


# Last time we searched the entire dataframe, so everything besides the Country_Region was false. When we embed this in the dataframe, we got 'NaN' for all of the other columns. Now, because we only searched the Country_Region column, the rest of the column data is preserved

# Let's do a more selective example- try to extract the Alabama data.

# In[ ]:


covid.isin(['Alabama']).head()


# Only the first row's state entry is true. 

# In[ ]:


covid[covid.isin(['Alabama'])].head()


# But when we try to get the actual data, most of it is empty because we're searching the entire dataframe.
# 
# Searching the 'Province_State' column will get us what we want.

# In[ ]:


covid['Province_State'].isin(['Alabama']).head()


# In[ ]:


covid[covid['Province_State'].isin(['Alabama'])].head()


# Here we go, now we've got the data for Alabama.
# 
# For this example there are many other ways to get this data. Knowing that each state has one entry, we could've just extracted the first row, because we can see that Alabama is the first state.

# In[ ]:


covid.head(1)


# Let's do another example where getting specific data is not so easy to recognize.

# In[ ]:


pollution=pd.read_csv('/kaggle/input/uspollution/pollution_us_2000_2016.csv')
pollution.head()


# This dataset actually has over a million entries...it's impressive that kaggle can perform tasks in decent time.
# 
# Getting data for a state isn't as easy as finding the specific row. Each state has many entries spread out over a million+ rows.

# In[ ]:


#get data for state of missouri
pollution['State'].isin(['Missouri'])


# When the list is so long, it can be hard to tell if this yielded any true matches. Add the .value_counts() method to make sure that there indeed are true matches.

# In[ ]:


pollution['State'].isin(['Missouri']).value_counts()


# Yes- over 19,000 entries for Missouri. Let's create a new dataframe for Missouri by embedded this code in the full dataframe.

# In[ ]:


missouri=pollution[pollution['State'].isin(['Missouri'])]
missouri


# Notice that the number of rows indicated on the bottom is the same as the number of true matches we found in the State column.

# Another feature of.isin() is to search for multiple elements at a time, whether they be in the same column or different columns.

# In[ ]:


#get data for Missouri and Illinois
pollution['State'].isin(['Missouri','Illinois']).value_counts()


# In[ ]:


#full data for both states
pollution[pollution['State'].isin(['Missouri','Illinois'])]


# In[ ]:


#when there a many columns, this is helpful to see what data you can look at
pollution.columns


# Another useful tool is to search two columns at once. Let's say you want all data from Missouri, but only site measurements in which CO was measured to be greater than 0.5 parts per billion. 

# In[ ]:


pollution['State'].isin(['Missouri']).value_counts()


# In[ ]:


#direct comparison doesn't require .isin()
(pollution['CO Mean']>0.5).value_counts()


# Let's combine these two conditions to see if any Missouri surpassed the indicated threshold.

# In[ ]:


((pollution['CO Mean']>0.5) & pollution['State'].isin(['Missouri'])).value_counts()


# 7,206 entries met both conditions. Let's put this into a dataframe.

# In[ ]:


pollution[(pollution['CO Mean']>0.5) & pollution['State'].isin(['Missouri'])].head()


# That long line of code might be tricky to understand as one, so I'll present a method that might make it easier.

# Save each condition as a variable so that it's easier to call later.

# In[ ]:


mizzou=pollution['State'].isin(['Missouri'])
threshold=(pollution['CO Mean']>0.5)

#same dataframe as above, just utilizing saved variables
pollution[mizzou & threshold].head()


# These are all Missouri, and notice that all indeed have a measured CO mean greater than 0.5. I picked a random column and number, but there are so many possibilities to isolate the data you want.
