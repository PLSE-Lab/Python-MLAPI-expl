#!/usr/bin/env python
# coding: utf-8

# # EDA & Pandas Tutorial
# Name: Teeny Chen <br>
# Email: chihtung@usc.edu <br>
# Data source: [https://www.kaggle.com/akhilv11/border-crossing-entry-data](https://www.kaggle.com/akhilv11/border-crossing-entry-data)

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


# ## 0. The Data
# The Bureau of Transportation Statistics (BTS) has been collecting records of inbound crossings at the US borders (Canada & Mexico) since 1996. The data specifies the port of entry (with its respective port code), State, date, as well as the mode of entry. This pandas tutorial will serve as an Exploratory Data Analysis (EDA) to derive insights from this vast amount of data.

# ## 1. Getting to know the data
# a. Load the dataset and display the first 5 rows to get a sense of what the data looks like.
# 
# > Use <code>.head()</code>
# > ##### Note that I have cleaned the data in the Date column so that it is in the datetime format.

# In[ ]:


entry = pd.read_csv('/kaggle/input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
entry['Date'] = pd.to_datetime(entry['Date'])
entry.head()


# b. Now, display the last ten rows.
# > Hint: use <code>.tail()</code>. Default is 5.

# In[ ]:


entry.tail(10)


# c. How many rows and columns does the dataframe have?
# > Use the <code>shape</code> method

# In[ ]:


entry.shape


# ## 2. Explore the data
# a. When is the first and last recorded entry dates in this dataset?
# > Select the <code>Date</code> column and apply <code>min()</code> or <code>max()</code>.

# In[ ]:


print('First recorded entry date: ' + str(entry.Date.min()))
print('Last recorded entry date: ' + str(entry.Date.max()))


# b. Display only entry records from <code>2019</code>. Sort from oldest to newest and by <code>Port Code</code>.
# > Use boolean indexing and <code>.sort_values()</code>. Since you are sorting by 2 columns, put the columns in a list.

# In[ ]:


entry[entry.Date >= '2019-01-01'].sort_values(['Date', 'Port Code'])


# c. How many of the entries are <code>Pedestrians</code>? (i.e. How many inbound <code>Pedestrians</code> are there?)
# > Use <code>.query()</code> to select only Pedestrian records and use <code>sum()</code> for the number of entries

# In[ ]:


entry.query('Measure == "Pedestrians"').Value.sum()


# ## 3. Split-Apply-Combine
# a. How many entries are there for each <code>Measure</code> in the year of <code>2018</code>?
# > Use <code>.groupby()</code> to group values that are alike, then count the number of records.

# In[ ]:


measure_count = entry[entry.Date == '2018'].groupby('Measure').Measure.count()
measure_count


# b. Plot a bar chart for the data above.
# > <code>.plot</code> to plot a graph. Specify bar for bar charts.

# In[ ]:


measure_count.plot.bar()


# c. Display the top 10 ports that have the most number of entry since <code>2000-01-01</code>.
# > Use <code>.size()</code> to get the number of entries. Use <code>.iloc[:]</code> to select the top 10 rows of the result.

# In[ ]:


entry[entry.Date > '2000-01-01'].groupby('Port Name')     .size()     .sort_values(ascending=False)     .iloc[:10]


# d. How many ports are there on the <code>US-Canada Border</code>?

# In[ ]:


len(entry[entry.Border == 'US-Canada Border'].groupby(['Border','Port Name']))

