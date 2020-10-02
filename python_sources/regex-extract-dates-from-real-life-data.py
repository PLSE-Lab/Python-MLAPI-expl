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


# # Extracting dates from a real-life dataset
# In this notebook, we will be working with the messy data and using regex to extract relevant information from the data.
# Each line of the file - dates.txt file corresponds to a medical note. Each note has a date that can be extracted, but each date is encoded in one of the many formats. 
# 
# We will look into all the types of dates that can be there and extract them. 
# The goal of this notebook is to correctly identify all of the different date variants encoded in the dataset and to properly normalize them.
# 
# Here is the list of some of the variants we might encounter in this dataset:
# - 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# - Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
# - 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# - Feb 2009; Sep 2009; Oct 2010
# - 6/2008; 12/2009
# - 2009; 2010

# # Starting off slow
# We will first look at how regex can extract information and make Dataframes in sample examples. Let's consider all the dates formats that may exist in any dataset.
# 
# ## Date variations for 23rd October 2002
# * 23-10-2002
# * 23/10/2002
# * 23/10/02
# * 10/23/2002
# * 23 Oct 2002
# * 23 October 2002
# * Oct 23, 2002
# * October 23, 2002

# In[ ]:


# importing important libraries
import re
import pandas as pd

# Making a sample example with all the variations of the date.
dateStr = "23-10-2002\n23/10/2002\n23/10/02\n10/23/2002\n23 Oct 2002\n23 October 2002\nOct 23, 2002\nOctober 23, 2002\n"
print(dateStr)


# - '\d'  is used to search for digits [0-9]
# - '{n}' is used to specify the number of matches that have to be there
# 
# ```findall``` function can be used to get all the data that matches.

# In[ ]:


re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', dateStr) # This will return a list of all the dates with '/' or '-' in between.


# Notice the difference in both the commands, we have {4} in first command and {2, 4} in second. This facitilates the number of matches that needs to be done. In the first command we strictly want 4 matches of digits but in second command we will take number of matches between 2 and 4.

# In[ ]:


re.findall(r'\d{2}[/-]\d{2}[/-]\d{2,4}', dateStr)


# Now, we are considering the text data.

# In[ ]:


re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}', dateStr)


# We are add an additional [a-z]* at the end of month column to check if additional characters are present after the specified match. 
# 
# Example - October, We matched Oct and then took all that was present after it.

# In[ ]:


re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}', dateStr)


# # Diving into real mess
# Now, let's dive into the real dataset and find all the dates in here. Here we will learn about the function ```extractall``` which will be used to create dataframes of the 3 values - month, day and year.

# In[ ]:


doc = []
with open('../input/textdata/dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head()


# First we are looking at the format:
# - mm/dd/yyyy
# - mm-dd-yyyy
# 

# In[ ]:


# Extracting the dates and storing it as a dataframe
dates_extracted = df.str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))')

# Marking the index that are used so that it does not get used again when extracting with some other format.
index_left = ~df.index.isin([x[0] for x in dates_extracted.index]) 
dates_extracted.head()


# Now, we are trying to find the format:
# - mm/dd/yy
# - mm-dd-yy

# In[ ]:


# Extracting the dates and storing it as a dataframe, notice here we are using the index_left to check only those which are left.
dates_extracted_temp = df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))')

# Marking the index that are used so that it does not get used again when extracting with some other format.
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

# Deleting the extra unnecessary columns.
del dates_extracted_temp[3]
del dates_extracted_temp[4]

dates_extracted_temp.head()


# In[ ]:


# Appending the values in the main data frame - dates_extracted
dates_extracted = dates_extracted.append(dates_extracted_temp)


# We'll look at the format:
# - mm/dd/yyyy
# - mm-dd-yyyy 
# 
# but, in text form.  Example - 23 Oct 2020

# In[ ]:


dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))'))
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])


# Here we are looking for the characters after the digits in day, ex - October 23rd 2020

# In[ ]:


dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))'))
del dates_extracted[3]
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])


# In[ ]:


dates_extracted.shape


# This shows that we have 230 complete values in the dataset, we will not extract the dates with missing values starting with without days.

# In[ ]:


# Without day
dates_without_day = df[index_left].str.extractall('(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))')
dates_without_day = dates_without_day.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))'))

# Replacing the missing value with '1'
dates_without_day['day'] = 1

# Appending in the main dataframe
dates_extracted = dates_extracted.append(dates_without_day)
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])


# After this we will look for only years.

# In[ ]:


# Only year
dates_only_year = df[index_left].str.extractall(r'(?P<origin>(?P<year>\d{4}))')
dates_only_year['day'] = 1
dates_only_year['month'] = 1
dates_extracted = dates_extracted.append(dates_only_year)
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])


# In[ ]:


dates_extracted.shape


# # Cleaning the data
# As we can see, we have extracted all the dates from the dataset, but the problem arises that we don't have uniform values in the colums. We have numbers and text files both in months column, years are present in both format - yy and yyyy. Now we will deal with this and convert the whole dataset to have uniform values.

# In[ ]:


# Year
dates_extracted['year'] = dates_extracted['year'].apply(lambda x: '19' + x if len(x) == 2 else x)
dates_extracted['year'] = dates_extracted['year'].apply(lambda x: str(x))


# In the months column, we are converting all the text into their corresponding numbers. Jan -> 1, Oct -> 10.
# Also, we are adding some additional words which can be considered as a typo in a real-life dataset. Age -> Aug

# In[ ]:


# Month
dates_extracted['month'] = dates_extracted['month'].apply(lambda x: x[1:] if type(x) is str and x.startswith('0') else x)
month_dict = dict({'September': 9, 'Mar': 3, 'November': 11, 'Jul': 7, 'January': 1, 'December': 12,
                   'Feb': 2, 'May': 5, 'Aug': 8, 'Jun': 6, 'Sep': 9, 'Oct': 10, 'June': 6, 'March': 3,
                   'February': 2, 'Dec': 12, 'Apr': 4, 'Jan': 1, 'Janaury': 1,'August': 8, 'October': 10,
                   'July': 7, 'Since': 1, 'Nov': 11, 'April': 4, 'Decemeber': 12, 'Age': 8})
dates_extracted.replace({"month": month_dict}, inplace=True)
dates_extracted['month'] = dates_extracted['month'].apply(lambda x: str(x))


# In[ ]:


# Day
dates_extracted['day'] = dates_extracted['day'].apply(lambda x: str(x))


# Now we will make a new column named 'date' with our cleaned data.

# In[ ]:


# Cleaned date
dates_extracted['date'] = dates_extracted['month'] + '/' + dates_extracted['day'] + '/' + dates_extracted['year']
dates_extracted['date'] = pd.to_datetime(dates_extracted['date'])


# In[ ]:


dates_extracted.head()


# In[ ]:


# Sorting the dates and storing them index wise in a new dataframe object.
dates_extracted.sort_values(by='date', inplace=True)
dates_from_text = dates_extracted.loc[dates_extracted.index, 'date']
dates_from_text


# #### I really hope you liked the notebook. Please provide some feedbacks. Thank you.
