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


import pandas as pd
import datetime
from datetime import datetime as dt
import numpy as np


# ***** 1. Write a Python script to display the
# * * a. Current date and time
# * * b. Current year
# * * c. Month of year
# * * d. Week number of the year
# * * e. Weekday of the week
# * * f. Day of year
# * * g. Day of the month
# * * h. Day of week****

# In[ ]:


d1=datetime.date.today()
d1


# In[ ]:


print('current year: ',d1.strftime("%Y"))
print('month of year: ',d1.strftime("%B"))
print('weak no.: ',d1.strftime("%W"))
print('weakday: ',d1.strftime("%w"))
print('day of year: ',d1.strftime("%j"))
print(' Day of the month: ',d1.strftime("%d"))
print('Day of week: ',d1.strftime("%A"))


# 2. Write a Python program to convert a string to datetime.
# Sample String : Jan 1 2014 2:43PM
# 
# Expected Output : 2014-07-01 14:43:00

# In[ ]:


s = pd.Series(input("Input datetime : "))
print("String Date:")
print(s)
r = pd.to_datetime(pd.Series(s))
df = pd.DataFrame(r)
print("Original DataFrame (string to datetime):")
print(df)


# 3. Write a Python program to subtract five days from current date.
# 
# Current Date : 2015-06-22
# 
# 5 days before Current Date : 2015-06-17

# In[ ]:


from datetime import datetime, timedelta
a_date = pd.datetime.now().date()
days = timedelta(5)

new_date = a_date - days
#Subtract 5 days from a_date


print(a_date,new_date)


# 

# 4. Write a Python program to convert unix timestamp string to readable date.
# 
# Sample Unix timestamp string : 1284105682
# 
# Expected Output : 2010-09-10 13:31:22

# In[ ]:


import datetime


# In[ ]:


print(
    datetime.datetime.fromtimestamp(
        int(input("Sample Unix timestamp string"))
    ).strftime('%Y-%m-%d %H:%M:%S')
)


# 5. Convert the below Series to pandas datetime :
# 
# DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
# 
# Make sure that the year is 19XX not 20XX

# In[ ]:


DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
DoB = DoB.str[:-2]+'19'+DoB.str[-2:]
pd.to_datetime(DoB,format='%d%b%Y')


# 6. Write a Python program to get days between two dates

# In[ ]:


d1=pd.to_datetime(input('date in formet like YYYY-MM-DD ')).strftime('%Y-%m-%b')
d1=datetime.strptime(d1, '%Y-%m-%b').date()
d2=pd.to_datetime(input('date in formet like YYYY-MM-DD ')).strftime('%Y-%m-%b')
d2=datetime.strptime(d2, '%Y-%m-%b').date()

day=d2-d1
print(day.days)


# 7. Convert the below date to datetime and then change its display format using the .dt module
# 
# Date = "15Dec1989"
# 
# Result : "Friday, 15 Dec 98"

# In[ ]:


A=pd.to_datetime(input('date in formet like 15Dec1989 ')).strftime('%A, %d %b %Y')
A

