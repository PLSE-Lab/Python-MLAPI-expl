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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dframe = pd.read_csv("../input/appendix.csv")


# In[ ]:


dframe.head()


# In[ ]:


ax = dframe.groupby("Institution").size().to_frame().plot(kind='bar',figsize=(10,10))
ax.set_ylabel("Count")
ax.set_xlabel("Institution")
ax.set_title("Count of program by institution")


# In[ ]:


dframe["year"]=dframe["Launch Date"].str.split('/').str[-1]


# In[ ]:


Unique_Courses = set(dframe["Course Number"])
len(Unique_Courses)


# The data is not presenting each course as unique.  A course that first launched in 2012 can also appear in 2013, 2014 and 2015 (assuming it stayed active and was not discontinued).  Additionally, there can be multiple occurrences of a course within the year as well.  It seems that the data has tracked multiple sessions of a course offered in each year, perhaps one session per semester.  Each session contains its own summary statistics.*emphasized text*

# I just showed that the number of unique course numbers is 188 by casting the "Course Number" series as a set and applying the length function.  Could also use a the .unique() method of a series

# In[ ]:


len(dframe["Course Number"].unique())


# When plotting number of courses by year, it should show the total number of active classes each year.  NOT the number of classes added in a given year

# What is the breakdown of the active courses by institution by year?

# In[ ]:


ax = dframe.groupby(["year","Institution"]).size().to_frame().unstack()[0].plot(kind="bar",figsize=(10,10))
ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.set_title("Course count by institution by year")


# Looks like a major buildup in classes through 2015, with a steep dropoff between 2015 and 2016

# In[ ]:


participants = dframe.groupby(["Course Subject","Institution"])["Participants (Course Content Accessed)"].sum().to_frame()
participants


# In[ ]:


participants.unstack().plot(kind="bar",figsize=(10,10))


# Let's do breakdown by sex.  I want to create columns showing total female and male participants based on percentage male, percentage female, and course accessed numbers

# In[ ]:


dframe["No Male"]=(dframe["% Male"]/100)*(dframe["Participants (Course Content Accessed)"])
dframe["No Female"]=(dframe["% Female"]/100)*(dframe["Participants (Course Content Accessed)"])


# In[ ]:


ax = dframe.groupby("Course Subject")[["No Male","No Female"]].sum().plot(kind="bar",figsize=(10,10))
ax.set_xlabel("Course Subject")
ax.set_ylabel("Count of Attendees")
ax.set_title("Course Subject Attendance by Sex")


# Looks like these open courses have been male dominated, although the gender gap is significantly more narrow in the humanities.

# In[ ]:


dframe.head()


# In[ ]:


dframe["Instructors"].str.split(",")


# In[ ]:




