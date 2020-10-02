#!/usr/bin/env python
# coding: utf-8

# # Connect to SQLite3
# First we must import the necessary libraries we may be using, and connect to the SQLite3 database which is part of the dataset.

# In[3]:


import pandas as pd # reading files
import sqlite3 # sqlite database

# open the database connection
conn = sqlite3.connect('../input/database.sqlite3')


# # A simple query...
# We request the number of elements in the "courses" table. This is equivalent to the number of courses taught at UW Madison since 2006.

# In[9]:


# request count
pd.read_sql("SELECT COUNT(*) AS count FROM courses", conn)


# # More complex examples

# In[10]:


# let's see a few courses, sorted by uuid (random essentially)
pd.read_sql("SELECT * FROM courses ORDER BY uuid ASC LIMIT 10", conn)


# In[6]:


# let us find a CS instructor who has 'remzi' in their name
pd.read_sql("SELECT * FROM instructors WHERE name LIKE '%remzi%'", conn)


# In[8]:


# get courses that have been taught by people with 'remzi' in their name
courses = pd.read_sql("""
  SELECT DISTINCT courses.* 
  FROM courses
  JOIN course_offerings co ON co.course_uuid = courses.uuid
  JOIN sections s ON s.course_offering_uuid = co.uuid
  JOIN teachings t ON t.section_uuid = s.uuid
  JOIN instructors i ON i.id = t.instructor_id
  WHERE i.name LIKE '%remzi%'
""", conn)

courses

