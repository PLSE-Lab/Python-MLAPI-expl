#!/usr/bin/env python
# coding: utf-8

# Hello, everyone.  I am just learning about data science and analytics.  This is my first notebook on Kaggle.
# 
# My purpose with this notebook is to provide a single dataframe ("rows") that combines the most interesting features of the dataset.  I am also still learning Python, so I have tried to wrangle the dataset into a dataframe with programming that is clean and efficient.
# 
# Some high-level peeks into the data are provided, but nothing incredibly in-depth.

# In[ ]:


import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import Image
from shutil import copyfile # copy database func

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# There are many tables in this database, so for clarity, I needed to map out the relationships between the tables.

# In[ ]:


Image("../input/uw-madison-database-table-relationships/UW Madison Database Table Relationships.jpg")


# For querying the data, I got some valuable sqlite insight from Keenan Thompson's kernel:
# https://www.kaggle.com/keenant/courses-and-instructors-by-average-gpa-given
# 
# In addition to the "section_gpas" view from that notebook, I have added calculated probabilities of making at least an A, A/AB, and A/AB/B.

# In[ ]:


# move the database into a writeable format
copyfile('../input/uw-madison-courses/database.sqlite3', './database.sqlite3')

# open connection
conn = sqlite3.connect('./database.sqlite3')


# In[ ]:


# we add a new view
c = conn.cursor()
c.execute("DROP VIEW IF EXISTS section_gpas")
c.execute("""
  CREATE VIEW
  section_gpas (course_offering_uuid, section_number, gpa, num_grades, a_prob, aab_prob, aabb_prob)
  AS
  SELECT
    course_offering_uuid,
    section_number,
    (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS gpa,
    a_count + ab_count + b_count + bc_count + c_count + d_count + f_count AS num_grades,
    (1.0 * a_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS a_prob,
    (1.0 * a_count + 1.0 * ab_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS aab_prob,
    (1.0 * a_count + 1.0 * ab_count + 1.0 * b_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS aabb_prob
  FROM grade_distributions
""");


# To design the dataframe, I have selected the features from the various tables in the order I would like them to appear in the dataframe.
# 
# All table joins are LEFT joins because I want to be sure to preserve the number of records from the governing table/view that contains the grades (grade_distbibutions / section_gpas).

# In[ ]:


rows = pd.read_sql("""
  SELECT
    su.abbreviation AS prefix, c.number AS number, co.name AS course_name, su.name AS subject_name,
    se.number AS section, co.term_code AS term, i.id AS instructor_id, i.name AS instructor_name,
    sc.start_time, sc.end_time, sc.mon, sc.tues, sc.wed, sc.thurs, sc.fri, sc.sat, sc.sun,
    r.facility_code AS facility, r.room_code AS room,
    gd.gpa, gd.num_grades, gd.a_prob, gd.aab_prob, gd.aabb_prob
  FROM section_gpas gd
  LEFT JOIN course_offerings co ON co.uuid = gd.course_offering_uuid
  LEFT JOIN sections se ON se.course_offering_uuid = gd.course_offering_uuid AND se.number = gd.section_number
  LEFT JOIN subject_memberships sm ON sm.course_offering_uuid = gd.course_offering_uuid
  LEFT JOIN courses c ON c.uuid = co.course_uuid
  LEFT JOIN rooms r ON r.uuid = se.room_uuid
  LEFT JOIN schedules sc ON sc.uuid = se.schedule_uuid
  LEFT JOIN teachings t ON t.section_uuid = se.uuid
  LEFT JOIN instructors i ON i.id = t.instructor_id
  LEFT JOIN subjects su ON su.code = sm.subject_code
  GROUP BY gd.course_offering_uuid, gd.section_number
""", conn)


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
print('The shape of rows is: %s' %(rows.shape,))
display(rows.head())


# Add some basic engineered features (class_days, class_length, and weekly_class_length) to the dataframe.

# In[ ]:


dict = {'false':False, 'true':True}
rows.replace(dict, inplace=True)
rows.start_time = pd.to_numeric(rows['start_time'], errors='coerce')
rows.end_time = pd.to_numeric(rows['end_time'], errors='coerce')

rows['class_days'] = rows[['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']].sum(axis=1)
rows['class_length'] = rows.end_time - rows.start_time
rows['weekly_class_length'] = rows.class_days * rows.class_length

print('The shape of rows is: %s' %(rows.shape,))
display(rows.head())


# In[ ]:


plt.hist(rows['num_grades'],bins=np.arange(0,1000),color='blue');
plt.xscale('log')
plt.yscale('log')
plt.title('Number of Grades Per Course Offering', fontsize=16)
plt.xlabel('Number of Grades', fontsize=16)
plt.ylabel('Course Offerings', fontsize=16)
plt.tick_params(labelsize=16)


# So, out of 193,262 course offerings, around 100,000 don't have any grades assigned!
# 
# Let's trim this dataframe, and make "rows2" out of it.

# In[ ]:


rows2 = rows[(rows.num_grades>0)]
rows2.shape


# While GPA is probably the best summarizing variable to use for comparing classes, instructors, etc., GPA doesn't give information about the grade distribution.
# 
# Some students may be more interested in their probability of making an A, AB, or B.  The plots below show how these probabilities can vary widely for a given GPA.

# In[ ]:


f = plt.figure(figsize=(15,4))
ax = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)

x = rows['gpa'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
y = rows['a_prob'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
ax.plot(x,y,'bo')
ax.grid(which='both')
ax.set_xlim([2,4.1])
ax.set_ylim([0,1.05])
ax.set_title('Probability of Getting an A', fontsize=16)
ax.set_xlabel('Class GPA', fontsize=16)
ax.set_ylabel('Probabilty', fontsize=16)
ax.tick_params(labelsize=16)

x = rows['gpa'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
y = rows['aab_prob'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
ax2.plot(x,y,'bo')
ax2.grid(which='both')
ax2.set_xlim([2,4.1])
ax2.set_ylim([0,1.05])
ax2.set_title('Probability of Getting >= AB', fontsize=16)
ax2.set_xlabel('Class GPA', fontsize=16)
ax2.set_ylabel('Probabilty', fontsize=16)
ax2.tick_params(labelsize=16)

x = rows['gpa'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
y = rows['aabb_prob'][(rows.weekly_class_length==150) & (rows.num_grades>40) & (pd.to_numeric(rows.number)<300)]
ax3.plot(x,y,'bo')
ax3.grid(which='both')
ax3.set_xlim([2,4.1])
ax3.set_ylim([0,1.05])
ax3.set_title('Probability of Getting >= B', fontsize=16)
ax3.set_xlabel('Class GPA', fontsize=16)
ax3.set_ylabel('Probabilty', fontsize=16)
ax3.tick_params(labelsize=16)


# Lets see which classes are the most common.  That is, which classes have the most grades.

# In[ ]:


temp = rows2[['prefix', 'number', 'num_grades', 'gpa', 'a_prob', 'aab_prob', 'aabb_prob']].groupby(['prefix', 'number']).agg({'num_grades':'sum', 'gpa':'mean', 'a_prob':'mean', 'aab_prob':'mean', 'aabb_prob':'mean'})
temp.sort_values(['num_grades'], ascending=False).head(50)


# As one may expect, a lot of the most common classes are the freshman/sophomore level classes.
# 
# Lets look in more detail at the most common class, ECON 101.

# In[ ]:


temp = rows2[['prefix', 'number', 'instructor_name', 'num_grades', 'gpa', 'a_prob', 'aab_prob', 'aabb_prob']][(rows2.prefix=='ECON') & (rows.number=='101')].groupby(['prefix', 'number', 'instructor_name']).agg({'num_grades':'sum', 'gpa':'mean', 'a_prob':'mean', 'aab_prob':'mean', 'aabb_prob':'mean'})
temp.sort_values(['gpa'], ascending=False)


# As you can see from the table, which is sorted by GPA, a student can have a very different probability of making a certain grade when comparing classes with similar GPA's.

# That wraps up this notebook.  Primarily aimed at wrangling and understanding the dataset, a few tools were presented to gain some insight into grade patterns.
# 
# What I would like to do is look for hidden nuggets, such as if the time of day, facility, class length, and year (term) factor as much into grade distributions as the instructor does.  Any suggestions for how to do this are welcome.
