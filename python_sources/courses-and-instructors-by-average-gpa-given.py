#!/usr/bin/env python
# coding: utf-8

# ## Connect to SQL database

# In[2]:


import pandas as pd # reading files
import sqlite3 # sqlite database
from shutil import copyfile # copy database func

# move the database into a writeable format
copyfile('../input/database.sqlite3', './database.sqlite3')

# open connection
conn = sqlite3.connect('./database.sqlite3')
conn


# # Calculating GPA
# The dataset we are working with has a table called "grade_distributions". Each row describes how a particular section of a course from a certain semester was graded - how many people got an A, AB, B, BC, C, D, and F, and other miscellaneous grades.
# 
# For a particular section, we can calculate what the average GPA was for students in the section by following the weights described by the [UW Madison registrar office](https://registrar.wisc.edu/grades-and-gpa/).
# 
# GPA = (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (total # of a, ab, b, bc, c, d, and f)
# 
# This isn't going to be a pretty SQL query...

# In[47]:


pd.read_sql("""
  SELECT
    course_offering_uuid,
    section_number,
    (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS gpa
  FROM grade_distributions
  LIMIT 5
""", conn)


# # Create a view
# What we just saw was a long, complicated query that we want to use often, so let's make it into a view.
# 
# However, we will want to include in this view the number of grades ("num_grades") that went into calculating the GPA so that we can weight it properly when averaging multiple sections. You'll see that column of the view used later.

# In[37]:


# we add a new view
c = conn.cursor()
c.execute("DROP VIEW IF EXISTS section_gpas")
c.execute("""
  CREATE VIEW
  section_gpas (course_offering_uuid, section_number, gpa, num_grades)
  AS
  SELECT
    course_offering_uuid,
    section_number,
    (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS gpa,
    a_count + ab_count + b_count + bc_count + c_count + d_count + f_count AS num_grades
  FROM grade_distributions
""")


# # Query the view
# Now that we have a view created, we can query that view like a table. We get the same result that we did earlier, but the view hides the complexity of the query!

# In[38]:


# and now we can query it...
pd.read_sql("SELECT * FROM section_gpas LIMIT 5", conn)


# # What can we do with this data?
# Well there is much more data in this dataset than just grade distributions for sections. There are courses, instructors, schedules, subjects, and rooms all associated in various ways. Here's how we can see all the tables...

# In[55]:


pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)


# # Let's find instructors who give the highest GPA's on average (top 15)
# 
# ### How are grades, instructors, teachings, sections, course offerings, courses, and subjects related?
# We have a lot of tables that are all related in some way.
# 
# 1. A subject has many courses.
# 2. A course can be offered in a semester through an entry in the "course_offerings" table.
# 3. A course offering may have 1 or more sections (i.e. lectures, discussions, labs).
# 4. Sections may be taught by any number of instructors through the "teachings" table.
# 5. A section may or may not have a grade distribution/gpa associated with it.
# 
# Let's first join all the necessary tables together. In this case, only instructors, their teachings, sections, and section gpas are relevant.

# In[8]:


pd.read_sql("""
  SELECT *
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  LIMIT 5
""", conn)


# Now we need to do the following modifications to the query:
# 
# 1. Group by instructor id
# 2. Select only the relevant columns (the instructor id, their name)
# 3. Also grab the average of each group's GPA, taking into account the number of grades per section (the weight it gets in the avg GPA calculation)
# 4. Order by GPA descending (highest first)
# 
# This can take a while.

# In[49]:


pd.read_sql("""
  SELECT 
    i.id, 
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  ORDER BY avg_gpa DESC
  LIMIT 10
""", conn)


# We have a problem. We have so many people who give 4.0 grades that it doesn't really seem meaningful... What's going on?
# 
# Small class sizes at Madison can often mean high level graduate classes where the professor works directly with the students, and high letter grades are more common...
# 
# As you can see below, the average number of students per section for these instructors is fairly low, and most listed have  graded fewer than 50 students total in their career at Madison. Take a look at the last two columns to get some indictation of this.

# In[51]:


pd.read_sql("""
  SELECT 
    i.id, 
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades), 
    SUM(gpas.num_grades)
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  ORDER BY avg_gpa DESC
  LIMIT 10
""", conn)


# Let's make the query hide instructors who have graded fewer than 250 students and who have an average section size of fewer than 50.
# 
# Now we have some results under 4.0. This is not a perfect way of filtering less meaningful data, but it does the job for now.

# In[52]:


pd.read_sql("""
  SELECT 
    i.id, 
    i.name, 
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  HAVING avg_num_grades >= 30 AND total_num_grades >= 250
  ORDER BY avg_gpa DESC
  LIMIT 15
""", conn)


# # Instructors who give the lowest GPA's on average (top 15)
# :(
# 
# All we have to do is sort by avg_gpa ASC this time.

# In[53]:


pd.read_sql("""
  SELECT 
    i.id,
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  HAVING avg_num_grades >= 30 AND total_num_grades >= 250
  ORDER BY avg_gpa ASC
  LIMIT 15
""", conn)


# # Courses that give the lowest GPA's on average (top 15)
# Now we need courses, course offerings, sections, and section GPA's.
# 
# Note that we also join subjects and subject memberships so we can add the course subject to the results and provide some more context.

# In[54]:


pd.read_sql("""
  SELECT 
    c.uuid, 
    c.name,
    GROUP_CONCAT(DISTINCT subjects.abbreviation) as subjects,
    c.number,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM courses c
  JOIN course_offerings co ON co.course_uuid = c.uuid
  JOIN subject_memberships sm ON sm.course_offering_uuid = co.uuid
  JOIN subjects ON sm.subject_code = subjects.code
  JOIN sections s ON s.course_offering_uuid = co.uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY c.uuid
  HAVING total_num_grades >= 250
  ORDER BY avg_gpa ASC
  LIMIT 15
""", conn)


# # Subjects/departments that give the lowest GPA on average (top 30)

# In[46]:


pd.read_sql("""
  SELECT
    subjects.name,
    subjects.abbreviation,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM subjects
  JOIN subject_memberships sm ON sm.subject_code = subjects.code
  JOIN course_offerings co ON co.uuid = sm.course_offering_uuid
  JOIN sections s ON s.course_offering_uuid = co.uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY subjects.code
  HAVING total_num_grades > 0
  ORDER BY avg_gpa ASC
  LIMIT 30
""", conn)

