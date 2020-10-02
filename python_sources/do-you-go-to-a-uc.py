#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Like many in the state of California, I attend college at a UC (University of California) school. Thus I was interested in parsing some basic statistics about the schools in this system. It's common knowledge among California residents which UC's are considered "better" or "worse," and I wanted to see if the statistics reflected that. Below is the median graduation debt for all the UC's in a four year interval. 

# In[ ]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""
SELECT INSTNM College 
FROM Scorecard 
WHERE Year=2000 
AND INSTNM like 'University of California%' 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
for i in range(2010,2014):
    column = pd.read_sql_query("""
    SELECT GRAD_DEBT_MDN FROM Scorecard 
    WHERE Year="""+str(i)+""" 
    AND INSTNM like 'University of California%' 
    AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
    df[str(i)]=column
conn.close()


df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Graduation Debt')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()


# It's interesting to see that the median debt for UC Berkeley has stayed even throughout these years, and has even dropped in 2013. Next, here is a plot of in-state tuition over the same 4 year interval.

# In[ ]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""
SELECT INSTNM College 
FROM Scorecard 
WHERE Year=2000 
AND INSTNM like 'University of California%' 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
for i in range(2010,2014):
    column = pd.read_sql_query("""
    SELECT TUITIONFEE_IN FROM Scorecard 
    WHERE Year="""+str(i)+""" 
    AND INSTNM like 'University of California%' 
    AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
    df[str(i)]=column
conn.close()


df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Tuition Fee')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()


# Curiously enough, all UC tuition rates shot up after 2010. However, it seems that UC Davis has the highest tuition rate of all the UC schools. Finally, here is a plot of median earnings 10 years after graduation of the 2001 cohort (?). 

# In[ ]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
df = pd.read_sql_query("""SELECT INSTNM College, 
md_earn_wne_p10
FROM Scorecard 
WHERE INSTNM like 'University of California%' 
AND Year=2011 
AND PREDDEG = 'Predominantly bachelor''s-degree granting'""", conn)
conn.close()

df.plot(kind='bar')
leg = plt.legend( loc = 'lower right')
ax = plt.subplot()
ax.set_ylabel('Income 10 Years after Graduation')
ax.set_xlabel('School')
ax.set_xticklabels(['UCB', 'UCD', 'UCI', 'UCLA', 'UCR', 'UCSD', 'UCSF', 'UCSB', 'UCSC'])
plt.show()


# It looks like students from Berkeley, LA, and San Diego earn the most after graduation. These are known to be top tier UC schools. However, is the difference really all that significant? Furthermore this information is aggregate and we have no way of knowing earnings by major within these groups. I don't know exactly how much we can get out of this data, but it's nice to know a little more about the schools that we attend.
