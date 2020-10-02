#!/usr/bin/env python
# coding: utf-8

# # Is College Still Worth It?
# This explores the median earnings of college grads and who escaped poverty. 
# 
# Methods:  
# 
# 1. Sort all colleges by their median earnings, 6 years after graduation (md_earn_wne_p6)
# 2. Compare this with percent 1st gen students (PAR_ED_PCT_1STGEN)
# 3. Compare conclusions with BLS data (http://www.bls.gov/emp/ep_chart_001.htm).

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3 

import matplotlib.pyplot as plt
con = sqlite3.connect('../input/database.sqlite')

raw = pd.read_sql('select UNITID, INSTNM, STABBR, ZIP, cast(LO_INC_DEBT_MDN as int) LO_INC_DEBT_MDN, cast(PAR_ED_PCT_1STGEN as float) PAR_ED_PCT_1STGEN, cast(count_wne_p6 as int) count_wne_p6,cast(NOPELL_DEBT_MDN as int) NOPELL_DEBT_MDN, cast(PELL_DEBT_MDN as int) PELL_DEBT_MDN,cast(DEBT_MDN as int) DEBT_MDN, cast(md_earn_wne_p6 as int) md_earn_wne_p6 from Scorecard', con)
# dropping all duplicates
raw = raw.drop_duplicates('UNITID')
# dropping all "PrivacySuppressed" schools
df = raw[(raw.PAR_ED_PCT_1STGEN > 0) & (raw.md_earn_wne_p6 > 0) & (raw.count_wne_p6 > 0)]
# setting up the graph
x=df.PAR_ED_PCT_1STGEN*100
y=df.md_earn_wne_p6
# converting states (STABBR) to ints
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
lbl.fit(list(df.STABBR.values))
colors = lbl.transform(df.STABBR.values)
area = df.count_wne_p6
plt.xlabel('Perect of 1st gen students')
plt.ylabel('Salary (6yrs after)')
plt.title('College Scorecard')
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


# # Large Dot Analysis
# If one looks at the biggest dot, max(count_wne_p6) = 12567, the reader will find that University of Phoenix has entered 6 lines of the exact same data! Same six year earnings: $45,000. Same median debt: $9,500. What a bargain! Changing only the location into UNITID, INSTNM, STABBR, ZIP. Ok, good catch, but it appears as though this is going to keep happening with various other online for-profit shcools. 
# 
# ### Methods
# 1. Sort by count_wne_p6  descending (shout-out to MS Excel for teaching me this trick).
# 2. Find a way to filter out all the for-profit schools 
# 
# ### Conclusion
# It seems as though only the online for-profits are keeping long-term stats on their graduates.
# 
# ### Whats Next
# I might be missing something. I would hope that public non-profits are also keeping stats. 

# In[ ]:


df = df.sort_values(by='count_wne_p6', ascending=False)
print(df[['INSTNM','count_wne_p6']].head(50))


# In[ ]:


df = raw[(raw.PAR_ED_PCT_1STGEN > 0) & (raw.md_earn_wne_p6 > 0) & (raw.count_wne_p6 > 0) & (raw.count_wne_p6 < 10000)]
print(len(df), "schools with 6yr stats and less than 10,000 students \n")
x=df.PAR_ED_PCT_1STGEN*100
y=df.md_earn_wne_p6
lbl = preprocessing.LabelEncoder()
lbl.fit(list(df.STABBR.values))
colors = lbl.transform(df.STABBR.values)
area = df.count_wne_p6
plt.xlabel('Percent of 1st gen students (by school)')
plt.ylabel('Salary (10yrs after)')
plt.title('College Scorecard')
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

