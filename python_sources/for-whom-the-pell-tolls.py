#!/usr/bin/env python
# coding: utf-8

# # Pell Grant Analysis
# 
# ### The Pell Grant History
# The [Pell Grant](https://en.wikipedia.org/wiki/Pell_Grant) is a form
# of student aid signed into law by President Jimmy Carter in 1978. It is an extension of the Higher Education
# Act (HEA) of 1965 (named after U.S. Senator Claiborne Pell of Rhode Island). 
# 
# ### The Intent
# The Pell grant is intended to make higher education more accessible, no matter the potential student's
# socioeconomic background. This in turn will give every HS grad the possibility to get a college degree and
# obtain a well-paying job. Thus, finally, we have a larger, stronger, and more educated American
# middle class (right?). Well, let's investigate
# 
# ### The Qualifications
# The Pell grant, as with most things over 30,
# has gone under many revisions. In its current form the recipient must achieve all of the below:
# 
# 1. Have a high school diploma or GED
# 2. Citizen or permanent-resident
# 3. [Expected Family Contribution] (https://en.wikipedia.org/wiki/Expected_Family_Contribution) is less than $5,730
# Achieving all three of these, the student it cut a $5,730 (maximum) check per academic year (max of 6 years) to use
# for educational purposes.
# 
# ### Thesis
# I do not intend to place blame on any party. I am going to stay impartial and report my findings as a scientist, without any political slant. What I think is happening is two-fold: 
# 
# 1. Despite the Pell grant, low-income students are still getting into debt by going to college.  
# 2. In addition to debt, they are not getting high-paying jobs in the long-run. 
# 
# ### Methods
# Everything is done in Python (with Pandas whenever possible). I learned some basic SQL for this project.
# 
# 1. Read in data from sSQLite database. (See: CollegeScorecardDataDictionary-09-12-2015.pdf for more info.)
# 2. Focus on school name, ZIP code, aid, debt, Pell stuff, and earnings.
# 3. Do simple if/then <,> arguements to gague how the per-school numbers are doing. 

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3 
con = sqlite3.connect('../input/database.sqlite')
raw = pd.read_sql('select UNITID, INSTNM, cast(LO_INC_DEBT_MDN as int) LO_INC_DEBT_MDN,cast(NOPELL_DEBT_MDN as int) NOPELL_DEBT_MDN, cast(PELL_DEBT_MDN as int) PELL_DEBT_MDN, PAR_ED_PCT_1STGEN, cast(DEBT_MDN as int) DEBT_MDN, cast(md_earn_wne_p6 as int) md_earn_wne_p6 from Scorecard', con)
print("Dropping all 'PrivacySuppressed' schools")
raw = raw.drop_duplicates('UNITID')
debt = raw[(raw.PELL_DEBT_MDN > 0) & (raw.DEBT_MDN > 0) & (raw.NOPELL_DEBT_MDN > 0)]
print(len(debt)/float(len(raw))*100, "% of raw data remaining")
print(len(debt), "schools remain")
print(sum(debt.PELL_DEBT_MDN > debt.DEBT_MDN)/float(len(debt)*100),
      "% of median Pell grant debt higher than median debt at any school")
print(sum(debt.NOPELL_DEBT_MDN > debt.DEBT_MDN)/float(len(debt)*100),      "% of median non-Pell grant debt higher than meidian debt at any school")
print(sum(debt.PELL_DEBT_MDN > debt.NOPELL_DEBT_MDN)/float(len(debt))*100,       "% of median Pell grant debt higher than median non-Pell debt at any school")
# next, onto long-term earnings.
print("\n Only keeping 'six yr earning' data schools")
earn = raw[(raw.md_earn_wne_p6 > 0) & (raw.DEBT_MDN > 0)]
print(len(earn)/float(len(raw))*100, "% of raw data remaning")
print(len(earn), "schools remain")
print(sum(earn.DEBT_MDN > earn.md_earn_wne_p6)/float(len(earn))*100,      "% of median debt higher than annual median earnings (6yrs out) by school.")
low_inc = earn[(earn.LO_INC_DEBT_MDN > 0)]
print("\n Only keeping 'median low income debt' data schools")
print(sum(low_inc.LO_INC_DEBT_MDN > low_inc.md_earn_wne_p6)/float(len(low_inc))*100,      "% of median low income debt higher than median annual earnings (6yrs out) by school.")
print(sum(low_inc.md_earn_wne_p6 > 30000)/float(len(low_inc))*100,      "% of low income students made more than $30k, 6 years out")
print(sum(low_inc.md_earn_wne_p6 > 40000)/float(len(low_inc))*100,      "% of low income students made >$40, 6 years out")
print(sum(low_inc.md_earn_wne_p6 > 50000)/float(len(low_inc))*100,      "% of low income students made >$50K, 6 years out")
print("\n Let's check low_inc data for the online for-profits:")
df = low_inc.sort_values(by='md_earn_wne_p6', ascending=False)
print(df[['INSTNM','md_earn_wne_p6']].head(50))


# # Results
# Online for-profits are polluting the stats, again. 
# # Conclusions
# A lot of Pell Grant students are getting mixed up in the massive online for-profit colleges. 
