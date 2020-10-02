#!/usr/bin/env python
# coding: utf-8

# # Analysis of College Scorecard data Instate Vs Outstate fees
# Author: Tejaswi
# 
# Date created: 2/25/2016

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score


# Reading the documentation. The data provided includes:
# 
# 1. **Root** - Institution Id (OPEID), Currently operating  
# 2. **About school** - Name (INSTNM), Location (CITY, STABBR, ZIP, LATITUDE, LONGITUDE), Degree Type, Public/Private Nonprofit/ Private forprofit (CONTROL), Religious affiliation, Revenue/Cost of school (Revenue per full-time student (TUITFTE), expenditures per Full time student (INEXPFTE), avg faculty salary (AVGFACSAL))
# 3. **Admissions** - Admission rate (ADM_RATE, ADM_RATE_ALL), SAT and ACT scores (SATVR\* for \_25 and \_75)
# 4. **Costs** - Average cost of attendance, Tutions and fees (TUITIONFEE_IN, TUITIONFEE_OUT)
# 5. **Student** - Number of undergraduate students (UGDS), Undergradute students by part-time, full-time, UG students by family income (INC_PCT_LO, ..), Retention rate , UG student demographics for earning cohorts (female, married, dependent, ..)
# 6. **Financial aid** - percent of students recieving federal loans (PCTFLOAN), cumulative median debt (DEBT_MDN)
# 7. **Completion** - completion rate for first-time full time students 
# 8. **Earnings** - Mean and median earnings institutional aggregate of all federally aided students (mn_earn_wne_p\*, md_earn_wne_p\* ), Threshold earnings 
# 9. **Repayment** - Cohort default rate, repayment rate 

# In[ ]:


sqlite_file = '../input/database.sqlite'
con = sqlite3.connect(sqlite_file)
df = pd.read_sql_query("SELECT INSTNM, ZIP, LATITUDE, LONGITUDE, TUITFTE, INEXPFTE, AVGFACSAL, TUITIONFEE_IN, TUITIONFEE_OUT, CONTROL,                        mn_earn_wne_p10, md_earn_wne_p10, sd_earn_wne_p10, pct25_earn_wne_p10, pct75_earn_wne_p10,                        ADM_RATE, ADM_RATE_ALL, SATVRMID, SATMTMID, SATWRMID                        PCIP14, PCIP15                        FROM Scorecard" , con)
con.close()


# # Fees In-State Out-State

# In[ ]:


plt.figure(figsize=(10,10))
sns.regplot(df.TUITIONFEE_IN, df.TUITIONFEE_OUT, scatter = True, fit_reg = False)


# plotting TUITION for in state and out state Shows an interesting plot. There seems to be two sets of data points , one where data points lie perfectly of the line of x = y which implies both tuitions are the same for these institutions. Lets check what type of insitutions these are. 
# 

# In[ ]:


print(df.CONTROL[df.TUITIONFEE_IN == df.TUITIONFEE_OUT].value_counts()/df.CONTROL.value_counts())
sns.lmplot(x="TUITIONFEE_IN", y="TUITIONFEE_OUT", col="CONTROL", data=df, size=4)
plt.show()


# This is interesting and follows intuition. 
# 
# * About 25% of Private for-profit and 56% of private non-profit have same fees for in and out of state while only 5% of public schools have the same fees. 
# * The plots show this visually to see how many points lie of the line. 
# * However, one can see how there a significant number of points just slightly off the line which would have been excluded in the comparison of in-state and out-state fees as the match needed to be perfect to the dollar. 
#  
# Instead we lets see how close the in-state and out-state fees are for this we will compute a new metric:
# 
# * In-state to out-state fees ratio i.e. TUITIONFEE_IN/TUITIONFEE_OUT
# * One needs to be careful about missing data. If TUITIONFEE_IN or TUITIONFEE_OUT are missing then those are not valid data points 
# * Missing info about type of institution (CONTROL) is also a problem 
# 
# *One can and should put more checks on the data i.e. checking if there are unrealistic values for any of the colums, but i will skip that for now*
# drop missing data rows

# In[ ]:


df_TUITcomplete = df[['TUITIONFEE_IN','TUITIONFEE_OUT', 'CONTROL']].dropna(how='any')
# aggegating results 
df_TUITcomplete.groupby('CONTROL').mean()


# Now the difference is more apparent. These almost no in-state out-state fee difference for private instituions while for public institutions the out state fee is typically about twice in-state. Also notice though that private colleges are far more expensive, being roughly twice as expensive as avg public out-state and five times more expensive the average in-state public institutions
