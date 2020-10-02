#!/usr/bin/env python
# coding: utf-8

# # FCE Data Analysis (DSC)
# 
# ### Author: Yeung
# 
# ### Goals: Find out the most high-rated professors, courses, and what features in FCE are most correlated to Overall course rate. 
# 
# ### Universe of data: I subest the data in terms of my major: STAT and ML, in recent 5 years

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Data Pre Process and EDA
# 
# To ensure the validity of results, I subset observations having at least $\frac{1}{3}$ response rate and at least $25$ possible responents. It allows me to filter out summer courses and possible biased responses.
# Let's figure out what our data looks like!

# In[ ]:


df = pd.read_csv("../input/fce_data.csv")
df.head()
df = df.drop(columns=['Semester', 'Hrs Per Week 5', 'Hrs Per Week 8'])
df = df[(df["Possible Respondents"] >= 25) & (df["Level"] == "Undergraduate") & (df["Year"] >= 2014) & (df["Response Rate %"] > 33)]
statml = df[(df["Dept"] == "STA") | (df["Dept"] == ("MLG"))]
statml = statml[['Year', 'Name','Hrs Per Week', 'Course ID','Interest in student learning', 'Clearly explain course requirements', 'Clear learning objectives & goals',
                'Instructor provides feedback to students to improve', 'Demonstrate importance of subject matter', 'Explains subject matter of course',
                "Overall teaching rate", 'Overall course rate']]
statml = statml.fillna(statml.mean())
statml.describe()


# In[ ]:


statml_avg_by_prof = statml.groupby(["Name"], as_index=False).mean()


# ## Find the best STAT Professors. Congratulations Joel (my 401 professor)! You're in top 5 :)

# In[ ]:


statml_avg_by_prof.nlargest(10, "Overall teaching rate")


# ## Find the most time-consuming STATML course! 

# In[ ]:


# Find the most time-consuming STATML course! 
statml_by_course = statml.groupby(["Course ID"], as_index=False).mean()
statml_by_course.nlargest(10, "Hrs Per Week")


# ## Find the least time-consuming STATML course! 

# In[ ]:


# Find the least time-consuming STATML course! 
statml_by_course.nsmallest(10, "Hrs Per Week")


# Other than introductory level courses in STAT, it is interesting to see that 10315 (ML for SCS), 10401 (ML) and 36462 (Data Mining) which are the advanced courses are also in the list of least time-consuming courses.

# ## Discover features related to Overall Course Rate by a correlation matrix

# In[ ]:


ax1 = statml.plot.scatter(x='Explains subject matter of course', y='Overall course rate')
ax2 = statml.plot.scatter(x="Clear learning objectives & goals", y='Overall course rate')
statml_subset = subset[['Explains subject matter of course', 'Overall course rate']]
statml_subset.corr(method="pearson")
statml_corr = subset.corr(method="pearson").style.background_gradient(cmap='coolwarm')
statml_corr


# ## Some interesting findings from the correlation matrix
# 
# * *Clear learning objectives & goals* is the most correlated field to Overall Course Rate. This can be a suggestion for faculty to improve FCE rating! 
# 
# * *Hours Per Week* is basically uncorrelated with all fields expect the ***Year***. This means we are experiencing an increasing hours per week in Stat&ML courses!

# ## What I will take advantage of using the results I found
# 
# * Data Mining (36462) is also less time-consuming, might want to take this as my advanced stat elective next semester
# 
# * **GORMLEY MATTHEW** is one of the most highly rated professors in ML department. I will decide to take his ML course next semester
