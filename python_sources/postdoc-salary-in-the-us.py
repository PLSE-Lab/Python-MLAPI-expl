#!/usr/bin/env python
# coding: utf-8

# ## Motivation
# Postdoctoral fellows make a significant workforce in the scientific community, contribute to or lead big scientific breakthroughs, yet it remains a low paying "job"  in the US (probably in Canada too).  The physical, phychological and  emotional turbulance postdocs go through is immense and intense. Before landing a permanent job it's also a significant time investment:  5-6 years as PhD students and 2-4 years or more as postdocs. I don't think they want to be reimbursed in terms of money (that's not what they are after when they chose to do a PhD in the first place), but at least it is good to know how much they make during their transition from a PhD student to a permanent job into academia or elsewhere. Many postdocs are in their mid to late 30s, and come to the US with a young family to raise, which can be a daunting task, especially if you live in high-rent cities and your spouse are not allowed to work legally.
# 
# **** I put job  in quotes because  I'm really not sure whether it's a job as we know it

# In[ ]:


# import packages
import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# import data
df=pd.read_csv("../input/h1b_kaggle.csv")


# In[ ]:


df.shape

# as is seen it is a dataset of over 3 million records of H1B visa application. Job titles include
# a large number of different positions they are hired for. Among them I sm only interested in
# postdoctoral fellows


# In[ ]:


# show header
df.head()


# In[ ]:


# show column names (see data description for details)
df.columns


# In[ ]:


# create postdoctoral dataset I am interested in.
# postdocs are often writtent in official job titles as "Postdoctoral", " Post doctoral" or
# simply "Postdoc" fellows. I used all 3 terms to filter rows and create the dataset containing
# information only about postdocs.

# filter job title containing "POSTDOCTORAL"
post1 = df[df['JOB_TITLE'].str.match('POSTDOCTORAL', na=False)]
# filter job title containing "POST DOCTORAL"
post2 = df[df['JOB_TITLE'].str.match('POST DOCTORAL', na=False)]
# filter job title containing "POSTDOC"
post3 = df[df['JOB_TITLE'].str.match('POSTDOC ', na=False)]
# join the three dataframes
postdoc = post1.append([post2, post3])


# In[ ]:


# show summary stats
print('mean postdoc salary is:', postdoc['PREVAILING_WAGE'].mean())
print('meadian postdoc salary is:', postdoc['PREVAILING_WAGE'].median())
print('minimum salary:', postdoc['PREVAILING_WAGE'].min())
print('maximum postdoc salary:', postdoc['PREVAILING_WAGE'].max())

# The mean ($ 117K) looks very high, so I looked at median, which is reasonable. So there must be some 
# very large numbers driving the mean to become so large?
# Looked at the max value, which is absurd for a postdoc salary!


# In[ ]:


# so to detect outliers/potential flaws in data by I decided to see salaries more than 140K.
(postdoc[postdoc['PREVAILING_WAGE']>140000]).shape

# there are 44 out of 37793 with a postdoc salary more than $140K. These are definitely problem rows, 
# visual inspection of rows also showed that all of those cases are either denied or withdrawm
# so decided to removed them


# In[ ]:


# drop 44 rows with salaries more than 140k
postdoc = postdoc[postdoc['PREVAILING_WAGE']<140000]


# In[ ]:


# now let's see some summary stats
postdoc['PREVAILING_WAGE'].describe()


# In[ ]:


# ploting salary distribution to see how it looks like in visuals
sns.distplot(postdoc['PREVAILING_WAGE'])


# In[ ]:


# Now I want to create another summary viaual for different years to see the trend. 
# But before that I want to inspect data types

postdoc.info()
# looks like we need to change 'YEAR' data type, which definitely isn't floating type


# In[ ]:


# change YEAR column from floating to integer
postdoc['YEAR'] = postdoc['YEAR'].astype(int)

# now see how many entries in each year
postdoc['YEAR'].value_counts()


# In[ ]:


sns.boxplot(x='YEAR' , y='PREVAILING_WAGE', data = postdoc)


# ## Conclusion
# Nothing surprising from the analysis, postdocs are paid around $40k per annum in the US. It is true this is bigger than what what they were getting for student stipend. However, if someone at their 30s make this much to maintain a family, then suddently this amount doesn't seem much. 
# Postdoctoral salary also seems to fair poorly compared with other H1B job titles. I haven't run those numbers but I found some interesting kernels doing exactly this https://www.kaggle.com/javidimail/h-1b-wage-distribution and this
# https://www.kaggle.com/arathee2/story-telling-with-data-science
