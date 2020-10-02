#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.thestatesman.com/wp-content/uploads/2019/07/google.jpg" style="width:1000px;height:800px;">

# # Libraries and Information of Data

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Reading the data

# In[ ]:


# gjt = google job tittle

gjt=pd.read_csv("/kaggle/input/google-job-skills/job_skills.csv")


# In[ ]:


gjt.head(10)
#it gives first 10 rows in data


# In[ ]:


gjt.columns
# shows that we have which columns


# In[ ]:


# describing the data set
gjt.describe()


# In[ ]:


gjt.dtypes
#it gives int,float,object etc.


# #### Checking data

# In[ ]:


# cheking the null values in the dataset

gjt.isnull().any()


# In[ ]:


#I'll check if there is any NaN
gjt.isnull().sum()


# In[ ]:


#calculates the number of rows and columns
print(gjt.shape)


# # Visualization Part

# In[ ]:


gjt.Title.value_counts().head(20)


# In[ ]:


sns.catplot(y = "Category", kind = "count",
            palette = "colorblind", edgecolor = ".6",
            data = gjt)
plt.show()
#this graph gives categorical numbers that they position


# In[ ]:


gjt.Title.value_counts().head(20).plot.bar()
#it is giving about numbers of the tittle


# In[ ]:


sns.set(style="darkgrid")
sns.countplot(gjt['Company'])
plt.title('')

print(gjt['Company'].value_counts())


# In[ ]:


#it gives most 10 place in world. 
plt.title('Top 10 Location')
top_location=gjt['Location'].value_counts().sort_values(ascending=False).head(10)
top_location.plot(kind='bar')


# In[ ]:


# checking most popular top 20 types of job Titles 

plt.rcParams['figure.figsize'] = (19, 8)

color = plt.cm.PuRd(np.linspace(0, 1, 20))
gjt['Title'].value_counts().sort_values(ascending = False).head(20).plot.bar(color = color)
plt.title("Most Popular 20 Job Titles of Google", fontsize = 20)
plt.xlabel('Names of Job Titles', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()

