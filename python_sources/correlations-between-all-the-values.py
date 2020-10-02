#!/usr/bin/env python
# coding: utf-8

# **Importing all the necessary libraries.**

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt


# **Importing the data into a DataFrame.**

# In[ ]:


df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()


# **Correlation can only be calculated between numeric values. This is not possible between numeric and categorical data, hence a little transfromation should be of some help. The various transformations are as follows:**
# * Male and Female have been given the binary values i.e, 0 and 1 respectively.
# * The Various ethnic groups have been assigned numeric values between [1,5], starting with group A and going on till group E.
# * The various degrees have been numbered in such a manner so as to facilitate the higher levels of education with a higher numeric value.
# * Lunch: 1 has been assigned to "standard" and 2 to "free/reduced".
# * test perparation course: 0 has been assigned to "none" and 1 to "completed".

# In[ ]:


df.replace(to_replace='male', value=0, inplace=True)
df.replace(to_replace='female', value=1, inplace=True)
df.replace(to_replace=['group A', "group B", "group C", "group D", "group E"], value=[1,2,3,4,5], inplace=True)
df.replace(to_replace=["bachelor's degree", 'some college', "master's degree", 
                       "associate's degree", 'high school', 'some high school'],
                        value=[5,3,6,4,2,1], inplace=True)
df.replace(to_replace=['standard', 'free/reduced'], value=[1,2], inplace=True)
df.replace(to_replace=['none', 'completed'], value=[0,1], inplace=True)


# In[ ]:


df.head()


# **The plot below lets us visually represent how the parental level of education has an effect on a child's grades. The graph clearly shows that the children whose parents have a attained a higher degree perform better than others. **

# In[ ]:


plt.scatter(df['math score'],df['parental level of education'])
plt.scatter(df['writing score'],df['parental level of education'])
plt.scatter(df['reading score'],df['parental level of education'])
plt.legend(['Math Score', 'Writing Score', 'Reading Score'])
plt.vlines(x=40, ymin=0, ymax=7, linestyles='dashed')
plt.xlabel('Marks')
plt.ylabel('Parental Education')
plt.show()


# Considering that the pass marks is above 40(inclusive) then the number of students failing in the various subjects decreases with the increase in the parental level of education.

# **Lastly, after converting all the categorical data into numberic, we can calculate the correlation between the various entities.**

# In[ ]:


fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-0.5, vmax=1)
fig.colorbar(cax)
ticks = range(0,8)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()

