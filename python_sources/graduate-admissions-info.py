#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt
import pylab as P


# Any results you write to the current directory are saved as output.


# ### Dataset Description
# The dataset contains several parameters which are considered important during the application for Masters Programs. 
# The parameters included are : 
# 1. GRE Scores ( out of 340 ) 
# 2. TOEFL Scores ( out of 120 ) 
# 3. University Rating 
# ( out of 5 ) 
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
# 5. Undergraduate GPA ( out of 10 ) 
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 )
# 

# ### Load the data

# In[ ]:


admission_file = ("../input/Admission_Predict_Ver1.1.csv")
admission_data=pd.read_csv(admission_file)


# I'd like to see how many rows and columns I have, to get an idea about how much data there is.

# In[ ]:


admission_data.shape


# So from the results above, there are 500 rows, and 9 columns.
# 
# Next, I'll look at the datatypes and how many nulls there are in each of the 9 columns.

# In[ ]:


admission_data.info()


# There are two datatypes in this dataset: int64, whole numbers and float64, decimal numbers. 
# 
# Use head to see the first 5 lines of the data

# In[ ]:


admission_data.head()


# Now view the last five (using tail)

# In[ ]:


admission_data.tail()


# I prefer to remove the spaces in column names and make them all lowercase when doing analysis in pandas. It's easier to work with columns without spaces and not in title case.

# In[ ]:


admission_data.columns = ['serial', 'gre', 'toefl','rating', 'sop', 'lor', 'cgpa', 'research', 'chances']


# After looking at the data, Serial No does not look useful, it's just row numbers, so that column can be dropped. 

# In[ ]:


admission_data = admission_data.drop(['serial'], axis=1)


# Now look at the statisics for the dataset: count, mean, std, min value, 25th percentile, 50th percentile, 75th percentile, max value

# In[ ]:


admission_data.describe()


# The describe() method shows information about each column. One thing to note (as seen above in the info() method, there are no null values. The GRE values hare spread from a minimum score of 290 to a maximum score of 340. The TOEFL scores range from 92 to 120.  
# 
# Get the average GRE Score, SOP, University rating, GPA
# 

# In[ ]:


avg_GRE= admission_data['gre'].mean()
avg_SOP = admission_data['sop'].mean()
avg_Rating = admission_data['rating'].mean()
avg_cgpa = admission_data['cgpa'].mean()


# ### Get the correlation data
# 

# In[ ]:


admission_data.corr()


# In looking at the data, the higher correlations are between the GRE and TOEFL scores have a correlation of 0.82720. This makes sense, since the GRE has two parts, math and English, and TOEFL is for English-language proficiency. It makes sense that if one scored highly on GRE they would also score highly on the TOEFL, and vice versa. CGPA, TOEFL, GRE, and chance of admitance are all highly correlated. Again, this makes sense because admitance to university would be related to how high they scored on the exams and how will they did during their undergraduate eduation (CGPA).

# See list of unique values of GRE Score

# In[ ]:


admission_data['gre'].unique()


# Return a list of unique values

# In[ ]:


admission_data['toefl'].unique()


# ### Create a histogram of GRE scores (shortcut to features of matplotlib/pylab packages)

# In[ ]:


admission_data['gre'].hist()
P.show()


# ### Create a histogram of the TOEFL scores

# In[ ]:


admission_data['toefl'].hist()
P.show()


# ### Create a histogram of the Undergraduate GPA

# In[ ]:


admission_data['cgpa'].hist()
P.show()


# ### Create a histogram of the Statement of Purpose and Letter of Recommendation Strength 

# In[ ]:


admission_data['sop'].hist()
P.show()


# ### Create a histogram of the Research variable (yes - 1 or no - 0)

# In[ ]:


admission_data['research'].hist()
P.show()


# ### Create a histogram of the Research variable (yes - 1 or no - 0)

# In[ ]:


# draw a histogram (shortcut to features of matplotlib/pylab packages)
import pylab as P
admission_data['rating'].hist()
P.show()


# In[ ]:


#boxplot
admission_data.boxplot(column='gre')


# In[ ]:


#boxplot
admission_data.boxplot(column='toefl')


# In[ ]:


#boxplot
admission_data.boxplot(column='cgpa')


# In[ ]:


#boxplot
admission_data.boxplot(column='chances')


# In[ ]:


#values skew toward acceptance (average is over 70%)


# In[ ]:


# draw a histogram (shortcut to features of matplotlib/pylab packages)
import pylab as P
admission_data['chances'].hist()
P.show()
#note it skews much more towards being admitted


# In[ ]:


#boxplot one way to create it
bxplt= admission_data.boxplot(column='chances', by = 'cgpa')
xticks = [10,30,50, 70, 90, 110, 130, 150, 170, 190]
bxplt.xaxis.set_ticks(xticks)
bxplt.set_xticklabels(xticks, fontsize=16)


# In[ ]:



#boxplot using matlib
admission_data.boxplot(by=['chances'], column=['cgpa'])
# set your own proper title
plt.title('Boxplot of CGPA grouped by Admit Chance')
# get rid of the automatic 'Boxplot grouped by group_by_column_name' title
plt.suptitle("")
# Customize x tick lables
x = [1]
# create an index for each tick position

plt.xticks(range(0, 75, 10), fontsize=14)


# In[ ]:


#get first 5 rows
admission_data.iloc[0:5,:]


# In[ ]:


# scatter plot matrix
scatter_matrix(admission_data)
plt.show()

