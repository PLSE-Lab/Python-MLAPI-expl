#!/usr/bin/env python
# coding: utf-8

# The data were obtained in a survey of students math and portuguese language courses in secondary school. It contains a lot of interesting social, gender and study information about students. We will use it to do some exploratory data analysis (EDA) to predict students final grade. 
# 
# What we want to know:
# - Correlation between features
# - Weekly consumption of alcohol by the students
# - Final exam scores based on student's alcohol consumption
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/student-alcohol-consumption/student-mat.csv')
data.head(10)


# In[ ]:


data.info


# In[ ]:


data.columns


# Attributes that we're interested in:
# 
# - Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# - Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# - G3 - final grade (numeric: from 0 to 20, output target)

# ### Correlation between features

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot = True, fmt= ".2f", cbar = True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)


# It seems that only exam scores that are highly correlated with each other. Now let's combine weekdays alcohol consumption with weekend alcohol consumption. 

# In[ ]:


data['Dalc'] = data['Dalc'] + data['Walc']


# In[ ]:


# Let's check if students drink alcohol
list = []
for i in range(11):
    list.append(len(data[data.Dalc == i]))
    
ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)
plt.ylabel('Number of Students')
plt.xlabel('Weekly alcohol consumption')


# In[ ]:


# Visualize final exam scores based on student's alcohol consumption
labels = ['2', '3','4', '5', '6', '7', '8', '9', '10']
colors = ['red', 'yellow', 'green', 'blue', 'grey', 'purple', 'cyan', 'brown', 'pink']
explode = [0,0,0,0,0,0,0,0,0]
sizes = []

for i in range(2,11):
    sizes.append(sum(data[data.Dalc == i].G3))
    
total_grade= sum(sizes)
average = total_grade/float(len(data))

plt.pie(sizes, explode=explode, colors=colors, labels=labels, autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Total grade : '+str(total_grade))
plt.xlabel('Students Grade Based on Weekly Alcohol Consumption')


# It seems that those who consumed alcohol twice a week perform better. Let's take a look with swarm plot to understand whether alcohol really does impact the grades. 

# In[ ]:


ave = sum(data.G3)/float(len(data))
data['ave_line'] = ave
data['average'] = ['above average' if i > ave else 'under average' for i in data.G3]

sns.swarmplot(x='Dalc', y= 'G3', hue = 'average', data = data, palette={'above average':'blue', 'under average':'red'})
plt.savefig('graph.png')


# We noticed that students with the highest grade consumes alcohol twice a week. 

# In[ ]:


# Final exam average grades
sum(data[data.Dalc == 2].G3)/float(len(data[data.Dalc == 2]))


# In[ ]:


# Average grade
list = []
for i in range(2,11):
    list.append(sum(data[data.Dalc == i].G3)/float(len(data[data.Dalc == i])))
ax = sns.barplot(x = [2,3,4,5,6,7,8,9,10], y = list)
plt.ylabel('Average Grades of students')
plt.xlabel('Weekly alcohol consumption')


# In[ ]:




