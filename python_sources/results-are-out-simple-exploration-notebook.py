#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from IPython.display import HTML
sns.set(rc = {'figure.figsize':(15,8)})
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 50000
pd.options.display.max_columns = 1000

sns.set(rc = {'figure.figsize':(15,8)})

def printmd(string):
    display(Markdown(string))
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Toggle Hide/Show code"></form>''')


# In[ ]:


data = pd.read_csv('../input/StudentsPerformance.csv')


# # Data Summaries
# 
# ## Data Description
# *  Gender : Gender of the student
# * Ethnicity : Ethnicity to which the student belongs
# * Parent_lvl_of_edu : Education level of the parents/gaurdian of the student
# * Lunch : Standard of the lunch provided to the student in school
# * Test_prep_course : Whether the student took the preparation course 
# * Math_score : Mathematics score of the student
# * Reading_score : Reading score of the student
# * Writing_score : Writing score of the student
# 
# **New columns**
# * id : Identifier for each student

# In[ ]:


data.loc[:,'id'] = range(len(data))


# In[ ]:


data.columns = ['gender', 'ethnicity', 'parent_lvl_of_edu', 'lunch',
       'test_prep_course', 'math_score', 'reading_score',
       'writing_score','id']
data = data[['id','gender', 'ethnicity', 'parent_lvl_of_edu', 'lunch',
       'test_prep_course', 'math_score', 'reading_score',
       'writing_score']]


# ### Dimensions of the data 
# 

# In[ ]:


print('Rows : ',data.shape[0],'\nColumns : ',data.shape[1],'\n')


# ### Data types of each column

# In[ ]:


print('Data Types of Each Column\n',data.dtypes)


# ### Glimpse of the data file

# In[ ]:


data.head()


# ### Null values in each column 

# In[ ]:


data.isnull().sum()


# # Data Exploration

# ## Distribution of different variables
# 
# I have plotted bar charts and violin plots to take a look at the distribution of all the variables provided in the data. This gives us a clear picture of the distribution of the data instead of looking at some numbers.

# In[ ]:


sns.set(rc = {'figure.figsize':(15,10)})
plt.subplots_adjust(hspace = 0.4)
plt.subplot(221)
sns.countplot(data.gender)
plt.title('Gender Distribution')
plt.subplot(222)
sns.countplot(data.ethnicity)
plt.title('Race/Ethnicity Distribution')
plt.subplot(223)
sns.countplot(data.parent_lvl_of_edu)
plt.title('Education Level of Parent Distribution')
plt.xticks(rotation = 45)
plt.subplot(224)
sns.countplot(data.lunch)
plt.title('Lunch Distribution')
plt.show()


# * Gender distribution is quite balanced in this data. The number of male students is approximately similar to the number of female students.
# * **Group C** is the majority ethnicity of the data whereas students of  **Group A** ethnicity are the minority.
# * Majority parents education level is **Some college** and **Associate's degree**. **Master degree** holders parents are quite in number.
# * Most students gets **standard** lunch.
# 
# 
# Parents education level plays a vital role in student scores. I hope to see higher scores for the students whose parents hold master's degree. Lunch variable may affect the score, reduced/free lunch may lead to reduced nutrition to children and in turn affecting their performance in tests. 

# In[ ]:


fig,ax = plt.subplots(figsize = [6,6])
sns.countplot(data.test_prep_course,axes = ax)
plt.title('Test preperation course Distribution')
plt.show()


# * Majority of students have not taken the Test prep course.
# 
# I would like the see the average scores across this variable to identify whether this additional preparation course is worth spending money and time.

# In[ ]:


sns.set(rc = {'figure.figsize':(15,9)})
grid = plt.GridSpec(1, 5, wspace=0.2, hspace=1)
plt.subplot(grid[0, :2])
ax = sns.violinplot(data.math_score, orient = 'v', color='red')
ax = sns.violinplot(data.reading_score,orient = 'v',color = 'blue')
ax = sns.violinplot(data.writing_score, orient = 'v', color="orange")
plt.setp(ax.collections, alpha=.5)
plt.ylabel('Score')
plt.title('Overlapped distritbution of different Scores')
plt.subplot(grid[0, 2])
ax = sns.violinplot(data.math_score, orient = 'v', color='red')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Math Score')
plt.subplot(grid[0, 3])
ax = sns.violinplot(data.reading_score, orient = 'v', color='blue')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Reading Score')
plt.subplot(grid[0, 4])
ax = sns.violinplot(data.writing_score,orient = 'v',color = 'orange')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Writing Score')
plt.show()


# Using this violin plot I have tried to show the distribution of all test scores and their overlapping areas. We observe that Reading and Writing score have similar maximas and distribution as well. Mathmatics score have a slight different distribution than the other two scores.
# 
# Similar distribution of Reading and Writing score indicates huge correlation between the two.

# ## Distribution of Scores across different Variables
# 
# Let's analyse how distribution of scores varies with different variables using different visuals.

# In[ ]:


sns.violinplot(x = 'variable', y = 'value', hue = 'lunch',
              data = pd.melt(data, 
               id_vars = ['id_var','lunch'], 
               value_vars=['reading_score','math_score','writing_score']),
              split = True,
              palette = 'Set1')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Types of Lunch')
plt.show()


# We observe very different distributions of test scores across different type of lunch. 
# * Free/reduced lunch consists of lowest scoring students
# * Standard lunch population is mostly distributed above 50 in terms of test score.
# * Standard lunch have high average score than free/reduced lunch

# In[ ]:


sns.violinplot(x = 'variable', y = 'value',hue='gender',
               data = pd.melt(data, 
               id_vars = ['id_var','gender'], 
               value_vars=['reading_score','math_score','writing_score'])
               ,split = True,
              palette = 'Set2')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Types of Gender')
plt.show()


# We see similar distributions for Reading and Writing score across different genders. 
# * Male gender have better average score for **Mathematics** as compared to female gender.
# * Female gender have slightly better average score for **Reading and Writing** score.

# In[ ]:


sns.boxplot(x = 'variable', y = 'value',hue='test_prep_course',
               data = pd.melt(data, 
               id_vars = ['id_var','test_prep_course'], 
               value_vars=['reading_score','math_score','writing_score'])
               ,
              palette = 'Set3')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Test preparation course flag')
plt.show()


# We observe that Preparation Course does give a slight boost to scores in every section(reading, writing and mathematics). We see a significant increase in **Writing Score** if a student completed the Prep. course.

# In[ ]:


grid = plt.GridSpec(1, 7, wspace=0.6, hspace=1)
plt.subplot(grid[0, :4])
sns.boxplot(x = 'variable', y = 'value',hue='ethnicity',
               data = pd.melt(data, 
               id_vars = ['id_var','ethnicity'], 
               value_vars=['reading_score','math_score','writing_score']),
              palette = 'Set1')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Ethnicities')

plt.subplot(grid[0,4:])
sns.countplot(data.ethnicity,palette = 'Set1')
plt.title('Race/Ethnicity Distribution')
plt.xlabel('')

plt.show()


# There is no relation between population of an ethnicity and their respective test scores. But in case of **Group A** ethnicity, we see that the population of students is the lowest as well as their average scores are lowest among other ethnicities. 

# In[ ]:


grid = plt.GridSpec(1, 7, wspace=0.6, hspace=1)
plt.subplot(grid[0, :4])
sns.boxplot(x = 'variable', y = 'value',hue='parent_lvl_of_edu',
               data = pd.melt(data, 
               id_vars = ['id_var','parent_lvl_of_edu'], 
               value_vars=['reading_score','math_score','writing_score']),
              palette = 'Set3')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Levels of Parent Education')

plt.subplot(grid[0,4:])
sns.countplot(data.parent_lvl_of_edu,palette = 'Set3')
plt.title('Parent Education Level Distribution')
plt.xticks(rotation = 45)
plt.xlabel('')
plt.show()


# As expected, students whose parents hold a master's degree have highest scores among all other students. 
# * Average scores in each test follows the order of parent education level
# * Students whose parents have only completed high school performs worst in their tests.

# ## Correlation between Tests
# 
# If Test scores are **highly correlated**, we can evaluate students on the basis of average of all three tests.

# In[ ]:


fig,ax = plt.subplots(figsize = [9,8])
sns.heatmap(data[['reading_score','math_score','writing_score']].corr(),
           cmap="YlGnBu",annot= True,axes = ax)
plt.show()


# We see a huge correlation between writing and reading score. That means predicting reading or writing by using either one of them is quite easy using simple algorithms like linear regression. 

# I hope the reader of this notebook got to understand the underlying trends in data through this short exploration. Let me know if I missed something.
# 
# Thanks.
# 
# Peace.

# In[ ]:





# In[ ]:




