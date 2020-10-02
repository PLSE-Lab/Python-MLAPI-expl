#!/usr/bin/env python
# coding: utf-8

# # Student Performance

# Lets first start by having a look at our data and then performing EDA.

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objs as go 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[72]:


# Importing the data with pandas
df = pd.read_csv("../input/StudentsPerformance.csv")


# In[73]:


df.head()


# It seems that we have a lot to work with. Lets investigate further.

# In[74]:


df.info()


# In[75]:


df.describe()


# Our data seems to be in good shape. Are there any null values?

# In[76]:


df.isnull().values.any()


# By using the method info, we found out the size of our table and the data types in it.
# By using describe, we could find out details about our scores in the exams. After seeing some details, lets think of  some questions.

# Considering we have 3 columns with scores and 5 with details about the people taking the tests, we can formulate some question based on the difference in test scores.
# In which way the questions could be useful to a institution? 
# In case of significant differences in results based on biological indicators, further studies should be made.
# In case of significant differences in results based on parental level of education, further studies should be made regarding the relationship between the parent and child's education.
# In case of significant differences in results based on preparation or lunch, the quality of those services should be further studied as they could have an impact on the results.
# 
# 1. Based on gender, are there any significant differences(10 points) in test scores?
# 2. Based on race/ethnicity, are there any significant differences(10 points) in test scores?
# 3. Based on parental level of education, are there any significant differences(10 points) in test scores?
# 4. Based on test preparation course, are there any significant differences(10 points) in test scores?
# 5. Based on lunch, are there any significant differences(10 points) in test scores?

# ## Based on gender, are there any significant differences(10 points) in test scores?

# In[77]:


px.violin(df, y="math score", x="gender", color="gender", box=True)


# In[78]:


px.violin(df, y="reading score", x="gender", color="gender", box=True)


# In[79]:


px.violin(df, y="writing score", x="gender", color="gender", box=True)


# There is a clear difference in score. Female students tend to have higher grades in writing and reading, while men have a higher grade in math. It seems that also female student's score vary in a wider range. Lets check if it isn't because there are more females taking the exams.

# In[80]:


np.sum(df.gender =='female')


# There are more women, but not by a wide margin. As seen from the graphs, there are also more outliers that could distort the graph.

# Even if there is an advantage for male students in math, overall it seems that female students have the advantage. The average might be distorted by the outliers. Lets look at the medians in the graph.
# 
# Math score: Female: 65 Male: 69
# 
# Reading score: Female: 73 Male: 66
# 
# Writing score: Female: 74 Male: 64
# 
# It seems that the median scores of female students are better than men by a considerable margin. Lets create a new column for the average of the grade.

# In[81]:


df['avg_grade'] = 0
df['avg_grade'] = df.apply(lambda x : (df['math score'] + df['writing score'] + df['reading score'])/3, axis = 0)
df.head()


# In[82]:


px.violin(df, y="avg_grade", x="gender", color="gender", box=True, labels={'avg_grade': 'Average of the 3 scores', 'gender': 'Student gender'}, title='Average of the 3 scores for female and male students')


# As expected, the female median is higher than the median for males. If we checked the mean, it would have been distorted by the outliers, but basing our analysis on the median gives us a clearer idea. Is the difference significant enough? No. The female median is 70.3 and the male one is 66.3. The only place where there is a significant difference is in the writing scores. Further investigation should be taken into understanding what makes the median female score be 10 points higher than the male one.

# Why have we chosen the median to be our indicator? Because there are a few outliers that could have messed with our data as it often happens when people don't get the whole picture. Lets compare the averages.

# In[83]:


print('female', np.average(df.avg_grade[df.gender == 'female']), 'vs male', np.average(df.avg_grade[df.gender == 'male']))


# It seems that in this case, the average tells the same story, but in the cases where there are a lot of outliers, it could tell a different story. If there were a lot more female students with 100 or 0, the average would have been affected.

# ## Based on race/ethnicity, are there any significant differences(10 points) in test scores?

# In[84]:


fig = px.scatter(df, y="avg_grade", x="race/ethnicity", color="race/ethnicity", labels={'avg_grade': 'Average of the 3 scores', 'race/ethnicity': 'Race/Ethnicity of the student'})

fig.update(layout = dict(showlegend = False))


# In[85]:


fig = px.violin(df, y="race/ethnicity", x="avg_grade", color="race/ethnicity", orientation = 'h', 
                labels={'avg_grade': 'Average of the 3 scores', 'race/ethnicity': 'Race/Ethnicity of the student'})

fig.update(layout = dict(showlegend = False))


# In[86]:


df["race/ethnicity"].value_counts()


# The C and D group is overrepressented in the data. Group B and E together are a bit over group C.

# The median values should showcase the same story as the average would. Lets see which groups fares better.

# In[87]:


print('group A', np.average(df.avg_grade[df["race/ethnicity"] == 'group A']), '\nvs group B', 
      np.average(df.avg_grade[df["race/ethnicity"] == 'group B']), '\nvs group C',
     np.average(df.avg_grade[df["race/ethnicity"] == 'group C']), '\nvs group D',
     np.average(df.avg_grade[df["race/ethnicity"] == 'group D']), '\nvs group E',
     np.average(df.avg_grade[df["race/ethnicity"] == 'group E']))


# This is a happy situation in which the data doesn't have many outliers, but in general that is not the case. We can see that the values of the averages are close to the median, the one being the farthest away is for group C, having a difference of 1.2. There are a lot of distinct values on a bigger range than the others, so it is understandable why this is the case.

# There are significant differences, but we should consider them while keeping in mind the count.
# Weirdly enough, it seems the differences are in this format A<B<C<D<E.
# C is the most numerous group, while D comes right after it, and then B. The last ones are E and then A.
# The differences between groups B,C,D are not significant by the margin we decided.
# Group A and E have a difference of 10. They are also the smallest group and the most prone to be affected by values that are too far away from the mean. From the graph, we don't see any outliers in group A and we have only one in group E, but that is an outlier than has a very small value, not a very big one. There is an even bigger difference between the medians of the two groups(A: 61.3, E:73.5). This will need further investigation, as it indicates big differences in scores at the exams.

# ## Based on parental level of education, are there any significant differences(10 points) in test scores?

# In[88]:


fig = px.violin(df, y="parental level of education", x="avg_grade", color="parental level of education", 
          orientation = 'h', labels={'avg_grade': 'Average of the 3 scores', 'parental level of education': 'Parental level of education'})

fig.update(layout = dict(showlegend = False))


# In[89]:


df["parental level of education"].value_counts()


# The number of values for some college, associate's degree, high school, some high school are close in comparison with bachelor's degree and master's degree. We need to be careful when comparing them because there is a big difference in the quantity of data.
# The medians: 
# associate's degree 69.6
# bachelor's degree: 71.6
# high school: 65
# master's degree: 73.3
# some college: 68.6
# some high school: 66.6
# There are any significant differences here. There is a difference between high school and master's, but it doesn't qualify as a significant difference. Again, the quantity of data is different(between master's degree and high school), so it might tell the whole story.
# There aren't many interesting things here. 

# ## Based on test preparation course, are there any significant differences(10 points) in test scores?

# In[90]:


px.violin(df, y="avg_grade", x="test preparation course",box = True,  color="test preparation course", orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'test preparation course': 'Preparation Course'})


# In[91]:


df["test preparation course"].value_counts()


# Does it tell the whole story? No. It might be that conscientious students tend to take preparation course. This is might be why there are outliers that have a small score on the ones that don't do the preparation course. There are differences, but they could be influenced by many external factors that are not at disposal. It is not very important to do further studies on.

# ## Based on lunch, are there any significant differences(10 points) in test scores?

# In[92]:


px.violin(df, y="avg_grade", x="lunch", color = "lunch", box = True, orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'lunch': 'Lunch'})


# Wow. Surprinsingly enough, the difference is bigger even than that found in preparation course. If having the free lunch means a worse material background. Lets check the count.

# In[93]:


df["lunch"].value_counts()


# For the standard lunch, it seems that the majority sit around the median. The difference is very close to our margin for significant difference. This could be a sign that it needs further study, but lets first check how the violin plot looks when we also take into consideration the test preparation course, these two being the only that are decided by factors that are not biological.

# In[94]:


px.violin(df, y="avg_grade", x="lunch", color="test preparation course", box = True, orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'lunch': 'Lunch'})


# We can see that the violins for the standard/none is very similar to the free/completed, with the significant difference that most of the values in free/completed are in the third quartile, while those for the standard/none are around the median. IF these are indicators for being conscientious and having a good material background, it could indicate that a student should have at least one of those to fare better. By a small margin, it seems that being conscientious is more important, as there aren't any outliers.

# This finalizes the analysis we set on to do. The current kernel focused on asking questions and exploration.
