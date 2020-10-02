#!/usr/bin/env python
# coding: utf-8

# # Report on Student Performance

# #### The main aim of this report is to analyze the various factors which affect the overall score of the students and to find out what are the factors which could be tweaked to see a considerable change in the performance of the students

# ___

# First we will start with importing all the required libraries. And in the second step fetched the dataset into a variable "df".

# In[ ]:


import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(15,30)})
sns.set(font_scale=3)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


df.head()


# ___

# ### In the next line of code I added another columns, average_score. Which goes by the name. It contain the average score of every candidate.

# In[ ]:


df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3


# In[ ]:


df.head()


# ___

# ## Analytical Section

# i) This section contains data grouped by gender and race/ethnicity.

# In[ ]:


gender_race_group = df.groupby(['gender', 'race/ethnicity']).mean().reset_index()


# In[ ]:


gender_race_group.head()


# In[ ]:


sns.catplot(x = 'gender', y = 'average_score', data = gender_race_group,height=10, aspect = 2.1, kind = "bar", hue = "race/ethnicity", palette="Blues_d")


# The plot above shows the gender vs average score and how it is affected by the ethnic group or race. And we can clearly deduce from the plot that, both the genders, male and female shows the variation in their average score and this variation is affected by their race/ethnicity. But on to a degree of 10 points.

# ___

# In[ ]:


sns.catplot(x = 'race/ethnicity', y = 'average_score', hue = 'gender',height=10, aspect = 2.1, data = gender_race_group, kind='bar', palette="Blues_d")


# When we compare the performance of males and females with in a particular race, we get the graph shown above. All the groups shows a difference of 7 to 5 points, but group E performed better and the difference between the scores of genders in this group is also less when compared with other groups.

# ___

# ii) This section shows how 'lunch' influences the performance of each gender.

# In[ ]:


gender_lunch_group = df.groupby(['gender', 'lunch']).mean().reset_index()


# In[ ]:


gender_lunch_group


# In[ ]:


sns.catplot(x = 'lunch', y = 'average_score',height=10, aspect = 2.1, hue = 'gender', data = gender_lunch_group, palette="Blues_d", kind = "bar")


# According to the graph, students who took standard lunch performed better than those who took free/reduced lunch. In the free/reduced category, difference in the marks scored by male students and female students is very less as compared to standard category.

# In[ ]:


gender_parent_group = df.groupby(['gender', 'parental level of education']).mean().reset_index()


# In[ ]:


gender_parent_group.head()


# In[ ]:


sns.catplot(x = 'gender', y = 'average_score', data = gender_parent_group,height=10, aspect = 2.1,palette="Blues_d",  hue = 'parental level of education', kind = 'bar')


# The above graph shows that incase of female students, lowest marks are obtained by those whose parents have only high school degree and highest average marks is obtained by those whose parents have bachelor's degree. Associcate degree and a college degree does not affect the marks much. 
# 
# And Incase of male students, highest average is scored by those students whose parents have master's degree and the lowest is scored by those whose parents have only a high school degree.

# ___

# iii) This section contains analysis of gender and test preperation

# In[ ]:


gender_test_group = df.groupby(['gender', 'test preparation course']).mean().reset_index()


# In[ ]:


gender_test_group.head()


# In[ ]:


sns.catplot(x = 'gender', y = 'average_score', hue = 'test preparation course',height=10, aspect = 2.1,palette="Blues_d", data = gender_test_group, kind = 'bar')


# The difference between female and male is of 2 to 3 points. And the difference between the score of those students who were prepared and who were not prepared is around 5 to 7 points, which is quite significant.

# iv) The part contains analysis of student performance and how it varies with lunch.

# In[ ]:


gender_race_lunch = df.groupby(['gender', 'race/ethnicity', 'lunch']).mean().reset_index()


# In[ ]:


gender_race_lunch.head()


# In[ ]:


sns.catplot(x = 'gender', y = 'average_score', hue = 'lunch',height=10, aspect = 2.1,palette="Blues_d", data = gender_race_lunch, kind = 'bar')


# As we can see from this graph female gender scored most marks as compared to male gender. But in botgh cases, scores are same for those students who were on free/reduced lunch. This means student "performance increases drastically when they took standard lunch".

# ---

# v) This section contains an analysis of how average score depends on parental education and their race/ethnicity.

# In[ ]:


race_parent_group = df.groupby(['race/ethnicity', 'parental level of education']).mean().reset_index()


# In[ ]:


sns.catplot(x = 'parental level of education', y = 'average_score', hue = 'race/ethnicity',height=10, aspect = 2.1,palette="Blues_d", data = race_parent_group, kind = 'bar').set_xticklabels(rotation=30)


# ___

# This notebook can be further expanded for more meaningful analysis. So, feel free to fork it.

# In[ ]:




