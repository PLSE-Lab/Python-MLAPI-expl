#!/usr/bin/env python
# coding: utf-8

# # Student Performance in Exams
# 
# This notebook provides the in depth analysis on the [student performance in exams at public schools](http://roycekimmons.com/tools/generated_data/exams).
# 
# ![img_exam](https://www.sciencenewsforstudents.org/wp-content/uploads/2019/11/860_test_anxiety.png)

# ## Importing Libraries

# In[ ]:


#!/usr/bin/env python -W ignore::DeprecationWarning

# Data Handling 
import pandas as pd
import numpy as np
from itertools import combinations

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import HTML
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style('whitegrid')


# In[ ]:


pwd


# ## Reading Data

# In[ ]:


df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Viewing data

# In[ ]:


df.reset_index()
df.head(10)


# In[ ]:


(df.head(20)
 .style
 .hide_index()
 .bar(color='#70A1D7', vmin=0, subset=['math score'])
 .bar(color='#FF6F61', vmin=0, subset=['reading score'])
 .bar(color='mediumspringgreen', vmin=0, subset=['writing score'])
 .set_caption(''))


# ### Analysis of categorical attributes
# 
# ### A Bar Plot & Count Plot w.r.t **Gender**, **Race/ethnicity**, **Parental Level of Education**, **Lunch** and **Test Preparation Course**.

# In[ ]:



for attribute in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
  f, ax = plt.subplots(1,2)
  data = df[attribute].value_counts().sort_index()
  bar = sns.barplot(x = data.index, y = data, ax = ax[0], palette="Set2",)
  for item in bar.get_xticklabels():
    item.set_rotation(45)
  ax[1].pie(data.values.tolist() , labels= [i.title() for i in data.index.tolist()], autopct='%1.1f%%',shadow=True, startangle=90);
  plt.show()


# ### Distribution Plots of Numeric Attributes **Math Score**, **Reading Score** and **wrtiting Score**

# In[ ]:



for lab, col in zip(['math score', 'reading score', 'writing score'], ['tomato', 'mediumspringgreen', 'blue']):
  sns.distplot(df[lab], label=lab.title(), color = col, ).set(xlabel=lab.title(), ylabel='Count')
  plt.show()


# ### Relationship Between Numerical Attributes

# In[ ]:



for attr, col in zip(list(combinations(['math score', 'reading score', 'writing score'], 2)), ['#77DF79', '#82B3FF', '#F47C7C']):
  sns.jointplot(df[attr[0]], df[attr[1]], color = col)
  plt.show()


# ### Barplot between **Parent Level of education** and the student score in **Math**, **Reading** and **Writing**

# In[ ]:



df.groupby('parental level of education')['math score', 'reading score', 'writing score'].mean().plot(kind = 'bar');


# ### Barplot between **Parent Level of education** and the student score in **Reading** and **Writing**

# In[ ]:



cond_plot = sns.FacetGrid(data=df, col='parental level of education', hue='gender', col_wrap=3, height = 5)
cond_plot.map(sns.scatterplot, 'reading score', 'writing score' );


# ### Bar graph between **Race/Ethnicity** and **Test Preparation Course**

# In[ ]:


df.groupby('race/ethnicity')['test preparation course'].value_counts().plot(kind = 'bar', colormap='Set2')
plt.ylabel('Count');


# ## Do upvote if you like
