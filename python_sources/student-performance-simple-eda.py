#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/StudentsPerformance.csv")
print("Data Loaded")


# In[ ]:


df.head()


# Rename the columns for consistency of the data. Alter the data by using *inplace* = *True*.

# In[ ]:


df.rename(columns={'parental level of education': 'parental_level_of_education', 'test preparation course': 'test_preparation_course', 'math score': 'math_score', 'reading score': 'reading_score', 'writing score': 'writing_score'}, inplace = True)


# In[ ]:


# PLOT THE GENDERS OF THE DATASET
df['gender'].value_counts().plot.bar()


# Now let's check the educational attainment of the parent of the Students by making a bar chart.

# In[ ]:


df['parental_level_of_education'].value_counts().plot.bar()


# In[ ]:


#sns.countplot(x=df.parental_level_of_education)


# We will get the mean/average of the score's of the Student's based on the level of education of their Parents. 

# In[ ]:


df.groupby("parental_level_of_education", as_index=True)[["math_score", "reading_score", "writing_score"]].mean()


# As you can see on the above data there seems to be a difference with those students whose parent's educational attainment are higher. Let's now visualize the data that we have.

# In[ ]:


score_grouped = df.groupby("parental_level_of_education", as_index=True)[["math_score", "reading_score", "writing_score"]].mean().sort_values(by='writing_score',ascending=False)
score_grouped.plot.bar(title = "Students Score Average per Parental Level of Education", figsize=(20,10))


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "math_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Math Score");


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "writing_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Writing Score");


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("parental_level_of_education", "reading_score", "gender", data=df, kind="box", height=5, aspect= 2)
    g.set_axis_labels("Parental Level of Education", "Reading Score");


# As you can see above, the males are good in Math than females. But females takes the lead in the other two subject which is writing and reading. We can also some difference on the score depending in the parent's educational status. Those students whose parents have Master's Degree have significantly better scores than the other students whose parents finished some high school.

# Let's add a column for the average of each students for more exploration of our data.

# In[ ]:


df["overall_score"] = np.nan
df.head()


# Now let's assign the value for the *overall_score* column that we added by adding the 3 subjects and getting their average.

# In[ ]:


df["overall_score"] = round((df["math_score"] + df["writing_score"] + df["reading_score"]) / 3, 2)
df.head()


# Which among males and females prepared more for their exams?

# In[ ]:


df.groupby(["gender", "test_preparation_course"]).size()


# Despite females preparing less than males, they still got a better score. Let's check whether the lunch also affected the scores for their test. 

# In[ ]:


df.groupby(["gender", "lunch"]).size()


# In[ ]:


gender_test_preparation = df.groupby(["gender", "test_preparation_course"]).size().unstack(fill_value=0).plot.bar()
gender_test_preparation.plot(figsize=(10, 5))


# In[ ]:


gender_test_preparation = df.groupby(["gender", "lunch"]).size().unstack(fill_value=0).plot.bar()
gender_test_preparation.plot(figsize=(10, 5))


# In[ ]:


with sns.axes_style(style='ticks'):
    b = sns.catplot("test_preparation_course", "overall_score", "gender", data=df, kind="box", height=5, aspect=2)
    b.set_axis_labels("Test Preparation", "Test Scores Average")


# In average, females got higher score for the exam because more they prepared.
