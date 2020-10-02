#!/usr/bin/env python
# coding: utf-8

# # Student performace data

# ## Contents
# * [Glimpse of the data](#Glimpse-of-the-data)
# * [Frequenc of all categorical variables](#Frequency-of-all-categorical-variables)
# * [Scores grouped by parental level of education](#Scores-grouped-by-parental-level-of-education)
# * [Visializations by parental level ofeducation](#Visualizations by parental level of education)
# * [Mean Scores across completion of preparation course](#Mean-Scores-across-completion-of-course)
# * [Mean math scores by gender across type of lunch and race](#Mean-math-scores-by-gender-across-type-of-lunch-and-race)
# * [Mean reading scores by gender across type of lunch and race](#Mean-reading-scores-by-gender-across-type-of-lunch-and-race)
# * [Mean writing scores by gender across type of lunch and race](#Mean-writing-scores-by-gender across-type-of-lunch-and-race)
# * [Percentage of scores by gender across lunch type and test course completion](#Percentage-of-scores-by-gender-across-lunch-type-and-test-course)
# 
# 
# 
# 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ## Glimpse of the data
# 

# In[ ]:


df_student_data = pd.read_csv("../input/StudentsPerformance.csv")
df_student_data.head()


# In[ ]:


df_student_data.describe()


# ## Frequency of all categorical variables

# In[ ]:



df_categorical = df_student_data.iloc[:,0:5]          #selecting first first five columns in dataframe

df_categorical.apply(pd.value_counts).fillna(0)   # frequency of all the categorical variables 


# ## Scores grouped by parental level of education

# In[ ]:


df_edu_scores = df_student_data.groupby('parental level of education',as_index=False)[('math score','reading score','writing score')].mean() #Group by mean scores
df_edu_scores


# In[ ]:


df_edu_scores['edulevel'] = (4,5,2,6,3,1)
df_edu_scores_sorted= df_edu_scores.sort_values(["edulevel"], axis=0, ascending=True)


# ## Visualizations by parental level of education

# In[ ]:


sns.set_style("whitegrid")

plt.figure(figsize=(10,7)) 
g=sns.barplot(x="parental level of education", y="math score", data=df_edu_scores,palette="Blues",
              order=["some high school", "high school", "some college","associate's degree", "bachelor's degree", "master's degree"])

g.set(ylim=(0, 80))
plt.title('Mean Math Scores of students across different parental level of education')
plt.xlabel('Increasing level of education')
plt.ylabel('Mean Math Scores')
plt.show()


# In[ ]:


plt.figure(figsize=(12,7)) 
sns.lineplot(x="edulevel", y="math score", data=df_edu_scores,color='red' ,label='math')
sns.scatterplot(x="edulevel", y="math score", data=df_edu_scores,color='red')

sns.lineplot(x="edulevel", y="reading score", data=df_edu_scores,color='blue',label='reading')
sns.scatterplot(x="edulevel", y="reading score", data=df_edu_scores,color='blue')

sns.lineplot(x="edulevel", y="writing score", data=df_edu_scores,color='green',label='writing')
sns.scatterplot(x="edulevel", y="writing score", data=df_edu_scores,color='green')


plt.xticks(np.arange(7), ('0','Some High School', 'High School', 'Some College', 'Associates degree', 'Bachelors degree','Masters degree'))
plt.legend(loc='upper left')
plt.xlabel('Increasing level of education')
plt.ylabel('Mean Scores')
plt.title('Mean Scores of students across increasing parental level of education')
plt.show()
             


# Findings:-
# * The scores of students increases as the educational level of parents increase

# In[ ]:


df_edu_test = df_student_data.groupby('test preparation course',as_index=False)[('math score','reading score','writing score')].mean()
df_melted_edu = pd.melt(df_edu_test,id_vars="test preparation course",var_name="subject",value_name="Scores")
df_melted_edu=df_melted_edu.round(2)


# ## Mean Scores across completion of preparation course

# In[ ]:


g=sns.catplot(x='test preparation course', y='Scores', hue='subject', data=df_melted_edu,kind='bar',height=6.5, aspect=11.7/8.27,
             palette='bright')
for p in g.ax.patches:
    g.ax.annotate(str(p.get_height()), (p.get_x()+p.get_width()/2  , p.get_height() * 1.005),)
g.ax.yaxis.set_ticks(np.arange(0, 90, 5))
plt.legend()
plt.title('Mean Scores across completion of preparation course')
plt.show()


# Findings:
# * The scores of those students who have completed the test preparation are higher

# In[ ]:


df_GRL_test = df_student_data.groupby(['gender','race/ethnicity','lunch'],as_index=False)[('math score','reading score','writing score')].mean()


# ## Mean math scores by gender across type of lunch and race

# In[ ]:


ki=sns.catplot(x='gender', y='math score', hue='lunch',col='race/ethnicity', data=df_GRL_test,kind='bar',palette='bright',height=6, aspect=0.4)
ki.set(ylim=(0, 94))
plt.legend(loc='upper right')
plt.show()


# ## Mean reading scores by gender across type of lunch and race

# In[ ]:


ki=sns.catplot(x='gender', y='reading score', hue='lunch',col='race/ethnicity', data=df_GRL_test,kind='bar',palette='dark',height=6, aspect=0.4)
ki.set(ylim=(0, 95))
plt.legend(loc='upper right')
plt.show()


# ## Mean writing scores by gender across type of lunch and race

# In[ ]:


ki=sns.catplot(x='gender', y='writing score', hue='lunch',col='race/ethnicity', data=df_GRL_test,kind='bar',palette='Pastel1',height=6, aspect=0.4)
ki.set(ylim=(0, 96))
plt.legend(loc='upper right')
plt.show()


# In[ ]:


df_student_data_sum = df_student_data
df_student_data_sum['Scoresum'] =  df_student_data_sum ['math score'] + df_student_data_sum['reading score'] + df_student_data_sum['writing score']
df_student_data_sum['Scorepercentage']  = ( df_student_data_sum['Scoresum'] /300.0)*100
df_percentages = df_student_data_sum.iloc[:,[0,3,4,9]]  


# ## Percentage of scores by gender across lunch type and test course

# In[ ]:


sns.catplot(x='gender', y='Scorepercentage', hue='test preparation course',col='lunch', data=df_percentages,kind='bar',palette='GnBu',height=7, aspect=0.6)
plt.show()


# Findings:-
# * The female students  who have a standard lunch plan and have completed the test preparation course have scored the highest
# 

# 
