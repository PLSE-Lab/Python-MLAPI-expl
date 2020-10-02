#!/usr/bin/env python
# coding: utf-8

# Welcome to my first kernel ! This is my first project, vote if you like, thanks.

# <a id="Librarys"></a> <br>
# # **1. Import Librarys and Dataset:** 
# - Importing Librarys
# - Importing Dataset
# - Looking Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


student = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
student.head()


# Let's look for the data types and unique variable

# In[ ]:


print(student.dtypes)
print(student.nunique())


# <a id="Librarys"></a> <br>
# # **2. OK, Lets, do visualization !** 

# # First , I will do some exploration the math score, reading score, and writing score

# In[ ]:


box1 = go.Box(
    y=student["math score"],
    name="Math score distribution"
)
box2 = go.Box(
    y=student["reading score"],
    name="Reading score distribution"
)
box3 = go.Box(
    y=student["writing score"],
    name="Writing score distribution"
)
data= [box1, box2, box3]

layout = go.Layout(
        title="Score Distribution"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="score-dist")


# In[ ]:


fig, ax = plt.subplots(3,1, figsize=(10,14))
g1 = sns.regplot(x=student['math score'], y=student['reading score'], ax=ax[0])
g2 = sns.regplot(x=student['math score'], y=student['writing score'], ax=ax[1])
g3 = sns.regplot(x=student['reading score'], y=student['writing score'], ax=ax[2])


# # Second , I will explore correlation between columns

# # 1. Gender Column

# In[ ]:


g0 = sns.countplot(x='gender', data=student, palette='hls')
g0.set_title("Gender distribution")
g0.set_xlabel("Gender")
g0.set_ylabel("Count")
plt.show()


# There are more female students than male students

# Gender X Race/Ethnicity

# In[ ]:


bar1 = go.Bar(
    x = student[student["gender"] == 'female']["race/ethnicity"].value_counts().index.values,
    y = student[student["gender"] == 'female']["race/ethnicity"].value_counts().values,
    name = " female student"
)

bar2 = go.Bar(
    x = student[student["gender"] == 'male']["race/ethnicity"].value_counts().index.values,
    y = student[student["gender"] == 'male']["race/ethnicity"].value_counts().values,
    name = " male student" 
)

bar3 = go.Bar(
    x = student[student["gender"] == 'female']["race/ethnicity"].value_counts().index.values,
    y = student[student["gender"] == 'female']["race/ethnicity"].value_counts().values,
    name = " female student"
)

bar4 = go.Bar(
    x = student[student["gender"] == 'female']["race/ethnicity"].value_counts().index.values,
    y = student[student["gender"] == 'female']["race/ethnicity"].value_counts().values,
    name = " female student"
)


data = [bar1, bar2]
layout = go.Layout(
    xaxis = dict(
        title="Student Gender"
    ),
    yaxis = dict(
        title="Race/Ethnicity"
    ),
    title = "Student Gender X Race/Ethnicity"
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="gender-bar")


# Gender X other columns

# In[ ]:


#gender x parental level of education
bar1 = go.Bar(
    x = student[student["gender"]=='female']["parental level of education"].value_counts().index.values,
    y = student[student["gender"]=='female']["parental level of education"].value_counts().values,
    name='female'
)
bar2 = go.Bar(
    x = student[student["gender"]=='male']["parental level of education"].value_counts().index.values,
    y = student[student["gender"]=='male']["parental level of education"].value_counts().values,
    name='male'
)

#gender x lunch
bar3 = go.Bar(
    x = student[student["gender"]=='female']["lunch"].value_counts().index.values,
    y = student[student["gender"]=='female']["lunch"].value_counts().values,
    name='female'
)
bar4 = go.Bar(
    x = student[student["gender"]=='male']["lunch"].value_counts().index.values,
    y = student[student["gender"]=='male']["lunch"].value_counts().values,
    name='male'
)

#gender x test preapartion
bar5 = go.Bar(
    x = student[student["gender"]=='female']["test preparation course"].value_counts().index.values,
    y = student[student["gender"]=='female']["test preparation course"].value_counts().values,
    name='female'
)
bar6 = go.Bar(
    x = student[student["gender"]=='male']["test preparation course"].value_counts().index.values,
    y = student[student["gender"]=='male']["test preparation course"].value_counts().values,
    name='male'
)

data = [bar1, bar2, bar3, bar4, bar5, bar6]

fig = tls.make_subplots(rows=2, cols=2,
                        subplot_titles=('Gender Student X Parental level of education', 'Gender Student X Lunch','','Gender Student X test preparation'))
fig.append_trace(bar1, 1,1)
fig.append_trace(bar2, 1,1)
fig.append_trace(bar3, 1,2)
fig.append_trace(bar4, 1,2)
fig.append_trace(bar5, 2,2)
fig.append_trace(bar6, 2,2)

py.iplot(fig, filename='gender-bar')


# Lets Look the student score by gender

# In[ ]:


math_female = student.loc[student["gender"]=='female']["math score"].values.tolist()
math_male = student.loc[student["gender"]=='male']['math score'].values.tolist()
math_ = student['math score'].values.tolist()

hist1 = go.Histogram(
    x = math_female,
    histnorm = 'probability',
    name = 'female math score'
)

hist2 = go.Histogram(
    x = math_male,
    histnorm = 'probability',
    name = 'male math score'
)

hist3 = go.Histogram(
    x = math_,
    histnorm = 'probability',
    name = 'math score overall'
)

data = [hist1, hist2, hist3]
    
fig = tls.make_subplots(rows=2, cols=2,
                      subplot_titles=('Female math','Male math', 'Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)

fig['layout'].update(showlegend=True, title='Math Score Distribuition', bargap=0.1)
py.iplot(fig, filename='gender-bar')


# In[ ]:


female = student[student["gender"]=='female']
male = student[student["gender"]=='male']

fig, ax = plt.subplots(nrows=2, figsize=(20,10))
plt.subplots_adjust(hspace=0.4, top=1)

g1 = sns.distplot(female["math score"], ax=ax[0],
                         color = 'r')
g1 = sns.distplot(male["math score"], ax=ax[0],
                         color = 'b')
g1.set_title("Math score Distribution by gender", fontsize=15)
g1.set_xlabel("Math Score")
g1.set_ylabel("Frequency")

g2 = sns.countplot(x="math score", data=student, hue="gender",
                          palette='hls', ax=ax[1])
g2.set_title("Math Score Counting by gender", fontsize=15)
g2.set_xlabel("Math Score")
g2.set_ylabel("Count")

plt.show()


# In[ ]:


Male student have higher math score than female student


# In[ ]:


math_female = student.loc[student["gender"]=='female']['reading score'].values.tolist()
math_male = student.loc[student["gender"]=='male']['reading score'].values.tolist()
math_ = student['reading score'].values.tolist()

hist1 = go.Histogram(
    x = math_female,
    histnorm = 'probability',
    name = 'female reading score'
)

hist2 = go.Histogram(
    x = math_male,
    histnorm = 'probability',
    name = 'male reading score'
)

hist3 = go.Histogram(
    x = math_,
    histnorm = 'probability',
    name = 'reading score overall'
)

data = [hist1, hist2, hist3]
    
fig = tls.make_subplots(rows=2, cols=2,
                      subplot_titles=('Female reading','Male Reading', 'Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)

fig['layout'].update(showlegend=True, title='Reading Score Distribuition', bargap=0.1)
py.iplot(fig, filename='gender-bar')


# In[ ]:


female = student[student["gender"]=='female']
male = student[student["gender"]=='male']

fig, ax = plt.subplots(nrows=2, figsize=(20,10))
plt.subplots_adjust(hspace=0.4, top=1)

g1 = sns.distplot(female["reading score"], ax=ax[0],
                         color = 'r')
g1 = sns.distplot(male["reading score"], ax=ax[0],
                         color = 'b')
g1.set_title("Reading score Distribution by gender", fontsize=15)
g1.set_xlabel("Reading Score")
g1.set_ylabel("Frequency")

g2 = sns.countplot(x="reading score", data=student, hue="gender",
                          palette='hls', ax=ax[1])
g2.set_title("Reading Score Counting by gender", fontsize=15)
g2.set_xlabel("Reading Score")
g2.set_ylabel("Count")

plt.show()


# In reading score , female student have higher score

# In[ ]:


math_female = student.loc[student["gender"]=='female']['writing score'].values.tolist()
math_male = student.loc[student["gender"]=='male']['writing score'].values.tolist()
math_ = student['writing score'].values.tolist()

hist1 = go.Histogram(
    x = math_female,
    histnorm = 'probability',
    name = 'female writing score'
)

hist2 = go.Histogram(
    x = math_male,
    histnorm = 'probability',
    name = 'male writing score'
)

hist3 = go.Histogram(
    x = math_,
    histnorm = 'probability',
    name = 'writing score overall'
)

data = [hist1, hist2, hist3]
    
fig = tls.make_subplots(rows=2, cols=2,
                      subplot_titles=('Female writing','Male writing', 'Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)

fig['layout'].update(showlegend=True, title='Writing Score Distribuition', bargap=0.1)
py.iplot(fig, filename='gender-bar')


# In[ ]:


female = student[student["gender"]=='female']
male = student[student["gender"]=='male']

fig, ax = plt.subplots(nrows=2, figsize=(20,10))
plt.subplots_adjust(hspace=0.4, top=1)

g1 = sns.distplot(female["writing score"], ax=ax[0],
                         color = 'r')
g1 = sns.distplot(male["writing score"], ax=ax[0],
                         color = 'b')
g1.set_title("Writing score Distribution by gender", fontsize=15)
g1.set_xlabel("Writing Score")
g1.set_ylabel("Frequency")

g2 = sns.countplot(x="writing score", data=student, hue="gender",
                          palette='hls', ax=ax[1])
g2.set_title("Writing Score Counting by gender", fontsize=15)
g2.set_xlabel("Writing Score")
g2.set_ylabel("Count")

plt.show()


# In writing , female student have higher score

# # 2. Race/ethnicity Columns

# In[ ]:


sns.set_style("darkgrid")
fig, ax = plt.subplots(4,1, figsize=(15,15))
plt.subplots_adjust(hspace=0.4, top=1)

g1 = sns.countplot(x="race/ethnicity", data=student, ax=ax[0],
                  palette='hls')
g1.set_title("Race/ethnicity Distribution", fontsize=15)
g1.set_xlabel("Race/ethnicity", fontsize=12)
g1.set_ylabel("Count", fontsize=12)

g1 = sns.countplot(x="parental level of education", data=student, ax=ax[1],
                  palette='hls', hue="race/ethnicity")
g1.set_title("Race/ethnicity Distribution X Parental Education", fontsize=15)
g1.set_xlabel("Parental level education", fontsize=12)
g1.set_ylabel("Count", fontsize=12)


g2 = sns.countplot(x="lunch", data=student, ax=ax[2],
                  palette='hls', hue='race/ethnicity')
g2.set_title("Race/ethnicity Distribution X Lunch", fontsize=15)
g1.set_xlabel("Lunch", fontsize=12)
g1.set_ylabel("Count", fontsize=12)

g2 = sns.countplot(x="test preparation course", data=student, ax=ax[3],
                  palette='hls', hue='race/ethnicity')
g2.set_title("Race/ethnicity Distribution X test preparation", fontsize=15)
g1.set_xlabel("Test preparation", fontsize=12)
g1.set_ylabel("Count", fontsize=12)


# In[ ]:


math_A = student.loc[student["race/ethnicity"]=='group A']["math score"].values.tolist()
math_B = student.loc[student["race/ethnicity"]=='group B']['math score'].values.tolist()
math_C = student.loc[student["race/ethnicity"]=='group C']["math score"].values.tolist()
math_D = student.loc[student["race/ethnicity"]=='group D']['math score'].values.tolist()
math_E = student.loc[student["race/ethnicity"]=='group E']["math score"].values.tolist()
math_ = student['math score'].values.tolist()

hist1 = go.Histogram(
    x = math_A,
    histnorm = 'probability',
    name = 'math A score'
)

hist2 = go.Histogram(
    x = math_B,
    histnorm = 'probability',
    name = 'math  B score'
)

hist3 = go.Histogram(
    x = math_C,
    histnorm = 'probability',
    name = 'math C score'
)
hist4 = go.Histogram(
    x = math_D,
    histnorm = 'probability',
    name = 'math  D score'
)

hist5 = go.Histogram(
    x = math_E,
    histnorm = 'probability',
    name = 'math  E score'
)

hist6 = go.Histogram(
    x = math_,
    histnorm = 'probability',
    name = 'math score overall'
)


data = [hist1, hist2, hist3, hist4, hist5, hist6]
    
fig = tls.make_subplots(rows=3, cols=2,
                      subplot_titles=('Math A score','Math B score', 'Math C Score', 
                                     'Math D Score', 'Math E Score', 'Math Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)
fig.append_trace(hist4, 2,2)
fig.append_trace(hist5, 3,1)
fig.append_trace(hist6, 3,2)

fig['layout'].update(showlegend=True, title='Math Score Distribuition', bargap=0.1)
py.iplot(fig, filename='race-bar')


# In[ ]:


read_A = student.loc[student["race/ethnicity"]=='group A']["reading score"].values.tolist()
read_B = student.loc[student["race/ethnicity"]=='group B']['reading score'].values.tolist()
read_C = student.loc[student["race/ethnicity"]=='group C']["reading score"].values.tolist()
read_D = student.loc[student["race/ethnicity"]=='group D']['reading score'].values.tolist()
read_E = student.loc[student["race/ethnicity"]=='group E']["reading score"].values.tolist()
read_ = student['reading score'].values.tolist()

hist1 = go.Histogram(
    x = read_A,
    histnorm = 'probability',
    name = 'reading A score'
)

hist2 = go.Histogram(
    x = read_B,
    histnorm = 'probability',
    name = 'reading  B score'
)

hist3 = go.Histogram(
    x = read_C,
    histnorm = 'probability',
    name = 'reading C score'
)
hist4 = go.Histogram(
    x = read_D,
    histnorm = 'probability',
    name = 'reading  D score'
)

hist5 = go.Histogram(
    x = read_E,
    histnorm = 'probability',
    name = 'math  E score'
)

hist6 = go.Histogram(
    x = read_,
    histnorm = 'probability',
    name = 'reading score overall'
)


data = [hist1, hist2, hist3, hist4, hist5, hist6]
    
fig = tls.make_subplots(rows=3, cols=2,
                      subplot_titles=('Readding A score','Reading B score', 'Reading C Score', 
                                     'Reading D Score', 'Reading E Score', 'Reading Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)
fig.append_trace(hist4, 2,2)
fig.append_trace(hist5, 3,1)
fig.append_trace(hist6, 3,2)

fig['layout'].update(showlegend=True, title='Reading Score Distribuition', bargap=0.1)
py.iplot(fig, filename='race-bar')


# In[ ]:


write_A = student.loc[student["race/ethnicity"]=='group A']["writing score"].values.tolist()
write_B = student.loc[student["race/ethnicity"]=='group B']['writing score'].values.tolist()
write_C = student.loc[student["race/ethnicity"]=='group C']["writing score"].values.tolist()
write_D = student.loc[student["race/ethnicity"]=='group D']['writing score'].values.tolist()
write_E = student.loc[student["race/ethnicity"]=='group E']["writing score"].values.tolist()
write_ = student['writing score'].values.tolist()

hist1 = go.Histogram(
    x = write_A,
    histnorm = 'probability',
    name = 'writing A score'
)

hist2 = go.Histogram(
    x = write_B,
    histnorm = 'probability',
    name = 'writing  B score'
)

hist3 = go.Histogram(
    x = write_C,
    histnorm = 'probability',
    name = 'writing C score'
)
hist4 = go.Histogram(
    x = write_D,
    histnorm = 'probability',
    name = 'writing  D score'
)

hist5 = go.Histogram(
    x = write_E,
    histnorm = 'probability',
    name = 'writing  E score'
)

hist6 = go.Histogram(
    x = write_,
    histnorm = 'probability',
    name = 'writing score overall'
)


data = [hist1, hist2, hist3, hist4, hist5, hist6]
    
fig = tls.make_subplots(rows=3, cols=2,
                      subplot_titles=('Writing A score','Writing B score', 'Writing C Score', 
                                     'Writing D Score', 'Writing E Score', 'Writing Overall'))

fig.append_trace(hist1, 1,1)
fig.append_trace(hist2, 1,2)
fig.append_trace(hist3, 2,1)
fig.append_trace(hist4, 2,2)
fig.append_trace(hist5, 3,1)
fig.append_trace(hist6, 3,2)

fig['layout'].update(showlegend=True, title='Writing Score Distribuition', bargap=0.1)
py.iplot(fig, filename='race-bar')


# # 3. Parental Level of Education Columns

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(14,14))

g0 = sns.countplot(x='parental level of education', data=student, ax=ax[0],
                  palette='hls')
g0.set_title("Parental level of education Distribution")
g0.set_xlabel("Parental level of education")
g0.set_ylabel("Count")

g1 = sns.countplot(x='parental level of education', data=student, ax=ax[1],
                  palette='hls', hue='test preparation course')
g1.set_title("Parental level of education X test Preparation Course")
g1.set_xlabel("Parental level of education")
g1.set_ylabel("test preparation course ")


# In[ ]:


fig, ax = plt.subplots(3,1 ,figsize=(14,14))
plt.subplots_adjust(top=1, hspace=0.4)

g1 = sns.boxplot(x='parental level of education',data=student, y='math score', ax=ax[0],
                palette='hls')
g2 = sns.boxplot(x='parental level of education', data=student, y='reading score', ax=ax[1],
               palette='hls')
g3 = sns.boxplot(x='parental level of education', data=student, y='writing score', ax=ax[2],
                palette='hls')
plt.show()


# # 4. Test Preparation Course Columns

# In[ ]:


g0 = sns.countplot(x='test preparation course', data=student, palette='hls')
g0.set_title("Test preparation course distribution")
g0.set_xlabel("Test preparation")
g0.set_ylabel("Count")
plt.show()


# In[ ]:


Box0 = go.Box(
    x=student["test preparation course"],
    y=student["math score"],
    name="Math score x test preparation"
)
Box1 = go.Box(
    x=student["test preparation course"],
    y=student["reading score"],
    name="Reading score x test preparation"
)
Box2 = go.Box(
    x=student["test preparation course"],
    y=student["writing score"],
    name="Writing score x test preparation"
)
data = [Box0, Box1, Box2]
fig = tls.make_subplots(rows=2, cols=2, 
                        subplot_titles=("Math Score Dist","Reading Score Dist","Writing Score Dist"))
fig.append_trace(Box0, 1,1)
fig.append_trace(Box1, 1,2)
fig.append_trace(Box2, 2,1)

fig['layout'].update(height=1000, width=1200, title='Score Distribuition x Test Preparation', boxmode='group')
py.iplot(fig, filename="test-box")


# In[ ]:


fig = {
    "data": [
        {
            "type": 'violin',
            "x": student['test preparation course'],
            "y": student['math score'],
            "legendgroup": 'Math score',
            "scalegroup": 'No',
            "name": 'Math Score',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)


# In[ ]:


fig = {
    "data": [
        {
            "type": 'violin',
            "x": student['test preparation course'],
            "y": student['reading score'],
            "legendgroup": 'Reading score',
            "scalegroup": 'No',
            "name": 'Reading Score',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'Red'
            }
        },
        
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)


# In[ ]:


fig = {
    "data": [
        {
            "type": 'violin',
            "x": student['test preparation course'],
            "y": student['writing score'],
            "legendgroup": 'Writing score',
            "scalegroup": 'No',
            "name": 'Writing Score',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'Green'
            }
        },
        
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)

