#!/usr/bin/env python
# coding: utf-8

# 

# # Overview.
# 
# **In this kernel I will explore the placement dataset of jain university bangalore . we will start with some basic analysis and then we will try and have a look at what factors are important for a student to get placed and get a good salary. The aim is to have a basic idea of what the employers are looking for in a student before offering a job offer and do marks play an important role to decide a student's placement and salary. I will try and keep it simple by generating insights through simple visualizations like bar graphs and histograms. 
# Insights of the visualizations have been included which contain the answers to the above questions.**

# In[ ]:


import pandas as pd  #for data analysis
import matplotlib.pyplot as plt   #for plotting graphs
import numpy as np    # for mathematical functions
import seaborn as sns   # for interactive data visualizations.
import plotly.graph_objects as go  #  for interactive data visualizations.
import plotly.express as px   #for interactive data visualizations.


# In[ ]:


df=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# The data shows that 215 students have applied to get placed through the campus recruitment .

# In[ ]:


df.specialisation.unique()


# The college offers 2 specializations . marketing & HR and Marketing& finance.

# In[ ]:


data=go.Histogram(x=df.specialisation,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Specialization statistics", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Specialization",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# Out of the total students 55% students belong to MARKETING&FINANCE specialization. 

# In[ ]:


data=go.Histogram(x=df.status,histnorm="percent",marker=dict(
                              line=dict(width=2,
                                        color='black')))
layout=go.Layout(title="Placement percentage in Campus recruitment",
        xaxis = dict(
        tickangle = 0,
        title_text = "Placement Status",
        title_font = {"size": 20},
        title_standoff = 30),
        yaxis = dict(
        title_text = "Percentage",
        title_standoff = 30,
        title_font = {"size": 20}))
figure=go.Figure(data=data,layout=layout)
figure


# The placement rate looks average as only 68% of the students have been placed. 

# Lets analyze the placed students closely.

# In[ ]:


data=go.Histogram(x=df[df["status"] =="Placed"].specialisation,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Placement statistics", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Specialization",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# Out of the Placed Students 36 % students belong to MARKETING&HR specialization. Rest belong to MARKETING&FINANCE specialization.

# In[ ]:


data=go.Histogram(x=df[df["status"] =="Placed"].workex,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Effect of work experience on placement", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Work Experience",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# This came out as a surprise. Most of the students who have got placed don't have any work experience (57%).

# In[ ]:


data=go.Histogram(x=df[df["status"] =="Placed"].degree_t,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "degree statistics of placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "degree background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# 69% of the students who got placed belong to commerce and management background. only 28% of the placed students belong to science & technology background.

# In[ ]:


plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'YlOrRd', linewidths=3, linecolor='black')
plt.show()


# The correlation matrix clearly says that the salary dosen't depend much on the marks. It depends on other factors also like work experience, performance in interview, talent etc.

# In[ ]:


data=go.Histogram(x=df[df["status"] =="Placed"].salary,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Salary range of placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Salary",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# About 66% of the placed students have been offered a salary between  2 and 3 LPA. This shows that the placement for the has been quite average for a business school. 

# In[ ]:


fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="specialisation",y="salary",data=df,ax=ax)


# # Insights
# ****1. About 66% of the placed students have been offered a salary between 2 and 3 LPA. This shows that the placement for the has been quite average for a business school.
# 
# 2.The correlation matrix clearly says that the salary dosen't depend much on the marks. It depends on other factors also like work experience, performance in interview, talent etc.
# 
# 3.69% of the students who got placed belong to commerce and management background. only 28% of the placed students belong to science & technology background.
# 
# 4.Most of the students who have got placed don't have any work experience (57%).
# 
# 5.Out of the Placed Students 36 % students belong to MARKETING&HR specialization. Rest belong to MARKETING&FINANCE specialization.
# 
# 6.The placement rate looks average as only 68% of the students have been placed.
# 
# 
# 
# 
# 
# 
# 
# 

# # Lets check the characteristics of the top 50 placed students in terms of salary.

# In[ ]:


data=go.Histogram(x=df.nlargest(50,"salary").degree_t,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "degree statistics of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "degree background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# In[ ]:


data=go.Histogram(x=df.nlargest(50,"salary").specialisation,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "mba specialisation of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "specialisation",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# In[ ]:


data=go.Histogram(x=df.nlargest(50,"salary").workex,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "work experience statistics of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "work experience background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# In[ ]:


data=go.Histogram(x=df.nlargest(50,"salary").mba_p,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "mba marks of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "marks",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure


# In[ ]:


plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.nlargest(50,"salary").corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'YlOrRd', linewidths=3, linecolor='black')
plt.show()


# previous degree, hsc and ssc marks dont seem have much effect on deciding the salary of a candidate. it depends on various other factors like talent,skill level, performance in interview, work experience.

# 

# # INSIGHTS about top 50 placed students.
#  
#  1.**About 40% of the top 50 placed students have passed their mba with a percentage between 60 and 65. this shows that high marks high salary    is a myth.**
#  
#  2.**about 52% of them are freshers**. 
#  
#  3.**about 72% of them have Marketing Finance as their specialisation**.
#  
#  4.**60% of them are from commerce and management background.**

# # CONCLUSION
# **Placements play an important role in a student's career. however there are a lot of myths surrounding placements. Marks play an important role but its not the deciding factor for a student as evident from above analysis. sit depends on other factors as well like skill level,talent, performance in interviews, student's work experience and also his or her behaviour.**
