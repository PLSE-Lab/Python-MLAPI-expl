#!/usr/bin/env python
# coding: utf-8

# <h1><center>PLOTLY TUTORIAL - 5</center></h1>
# ***
# 
# Kaggle ML and Data Science Survey were live for one week in October 2018. This survey received 23,859 usable respondents from 147 countries and territories. The median time in the survey was around 17 minutes.
# 
# ---
# 
# **Top five countries in 2017:**
# 1. USA - 4197 participants
# 2. India - 2704 participants
# 3. Russia - 578 participants
# 4. United Kingdom - 535  participants
# 5. China - 471 participants
# 
# **Top five countries in 2018:**
# 1. USA - 4716 participants
# 2. India - 4417 participants
# 3. China - 1644 participants
# 4. Russia - 879 participants
# 5. Brazil - 736 participants
# 
# ---
# 
# **Age Distribution in 2017 (left) vs 2018 (right):**
# 
# * 18-21:** 7.2% - 12.7%**
# * 22-24:** 14.9% - 21.5%**
# * 25-29: **25.9% - 25.8%**
# * 30-34: **18.5% - 15.8%**
# * 35-39: **12.6% - 9.4%**
# * 40-44: **7.7% - 5.7%**
# * 44+: **12.9% - 8.4%**
# 
# **% of 30+** people in **2017 is 51.7%** but **39.3% in 2018**. It is a very tremendous decline. Did 30+ people lost their interest or find other platforms we do not know?
# 
# ---
# 
# There are much more men **(81% vs. 17%)** as in 2017 but hopefully, the gap is narrowing in the following years since many women are entering the field.
# 
# The more female start to code actively **(<1 year)** compared to male starters. **(27% vs 24%)**
# 
# Most related to active coding, the more female start to use ML methods at work or in school **(<1 year)** compared to male starters. **(36% vs 33%)**
# 
# % of the master degree of female and male is **52.9%** and **46.4%** respectively.
# 
# % of the bachelor's degree of female and male is **25.7%** and **32.1%** respectively.
# 
# 
# ---
# 
# Computer science, non-cs engineering, mathematics & statistics, and business disciplines are the leading majors. 
# 
# % of male in computers/technology industry is higher than % of female. **(26% vs 23%)**
# 
# % of female in academics/education industry is higher than % of male. **(14% vs 12%)**
# 
# Computers/technology is the leading industry for all majors. Second is the academics/education industry except for business disciplines where **accounting/finance is in the 2nd place.**
# 
# ---
# 
# Python is the most used programs regardless of gender and majors. On the other hand, % of female use R and SQL is higher than male. % of R and SQL is high for those who have major in Math & Stats and Business disciplines.
# 
# In 2018, there is nothing worth to debate about which programming language (Python vs R). **Learn Python (75%) ** and be part of the data science world!
# 
# ---
# 
# 
# Responses of female and male to **"Are you a data scientist?"** question are very similar and most of them consider themselves as a data scientist.
# 
# Responses of people who have different majors are very similar except many math & stats people who are very confident and say "definitely yes" probably results below.
# 
# Active coding in years is similar for all majors except people who major in math & stats. There are more experienced respondents. **(3+ years)**
# 
# Years spent in ML methods are similar for all majors except people who major in math & stats. There are more experienced respondents. **(>1 years)**
# 
# ---
# 
# Top online platforms female and male spent the most amount of time are more or less same: Coursera with **38%** for both genders. As an exception female spend more time on Data Camp **(11% vs 15%)** while male spend more time on Kaggle Learn. **(5% vs 7%)**
# 
# Coursera is again leader online platform for all majors. Secondly, CS and non-CS Engineers prefer Udemy while math & stats and business disciplines block prefer DataCamp.
# 
# ---
# 
# 
# Male and female respondents interact with numerical data mostly. On the other hand, male respondents interact with image and tabular data most compared to female respondents.
# 
# Respondents who in all majors interact with numerical data mostly **except computer engineers who interact with text data more often.**
# 
# ---
# 
# 
# **Almost 90%** of male and female respondents think that independent projects are at least important as academic achievements.
# 
# ---
# 

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import random

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/multipleChoiceResponses.csv')
df = df.iloc[:,~df.columns.str.contains('_')]
df.columns = df.iloc[0]
df = df.iloc[1:]
df.head()


# In[ ]:


df['Duration (in seconds)'] = df['Duration (in seconds)'].astype('int')
df['For how many years have you used machine learning methods (at work or in school)?'] = np.where(df['For how many years have you used machine learning methods (at work or in school)?'] =='I have never studied machine learning but plan to learn in the future', 
                                        'I have never studied ML but plan to learn',
                                       df['For how many years have you used machine learning methods (at work or in school)?'] )

male = df[(df['What is your gender? - Selected Choice'] == 'Male')]
female = df[(df['What is your gender? - Selected Choice'] == 'Female')]

arr_order1 =  [      
               '< 1 year', '1-2 years', '3-5 years', '5-10 years', 
            '10-20 years', '20-30 years','30-40 years', '40+ years'
]

arr_order2 =  [
   '< 1 year', '1-2 years', 
'2-3 years', '3-4 years', '4-5 years','5-10 years', '10-15 years', '20+ years'
    ]

arr_order3 = [
    '0','0-10', '10-20','20-30', '30-40', '40-50', '50-60', '60-70',
     '70-80', '80-90','90-100'
]


# In[ ]:


def HistChart(column, title, limit, angle, order):
    
    count_male = round((male[column].value_counts(normalize=True) * 100),3) 
    count_male = count_male.reindex_axis(order).to_frame()[:limit]
    count_female = round((female[column].value_counts(normalize=True) * 100),3)
    count_female = count_female.reindex_axis(order).to_frame()[:limit]
    
    color1 = random.choice(['red',  'navy', 'pink','orange', 'indigo', 'tomato' 
                            ])
    color2 = random.choice([ 'lightgreen',  'aqua','skyblue', 'lightgrey',  
                            'cyan','yellow'
                           ])
    
    trace1 = go.Bar(
        x=count_male.index,
        y=count_male[column],
        name='male',
        marker=dict(
            color = color1
        )
    )

    trace2 = go.Bar(
        x=count_female.index,
        y=count_female[column],
        name='female',
        marker=dict(
            color = color2
        )
    )

    data = [trace1,trace2]
    layout = go.Layout(xaxis=dict(tickangle=angle),titlefont=dict(size=13),
        title=title, yaxis = dict(title = '%')
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# In[ ]:


def PieChart(column, title, limit):
    
    color = ['red',  'navy',  'cyan', 'lightgrey','orange', 'gold','lightgreen', 
                            '#D0F9B1','tomato', 'tan']
    
    count_male = male[column].value_counts()[:limit].reset_index()
    count_female = female[column].value_counts()[:limit].reset_index()
    
    trace1 = go.Pie(labels=count_male['index'], 
                    values=count_male[column], 
                    name= "male", 
                    hole= .5, 
                    domain= {'x': [0, .48]},
                   marker=dict(colors=color))

    trace2 = go.Pie(labels=count_female['index'], 
                    values=count_female[column], 
                    name="female", 
                    hole= .5,  
                    domain= {'x': [.52, 1]})

    layout = dict(title= title, font=dict(size=12), legend=dict(orientation="h"),
                  annotations = [
                      dict(
                          x=.20, y=.5,
                          text='Male', 
                          showarrow=False,
                          font=dict(size=20)
                      ),
                      dict(
                          x=.81, y=.5,
                          text='Female', 
                          showarrow=False,
                          font=dict(size=20)
                      )
        ])
    
    fig = dict(data=[trace1, trace2], layout=layout)
    py.iplot(fig)


# In[ ]:


def major_comparison(comp, eng, math_stat, business, column,title, angle,limit, order=None):
    
    df1 = round(comp[column]             .value_counts(normalize=True), 4).to_frame()[:limit]
    df1 = df1.reindex_axis(order)

    df2 = round(eng[column]             .value_counts(normalize=True), 4).to_frame()[:limit]
    df2 = df2.reindex_axis(order)
    
    df3 = round(math_stat[column]             .value_counts(normalize=True), 4).to_frame()[:limit]
    df3 = df3.reindex_axis(order)
    
    df4 = round(business[column]             .value_counts(normalize=True), 4).to_frame()[:limit]
    df4 = df4.reindex_axis(order)
    
    trace1 = go.Bar(
        x=df1.index,
        y=df1[column],
        name='CS Engineer',
        marker=dict(
         color = 'red'
        )
    )

    trace2 = go.Bar(
        x=df2.index,
        y=df2[column],
         name='Non-CS Engineer',
        marker=dict(
            color = 'navy'
        )
    )

    trace3 = go.Bar(
        x=df3.index,
        y=df3[column],
         name='Math & Stats',
        marker=dict(
            color = 'grey'
        )
    )

    trace4 = go.Bar(
        x=df4.index,
        y=df4[column],
         name='Business Majors',
        marker=dict(
            color = 'aqua'
        )
    )

    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Computer Engineer', 'Non-CS Engineer',
                                                              'Math & Stats','Business Majors'))
    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    fig.append_trace(trace3, 2,1)
    fig.append_trace(trace4, 2,2)
    
    fig['layout'].update( height=500, width=850, 
                         title=title,  font=dict(size=10),
                         showlegend=False)  
    fig['layout']['xaxis1'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis2'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis3'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis4'].update(dict(tickangle=angle, tickfont = dict(size = 10)))
    py.iplot(fig)


# In[ ]:


colors = ['aqua', 'lightgrey', 'lightgreen', '#D0F9B1', 'khaki', 'grey']

def PieChart2(column, title, limit):
    count_trace = df[column].value_counts()[:limit].reset_index()
    trace1 = go.Pie(labels=count_trace['index'], 
                    values=count_trace[column], 
                    name= "count", 
                    hole= .5, 
                    textfont=dict(size=10),
                   marker=dict(colors=colors))
    layout = dict(title= title, font=dict(size=15))
    
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)


# <h3><center>In which country do you currently reside?</center></h3>

# In[ ]:


count_geo = df.groupby('In which country do you currently reside?')['In which country do you currently reside?'].count()

data = [dict(
        type = 'choropleth',
        locations = count_geo.index,
        locationmode = 'country names',
        z = count_geo.values,
        text = count_geo.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],
                      [0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],
                      [1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = ''),
      ) ]

layout = dict(
    title = 'Number of Participants by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# **Top five countries in 2017:**
# 1. USA - 4197 participants
# 2. India - 2704 participants
# 3. Russia - 578 participants
# 4. United Kingdom - 535  participants
# 5. China - 471 participants
# 
# **Top five countries in 2018:**
# 1. USA - 4716 participants
# 2. India - 4417 participants
# 3. China - 1644 participants
# 4. Russia - 879 participants
# 5. Brazil - 736 participants

# <h3><center>What is your gender?</center></h3>

# In[ ]:


PieChart2('What is your gender? - Selected Choice', '', 5)


# **There are much more men (81% vs. 17%) as in 2017 but hopefully, the gap is narrowing in the following years since many women are entering the field.**

# <h3><center>Duration (in seconds) spent in the survey</center></h3>

# In[ ]:


df = df[~((df['What is your current yearly compensation (approximate $USD)?'] == 'I do not wish to disclose my approximate yearly compensation') |
   (df['How long have you been writing code to analyze data?'] == 'I have never written code and I do not want to learn') |
           (df['How long have you been writing code to analyze data?'] == 'I have never written code but I want to learn') |
      (df['For how many years have you used machine learning methods (at work or in school)?'] == 'I have never studied ML but plan to learn') |
        (df['For how many years have you used machine learning methods (at work or in school)?'] == 'I have never studied machine learning and I do not plan to'))]
  
labels = ['male', 'female']
colors = ['navy', 'deeppink']

dur1 = df[(df['What is your gender? - Selected Choice'] == 'Male') & 
          (df['Duration (in seconds)'] < 3600)]['Duration (in seconds)']
dur2 = df[(df['What is your gender? - Selected Choice'] == 'Female') & 
          (df['Duration (in seconds)'] < 3600)]['Duration (in seconds)']

hist_data = [dur1, dur2]
fig = ff.create_distplot(hist_data, labels, colors=colors, 
                         show_hist=False)

fig['layout'].update(title='')
py.iplot(fig)


# **I exclude answers that take more than 1 hour because the survey is short and well-organized. Especially, people spending for more than 20-30 minutes have probably on other tabs.
# **

# In[ ]:


trace0 = go.Box(x=male[male['Duration (in seconds)'] < 3600]                ['Duration (in seconds)'], name="Male",fillcolor='navy')
trace1 = go.Box(x=female[female['Duration (in seconds)'] < 3600]                ['Duration (in seconds)'],name="Female",fillcolor='deeppink')

data = [trace0, trace1]
layout = dict(title='')

fig = dict(data=[trace0, trace1], layout=layout)
py.iplot(fig)


# <h3><center>What is your age (# years)?</center></h3>

# In[ ]:


age1 = round(df['What is your age (# years)?'].value_counts(normalize=True).             to_frame().sort_index(), 4)
age2 = round(df[df['What is your gender? - Selected Choice'] == 'Male']                    ['What is your age (# years)?'].value_counts(normalize=True).             to_frame().sort_index(), 4)
age3 = round(df[df['What is your gender? - Selected Choice'] == 'Female']                    ['What is your age (# years)?'].value_counts(normalize=True).             to_frame().sort_index(), 4)

trace = [
    go.Bar(x=age1.index,
    y=age1['What is your age (# years)?'],
                opacity = 0.8,
                 name="total",
                 hoverinfo="y",
                 marker=dict(
        color = age1['What is your age (# years)?'],
        colorscale='Reds',
        showscale=True)
                ),
    
    go.Bar(x=age2.index,
    y=age2['What is your age (# years)?'],
                 visible=False,
                 opacity = 0.8,
                 name = "male",
                 hoverinfo="y",
                 marker=dict(
        color = age2['What is your age (# years)?'],
        colorscale='Blues',
        reversescale = True,
        showscale=True)
                ),
    
    go.Bar(x=age3.index,
    y=age3['What is your age (# years)?'],
                 visible=False,
                opacity = 0.8,
                 name = "female",
                 hoverinfo="y",
                marker=dict(
        color = age3['What is your age (# years)?'],
        colorscale='Bluered',
        reversescale = True,
        showscale=True)    
                )
]

layout = go.Layout(title = '',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="",
                             titlefont=dict(size=20),
                             tickmode="linear"),
                   yaxis=dict(title="%",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False, False, False, False]}],
            label="Total",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False, False, False, False]}],
            label="Male",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True, False, False, False]}],
            label="Female",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# **Age Distribution in 2017 (left) vs 2018 (right):**
# 
# * 18-21:** 7.2% - 12.7%**
# * 22-24:** 14.9% - 21.5%**
# * 25-29: **25.9% - 25.8%**
# * 30-34: **18.5% - 15.8%**
# * 35-39: **12.6% - 9.4%**
# * 40-44: **7.7% - 5.7%**
# * 44+: **12.9% - 8.4%**
# 
# **% of 30+ people in 2017 is 51.7% but 39.3% in 2018. It is a very tremendous decline. Did 30+ people lost their interest or find other platforms we do not know?**

# **Computer science, non-cs engineering, mathematics & statistics, and business disciplines are the leading majors.  **

# <h3><center>What is the highest level of formal education that you have attained or <br>plan to attain within the next 2 years?</br></center></h3>

# In[ ]:


PieChart("What is the highest level of formal education that you have attained or plan to attain within the next 2 years?", "", 5)


# **% of master degree of female and male is 52.9% and 46.4% respectively.**
# 
# **% of the bachelor's degree of female and male is 25.7% and 32.1% respectively.**

# <h3><center>Which best describes your undergraduate major?</center></h3>

# In[ ]:


PieChart("Which best describes your undergraduate major? - Selected Choice", "", 6)


# **Let's compare answers of men & women and top 4 majors. (CS, Engineering, Math & Stats, Business Disciplines)**

# In[ ]:


comp = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Computer science (software engineering, etc.)']
eng = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Engineering (non-computer focused)']
math_stat = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Mathematics or statistics']
business = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'A business discipline (accounting, economics, finance, etc.)']


# <h3><center>What is your current yearly compensation (approximate $USD)?</center></h3>

# In[ ]:


arr_order4 =  [
       '0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
       '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
        '100-125,000','125-150,000','150-200,000', '200-250,000', 
        '250-300,000','300-400,000', '400-500,000','500,000+'
]

age1 = round(df['What is your current yearly compensation (approximate $USD)?']             .value_counts(normalize=True).to_frame().sort_index(), 4)
age1 = age1.reindex_axis(arr_order4)

age2 = round(df[df['What is your gender? - Selected Choice'] == 'Male']                    ['What is your current yearly compensation (approximate $USD)?'].             value_counts(normalize=True).to_frame().sort_index(), 4)
age2 = age2.reindex_axis(arr_order4)

age3 = round(df[df['What is your gender? - Selected Choice'] == 'Female']                    ['What is your current yearly compensation (approximate $USD)?'].             value_counts(normalize=True).to_frame().sort_index(), 4)
age3 = age3.reindex_axis(arr_order4)

trace = [
    go.Bar(x=age1.index,
    y=age1['What is your current yearly compensation (approximate $USD)?'],
                opacity = 0.7,
                 name="total",
                 hoverinfo="y",
                 marker=dict(
        color = age1['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Blues',
        reversescale = True,
        showscale=True)
                ),
    
    go.Bar(x=age2.index,
    y=age2['What is your current yearly compensation (approximate $USD)?'],
                 visible=False,
                 opacity = 0.7,
                 name = "male",
                 hoverinfo="y",
                 marker=dict(
        color = age2['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Reds',
        showscale=True)
                ),
    
    go.Bar(x=age3.index,
    y=age3['What is your current yearly compensation (approximate $USD)?'],
                 visible=False,
                opacity = 0.7,
                 name = "female",
                 hoverinfo="y",
                marker=dict(
        color = age3['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Bluered',
        reversescale = True,
        showscale=True)    
                )
]

layout = go.Layout(title = '',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="", tickangle=30,
                             titlefont=dict(size=12),
                             tickmode="linear"),
                   yaxis=dict(title="%",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False, False, False, False]}],
            label="Total",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False, False, False, False]}],
            label="Male",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True, False, False, False]}],
            label="Female",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In[ ]:


major_comparison(comp, eng, math_stat, business,'What is your current yearly compensation (approximate $USD)?','Salary distribution by majors',40, 20,arr_order4 )


# **Inferences:**
# 
# **- Salary distributions of male and female participants are similar.**
# 
# **- Respondents who majored in business disciplines earn higher salaries while computer engineers seem worst. **
# 
# **- Results are probably misleading since 1 dollar is not the same in USA and India. PPT purchasing power parity) adjusted salaries should be compared rather than local ones. **
# 
# [https://data.oecd.org/conversion/purchasing-power-parities-ppp.htm](http://)

# <h3><center>Select the title most similar to your current role (or most recent title if retired)</center></h3>

# In[ ]:


HistChart("Select the title most similar to your current role (or most recent title if retired): - Selected Choice", "Job titles by gender",10,15, df["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].unique())
order = df["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,"Select the title most similar to your current role (or most recent title if retired): - Selected Choice",'Job titles by majors',20, 10,order )


# **Inferences:**
# 
# **- Top three jobs for male are a data scientist, software engineer, data analyst (18%-14%-8%) for female are as a data scientist, data analyst, software engineer. (17%-10%-9%)**
# 
# **- % of data scientist is higher for all majors except computer engineers % of a software engineer is the highest.**

# <h3><center>In what industry is your current employer/contract?</center></h3>

# In[ ]:


HistChart('In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice', 'Industries by gender', 10,15, df['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].unique())
order = df['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice','Industries by majors',20, 10,order )


# **Inferences:**
# 
# **- % of male in computers/technology industry is higher than % of female. (26% vs 23%)**
# 
# **- % of female in academics/education industry is higher than % of male. (14% vs 12%)**
# 
# **- Computers/technology is the leading industry for all majors. Second is the academics/education industry except for business disciplines where accounting/finance is in the 2nd place.**

# <h3><center>What specific programming language do you use most often?</center></h3>

# In[ ]:


HistChart('What specific programming language do you use most often? - Selected Choice', 'Programs most often used by gender', 10, 15,
         df['What specific programming language do you use most often? - Selected Choice'].unique())
order = df['What specific programming language do you use most often? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What specific programming language do you use most often? - Selected Choice','Programs most often used by majors',20, 10,order )


# **Python is the most used programs regardless of gender and majors. On the other hand, % of female use R and SQL is higher than male. % of R and SQL is high for those who have major in Math & Stats and Business disciplines.**

# <h3><center>What programming language would you recommend <br>an aspiring data scientist to learn first?</br></center></h3>

# In[ ]:


HistChart('What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice', 'Recommended programs by gender', 10, 0,
         df['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].unique())
order = df['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice','Recommended programs by majors',20, 10,order )


# **There is nothing to debate. Learn Python and be part of the data science world!**

# <h3><center>Which ML library have you used the most?</center></h3>

# In[ ]:


HistChart('Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice', 'Top ML libraries by gender', 12,20,
         df['Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice'].unique())
order = df['Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice','ML libraries by majors',20, 10,order )


# **% of the female who use Scikit-learn and random forest is higher than % of male. **
# 
# **Male use Keras, Tensorflow and Pytorch libraries more compared to females.**
# 
# **Scikit-learn is the most used library for all majors. CS and non-CS block use Tensorflow and Keras more compared to math & stats and business block.**

# <h3><center>Which specific data visualization library or tool have you used the most?</center></h3>

# In[ ]:


HistChart('Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice', 'Top visualization libraries by gender', 10, 0,
         df['Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].unique())
order = df['Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice','Top visualization libraries by majors',20, 10,order )


# **% of men who used matplotlib is higher than % of women while for ggplot vice versa. **
# 
# **CS and Non-CS Engineer block do not use ggplot2 frequently as math & stats and business blocks.**
# 
# Warning: Visualization libraries are related to a specific language so results are also related to program choices.

# <h3><center>Do you consider yourself to be a data scientist?</center></h3>

# In[ ]:


HistChart('Do you consider yourself to be a data scientist?', 'Are you a data scientist?', 10,0,
         df['Do you consider yourself to be a data scientist?'].unique())
order = df['Do you consider yourself to be a data scientist?'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Do you consider yourself to be a data scientist?','Are you a data scientist?',20, 10,order )


# **Responses of female and male to "Are you a data scientist?" question are very similar and most of them consider themselves as a data scientist.**
# 
# **Responses of people who have different majors are very similar except many math & stats people who are very confident and say "definitely yes".**

# <h3><center>On which online platform have you spent the most amount of time?</center></h3>

# In[ ]:


HistChart('On which online platform have you spent the most amount of time? - Selected Choice', 'Top online platforms by gender', 8, 10,
         df['On which online platform have you spent the most amount of time? - Selected Choice'].unique())
order = df['On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'On which online platform have you spent the most amount of time? - Selected Choice','Top online platforms by majors',20, 10,order )


# **Top online platforms female and male spent the most amount of time are more or less same: Coursera with 38% for both genders. As an exception female spend more time on Data Camp (11% vs 15%) while male spend more time on Kaggle Learn. (5% vs 7%)**
# 
# **Courser is again leader online platform for all majors. Secondly, CS and non-CS Engineers prefer Udemy while math & stats and business disciplines block prefer DataCamp.**

# <h3><center>How long have you been writing code to analyze data?</center></h3>

# In[ ]:


HistChart('How long have you been writing code to analyze data?', 
          'Active coding by gender', 20, 13, arr_order1)
major_comparison(comp, eng, math_stat, business,'How long have you been writing code to analyze data?','Active coding by majors',20, 10,arr_order1 )


# **There is good news since the more female start to code actively (<1 year) compared to male starters. (27% vs 24%)**
# 
# **Active coding in years is similar for all majors except people who major in math & stats. There are more experienced respondents. (3+ years)**

# <h3><center>For how many years have you used machine learning methods (at work or in school)?</center></h3>

# In[ ]:


HistChart('For how many years have you used machine learning methods (at work or in school)?', 'Years spent in ML by gender', 10,13, arr_order2)
major_comparison(comp, eng, math_stat, business,'For how many years have you used machine learning methods (at work or in school)?', 'Years spent in ML by majors',20, 10, arr_order2 )


# **Most related to active coding, the more female start to use ML methods at work or in school (<1 year) compared to male starters. (36% vs 33%)**
# 
# **Years spent in ML methods is similar for all majors except people who major in math & stats. There are more experienced respondents. (>1 years)**

# <h3><center>What is the type of data that you currently interact with most often at work or school?</center></h3>

# In[ ]:


HistChart('What is the type of data that you currently interact with most often at work or school? - Selected Choice', 'Data types by gender', 9, 10,
         df['What is the type of data that you currently interact with most often at work or school? - Selected Choice'].unique())
order = df['What is the type of data that you currently interact with most often at work or school? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What is the type of data that you currently interact with most often at work or school? - Selected Choice','Data types by majors',20, 10,order )


# **Male and female respondents interact with numerical data mostly. On the other hand, male respondents interact with image and tabular data most compared to female respondents.**
# 
# **Respondents who in all majors interact with numerical data mostly except computer engineers who interact with text data more often.**

# <h3><center>Approximately what percent of your data projects involved <br>exploring unfair bias in the dataset and/or algorithm?</br></center></h3>

# In[ ]:


HistChart('Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?', 'Time spent on unfair bias by gender', 10,0,
         arr_order3)
major_comparison(comp, eng, math_stat, business,'Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?', 'Time spent on unfair bias by majors',40, 20,arr_order3 )


# **% of time spent on the unfair bias is more or less similar for both genders majors. I am not sure whether respondents understand the question or not correctly.**

# <h3><center>Approximately what percent of your data projects involve exploring model insights?</center></h3>

# In[ ]:


HistChart('Approximately what percent of your data projects involve exploring model insights?', '% of projects involve exploring model insights by gender', 10, 0, arr_order3)
major_comparison(comp, eng, math_stat, business,'Approximately what percent of your data projects involve exploring model insights?', '% of projects involve exploring model insights by majors',40, 20,arr_order3 )


# **% of time exploring model insights are more or less similar for both genders but those who major in math & stats and business disciplines spend more time on this subject.**

# <h3><center>Which better demonstrates expertise in data science: academic achievements or independent projects?</center></h3>

# In[ ]:


PieChart('Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:', "Which demonsrates expertise in Data Science?", 6)


# **Almost 90% of male and female respondents think that independent projects are at least important as academic achievements.**

# <h3><center>Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?</center></h3>
# 

# In[ ]:


PieChart('Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?', 'Are ML models black boxes?', 10)


# **13% of female say "I do not know; I have no opinion on the matter" while 7.95% of male replied with the same way.**

# You can also check the analysis of 2017 Kaggle survey and other Plotly notebooks below.
# 
# **PLOTLY TUTORIAL - 1 (Kaggle ML and Data Science Survey): https://www.kaggle.com/hakkisimsek/plotly-tutorial-1**
# 
# **PLOTLY TUTORIAL - 2 (2015 Flight Delays and Cancellations): https://www.kaggle.com/hakkisimsek/plotly-tutorial-2**
# 
# **PLOTLY TUTORIAL - 3 (S&P 500 Stock Data): https://www.kaggle.com/hakkisimsek/plotly-tutorial-3**
# 
# **PLOTLY TUTORIAL - 4 (Google Store Customer Data): https://www.kaggle.com/hakkisimsek/plotly-tutorial-4**
# 
# **Your feedback really matters - please share your thoughts and suggestions.**
