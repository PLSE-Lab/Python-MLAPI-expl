#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:26:50 2018

@author: abdullahkhilji
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.figure_factory as fffig
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import squarify


df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='latin-1', low_memory=False)

df.describe(include='all')

df.info()

temp_series = df['GenderSelect'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Gender distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="gender")


df['GenderSelect'].describe()
df['GenderSelect'].value_counts()


con_df = pd.DataFrame(df['Country'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_resp', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(10)




data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_resp'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = ''),
      ) ]
layout = dict(
    title = '',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')







tree=df['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('cubehelix_r',52))
plt.rcParams.update({'font.size':45})
fig=plt.gcf()
fig.set_size_inches(70,30)
plt.show()









df['Country'].describe()




df['Age'].describe()


fig = fffig.create_distplot([df[df['Age'] > 0]['Age']], ['age'])
py.iplot(fig, filename='Basic Distplot')





age_usa = df.groupby('Country').get_group('United States')
age_india = df.groupby('Country').get_group('India')

u_fig = fffig.create_distplot([age_usa[age_usa['Age'] > 0]['Age']], ['age'])
py.iplot(u_fig, filename='Basic Distplot')


i_fig = fffig.create_distplot([age_india[age_india['Age'] > 0]['Age']], ['age'])
py.iplot(i_fig, filename='Basic Distplot')


cnt_srs = df['FormalEducation'].value_counts()

trace = go.Scatter(
    x=cnt_srs.index,
    y=cnt_srs.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        color = cnt_srs.values,
        colorscale='Portland',
        showscale=True
    ),
)

layout = go.Layout(
    title=''
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="formaleducation")





cnt_srs = df['Tenure'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color = cnt_srs.values,
        colorscale='Jet',
        showscale=True
    ),
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="tenure")
plt.show()




temp_series = df['CareerSwitcher'].value_counts()
py.iplot(fig, filename="CareerSwitcher")





plt.subplots(figsize=(6,10))
learn=df['LearningPlatformSelect'].str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('Accent_r',15))
plt.show()





df['CoursePlatformSelect'] = df['CoursePlatformSelect'].astype('str').apply(lambda x: x.split(','))
t = df.apply(lambda x: pd.Series(x['CoursePlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)
t = t[t != 'nan'].value_counts()
sns.barplot(y=t.index, x=t)





plt.subplots(figsize=(10,10))
hard=df['HardwarePersonalProjectsSelect'].str.split(',')
hardware=[]
for i in hard.dropna():
    hardware.extend(i)
pd.Series(hardware).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gist_earth',10))
plt.show()




df['LanguageRecommendationSelect'].value_counts()[:2].plot.bar()
plt.show()




data = df[(df['CurrentJobTitleSelect'].notnull()) & ((df['LanguageRecommendationSelect'] == 'Python') | (df['LanguageRecommendationSelect'] == 'R'))]
plt.figure(figsize=(8,10))
sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)




data = df['MLToolNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)




data = df['MLMethodNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)




f,ax=plt.subplots(1,2,figsize=(22,8))
sns.countplot(y=df['ProveKnowledgeSelect'],order=df['ProveKnowledgeSelect'].value_counts().index,ax=ax[0],palette=sns.color_palette('inferno',15))
ax[0].set_title('How to prove my knowledge')
sns.countplot(df['JobSkillImportanceKaggleRanking'],ax=ax[1])
ax[1].set_title('')
plt.show()



plt.subplots(figsize=(10,8))
df.groupby(['EmployerSearchMethod'])['Age'].count().sort_values(ascending=True).plot.barh(width=0.8,color=sns.color_palette('winter',10))
plt.show()




data = df[(df['CurrentJobTitleSelect'].notnull()) & ((df['LanguageRecommendationSelect'] == 'Python') | (df['LanguageRecommendationSelect'] == 'R'))]
plt.figure(figsize=(8, 10))
sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)



plt.figure(figsize=(8,8))
sns.countplot(y='TimeSpentStudying', data=df, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))


temp_series = df['DataScienceIdentitySelect'].value_counts()
py.iplot(fig, filename="DataScienceIdentitySelect")



plt.figure(figsize=(12,8))
sns.countplot(x='EmploymentStatus', hue='DataScienceIdentitySelect', data=df)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('EmploymentStatus', fontsize=12)
plt.xticks(rotation=30)
plt.show()




