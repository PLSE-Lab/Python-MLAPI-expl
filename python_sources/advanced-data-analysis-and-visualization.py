#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


suicide = pd.read_csv("../input/master.csv")
suicide.head()


# In[ ]:


suicide.columns


# In[ ]:


len(suicide)


# In[ ]:


suicide.age.unique()


# ## Number of suicides committed in each country

# In[ ]:


type(suicide['suicides_no'][0])


# In[ ]:


suicide_in_countries = suicide.groupby('country')


# In[ ]:


suicide_in_countries


# In[ ]:


country_suicide_series = suicide_in_countries['suicides_no'].sum()


# In[ ]:


country_suicide_series.head()


# In[ ]:


# plt.figure(figsize=(10,30))
# sb.set_style('dark')
# sb.barplot(country_suicide_series.values,country_suicide_series.index)
# plt.show()


# In[ ]:


trace1 = go.Bar(
    y=country_suicide_series.values,
    x=country_suicide_series.index,
)

data = [trace1]
layout = go.Layout(
    title="Number of suicide committed in each country",
    xaxis={
        'title':"Countries",
    },
    yaxis={
        'title':"Number of suicide",
    }
)
figure=go.Figure(data=data,layout=layout)
py.offline.iplot(figure)


# In[ ]:


suicide.year.unique()


# ## Number of suicides genderwise

# In[ ]:


(suicide['suicides_no'][suicide['sex']=='male']).sum()


# In[ ]:


genderwise_suicide = suicide.pivot_table(index='sex' , aggfunc='sum')


# In[ ]:


genderwise_suicide['suicides_no']


# In[ ]:


suicide['suicides_no'].sum()


# In[ ]:


sb.barplot(genderwise_suicide.index , genderwise_suicide.suicides_no)
sb.set_style('white')


# ## Number of suicides genderwise in each country

# In[ ]:


country_wise_gender_suicide_df = pd.DataFrame({
    'country' : suicide.country,
    'sex' : suicide.sex,
    'suicides_no' : suicide.suicides_no
})


# In[ ]:


country_wise_gender_suicide = country_wise_gender_suicide_df.pivot_table(index='country' , columns='sex' , aggfunc='sum')


# In[ ]:


country_wise_gender_suicide.iloc[0]


# In[ ]:


# country_wise_gender_suicide.plot.bar(stacked = True , figsize=(30,10) , cmap='coolwarm')


# In[ ]:


country_wise_gender_suicide.columns = country_wise_gender_suicide.columns.droplevel()


# In[ ]:


country_wise_gender_suicide.female.head()


# In[ ]:


trace1 = go.Bar(
    x=country_wise_gender_suicide.index,
    y=country_wise_gender_suicide.female,
    name='Number of female suicides'
)
trace2 = go.Bar(
    x=country_wise_gender_suicide.index,
    y=country_wise_gender_suicide.male,
    name='Number of female suicides'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title="Number of suicides genderwise in each country",
    xaxis={
        'title':"Countries",
    },
    yaxis={
        'title':"Number of suicide",
    }
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


#  ## Suicides committed by various age groups

# In[ ]:


suicide.age.unique()


# In[ ]:


agewise_suicides = suicide.pivot_table(index='age' , aggfunc='sum')


# In[ ]:


agewise_suicides.columns


# In[ ]:


agewise_suicides_df = pd.DataFrame({
    'suicides_no' : agewise_suicides['suicides_no']
},index = agewise_suicides.index)


# In[ ]:


agewise_suicides_df


# In[ ]:


agewise_suicides_df.sort_values(by='suicides_no' , ascending=False).plot.bar()


# ## Suicides committed by various age groups in male and female

# In[ ]:


gender_agewise_suicide_df = pd.DataFrame({
    'sex' :suicide.sex,
    'age' : suicide.age,
    'suicide_no' : suicide.suicides_no
})


# In[ ]:


gender_agewise_suicide = gender_agewise_suicide_df.pivot_table(index='age' , columns='sex' , aggfunc='sum')


# In[ ]:


gender_agewise_suicide.iloc[0]


# In[ ]:


#gender_agewise_suicide.plot.pie(subplots=True , figsize=(30,10))


# In[ ]:


gender_agewise_suicide.columns = gender_agewise_suicide.columns.droplevel()


# In[ ]:


gender_agewise_suicide.female


# In[ ]:


fig = {
    'data' : [
        {
           'labels' : gender_agewise_suicide.index,
           'values' : gender_agewise_suicide.female,
            'type': 'pie',
            'name': 'Female suicides no',
            'domain': {'x': [0, .48],
                       'y': [0, .49]}
        },
        {
           'labels' : gender_agewise_suicide.index,
           'values' : gender_agewise_suicide.male,
            'type': 'pie',
            'name': 'Male suicides no',
            'domain': {'x': [.52, 1],
                       'y': [0, .49]}
        }  
    ],
    'layout': {'title': 'Suicides committed by various age groups in female and male'}
}

py.offline.iplot(fig)


# ## Suicides committed by various age groups in male and female in each country

# In[ ]:


country_gender_agewise_suicide_df = pd.DataFrame({
    'country' : suicide.country,
    'sex' : suicide.sex,
    'age' : suicide.age,
    'suicide_no' : suicide.suicides_no
})


# In[ ]:


country_gender_agewise_suicide_df.head()


# In[ ]:


country_gender_agewise_suicide = country_gender_agewise_suicide_df.pivot_table(index=['country' , 'age'] , columns='sex' , aggfunc='sum')


# In[ ]:


country_gender_agewise_suicide.loc['Albania']


# In[ ]:


country_gender_agewise_suicide.sample(10).plot.barh(figsize=(20,10))


# ## Population and suicide rate in yearly basics

# In[ ]:


suicide.year.unique()


# In[ ]:


yearly_suicide = suicide.groupby('year').mean()


# In[ ]:


yearly_suicide.head()


# In[ ]:


yearly_population = pd.DataFrame({
    'population' : yearly_suicide.population
})
    
yearly_suicide_no = pd.DataFrame({
    'suicide_no' : yearly_suicide.suicides_no
})


# In[ ]:


yearly_population.head()


# In[ ]:


# yearly_population.plot.line(color='red')
# yearly_suicide_no.plot.line()


# In[ ]:


trace0 = go.Scatter(
    x = yearly_population.index,
    y = yearly_population.population,
    mode = 'lines+markers',
)
layout1 = go.Layout(
    title="Population rate yearly basics",
    xaxis={
        'title':"Years",
    },
    yaxis={
        'title':"Number of population",
    }
)

data1 = [trace0]
figure=go.Figure(data=data1,layout=layout1)
py.offline.iplot(figure)

trace1 = go.Scatter(
    x = yearly_suicide_no.index,
    y = yearly_suicide_no.suicide_no,
    mode = 'lines+markers',
)

layout2 = go.Layout(
    title="Suicide rate yearly basics",
    xaxis={
        'title':"Years",
    },
    yaxis={
        'title':"Suicide rate",
    }
)

data2 = [trace1]
figure1=go.Figure(data=data2,layout=layout2)
py.offline.iplot(figure1)


# ## Merge population and suicide no with yearlywise

# In[ ]:


merge_pop_suicide = pd.merge(yearly_population , yearly_suicide_no , on='year')


# In[ ]:


merge_pop_suicide.head()


# In[ ]:


merge_pop_suicide.plot.bar(stacked=True,figsize=(30,10))


# ## Suicide rate in male and female yearly

# In[ ]:


yearly_countrywise_gender_suicide_df = pd.DataFrame({
    "year" : suicide.year,
    "sex" : suicide.sex,
    "suicides_no" : suicide.suicides_no
})


# In[ ]:


yearly_countrywise_gender_suicide = yearly_countrywise_gender_suicide_df.pivot_table(index='year', columns='sex',aggfunc='mean')


# In[ ]:


yearly_countrywise_gender_suicide.head()


# In[ ]:


x = yearly_countrywise_gender_suicide.index


# In[ ]:


yearly_countrywise_gender_suicide.columns


# In[ ]:


yearly_countrywise_gender_suicide.columns = ['female' , 'male']


# In[ ]:


yearly_countrywise_gender_suicide.head()


# In[ ]:


female = yearly_countrywise_gender_suicide['female']
male = yearly_countrywise_gender_suicide['male']


# In[ ]:


fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, female, color="blue", alpha=0.5 , label='Female suicide rate')
ax.plot(x, male, color="green", alpha=0.5 , label="Male suicide rate")
ax.set_xlabel('Years')
ax.set_ylabel('Avg. of suicide no')
ax.legend()


# ## Suicide rate over the period of time in each country
# 

# In[ ]:


country_yearwise_suicide_df  = pd.DataFrame({
    'country' : suicide.country,
    'year' : suicide.year,
    'suicides_no' : suicide.suicides_no
}) 


# In[ ]:


country_yearwise_suicide = country_yearwise_suicide_df.pivot_table(index='year' , columns='country' , aggfunc='mean')


# In[ ]:


country_yearwise_suicide.columns = country_yearwise_suicide.columns.droplevel()


# In[ ]:


country_yearwise_suicide.head()


# In[ ]:


country_yearwise_suicide.interpolate(axis=0 , inplace=True)


# In[ ]:


country_yearwise_suicide.fillna(method='bfill' , axis=0 , inplace=True)


# In[ ]:


country_yearwise_suicide.iloc[:,1].name


# In[ ]:


len(country_yearwise_suicide.columns)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,6))

l = len(country_yearwise_suicide.columns) 
for i in range(l - 90):
    
    ax.plot(country_yearwise_suicide.index, country_yearwise_suicide.iloc[:,i], alpha=0.5 , label=country_yearwise_suicide.iloc[:,i].name)

ax.legend()
ax.set_xlabel('Years')
ax.set_ylabel('Avg. of suicide no')


# In[ ]:


suicide.head()


# In[ ]:


suicide.generation.unique()


# ## GDP evalution of each country yearly basis

# In[ ]:


yearly_country_gdp_df = pd.DataFrame({
    'country' : suicide.country,
    'year' : suicide.year,
    'gdp' : suicide.iloc[:,9]
})


# In[ ]:


yearly_country_gdp_df = yearly_country_gdp_df.drop_duplicates()


# In[ ]:


yearly_country_gdp_df.head()


# In[ ]:


yearly_country_gdp_df = yearly_country_gdp_df.set_index('country')


# ## Evaluation  GDP of Albania country throughout the period

# In[ ]:


country_gdp = yearly_country_gdp_df[yearly_country_gdp_df.index == 'Albania']
country_gdp.head()


# In[ ]:


trace = go.Scatter(x=country_gdp.year , y=country_gdp.gdp , name=country_gdp.index[0],mode = 'lines+markers')
layout = go.Layout(
        title="Albania country's GDP evaluation",
        xaxis={
                'title':"Years",
              },
        yaxis={
                'title':"gdp rate",
              }
        )
data = [trace]
fig = go.Figure(data=data , layout=layout)
py.offline.iplot(fig)


# ## Generation evaluation

# In[ ]:



gen_sui = suicide.pivot_table('suicides_no', index='generation', aggfunc='sum')
x = gen_sui.index.values
y = gen_sui.values
y = y.reshape(6,)

fig, ax = plt.subplots(figsize=(10, 6))
explode = (0.1,0.1,0.1,0.5,0.1,0.1)
ax.pie(y, explode=explode, labels=x, autopct='%1.1f%%', shadow=True, startangle=0)
ax.axis('equal')
plt.show()


# ## Pairplot through age group

# In[ ]:


suide_pairplot = suicide.fillna(method='bfill')
suide_pairplot.head(2)


# In[ ]:


sb.pairplot(suide_pairplot , hue='age')


# In[ ]:


sb.distplot(suicide.iloc[:,6])
plt.show()

