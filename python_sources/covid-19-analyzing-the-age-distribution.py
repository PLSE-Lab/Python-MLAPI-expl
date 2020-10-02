#!/usr/bin/env python
# coding: utf-8

# Coronavirus 2019 (COVID-19) is an infectuous desease first identified in December 2019 in Wuhan, China.
# The World Health Organisation (WHO) recognized it as a pandemic on March 11, 2020. 
# 
# There has been numerous discussions as to the percentages of infected in different age groups. In particular there is a clame that the [20-29 year-olds could be leading carreers](http://twitter.com/DrEricDing/status/1239041092978343937). 
# The aim of this notebook to look into this question of number of positive cases by age group. 
# 
# 

# Import libraries

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go

py.init_notebook_mode()
pd.plotting.register_matplotlib_converters()


# # Population coverage by tests 
# 
# First, let's look at data on the number of confirmed cases v.s. number of tested. This will also allow us to select some "similar" cases that we can compare. 

# We make use of data collected by https://ourworldindata.org/

# In[ ]:


tests_vs_confirmed= pd.read_csv("../input/covid19-tested-vs-confirmed/tests-vs-confirmed-cases-covid-19-per-million.csv")

tests_vs_confirmed=tests_vs_confirmed.dropna()

tests_vs_confirmed=tests_vs_confirmed.rename(columns={'Total COVID-19 tests per million people':'tested',
                                  'Total confirmed cases of COVID-19 per million people':'confirmed'})


# In[ ]:


fig1 = px.scatter(tests_vs_confirmed, x="tested", y="confirmed", hover_data=['Entity'])

fig1.update_layout(xaxis_type="log", yaxis_type="log", 
                   title_text='Number of people tested per million vs number of positive per million')

fig1.show()


# On this log-log plot, where individual points correspond to countries we see that there is a correlation between the number of tested people and the number of positive cases: i.e. the more people per million were tested - the more positive cases were found.

# We can fit this dependance with a simple linear regressor, or a more complicated RANSAC Regressor that is not perturbed by outliers. 

# In[ ]:



from sklearn.linear_model import RidgeCV,RANSACRegressor
from sklearn.metrics import median_absolute_error, r2_score

X=np.array(tests_vs_confirmed['tested'].apply(np.log1p))
y=np.array(tests_vs_confirmed['confirmed'].apply(np.log1p))



regr = RidgeCV()
regr.fit(X.reshape(-1, 1), y)

X_line=np.linspace(1.0, 10.0, num=10)

y_line = regr.predict(X_line.reshape(-1, 1))

regr1 =  RANSACRegressor(random_state=42,residual_threshold=2.)



#regr1 = HuberRegressor()
regr1.fit(X.reshape(-1, 1), y)



y_line1 = regr1.predict(X_line.reshape(-1, 1))


# In[ ]:


print("R2:{} ".format(regr.score(X.reshape(-1, 1),y)))


# In[ ]:


print("R2:{} ".format(regr1.score(X[regr1.inlier_mask_].reshape(-1,1), y[regr1.inlier_mask_])))


# The RANSAC seems to perform better and we only have a couple of strong outliers:

# In[ ]:


tests_vs_confirmed[ np.logical_not(regr1.inlier_mask_)]


# In[ ]:


fig1.add_trace(go.Scatter(
        x=np.exp(X_line),
        y=np.exp(y_line1),
        mode="lines",
        line=go.scatter.Line(color="red"),
        showlegend=False))
fig1.show()


# Let us select the countries with a high number of tested people per million and a high number of positives per million. These are South Korea, Norway, Italy and Iceland. It turns out that for all these countries there is also some data on the age distribution of positive cases.

# # South Korea
# 
# First let's import data on South Korea. We will need the data on the total number of people in different age groups. For this we use World Bank Data.

# In[ ]:


populations = pd.read_csv("../input/global-population-estimates/data.csv")


# In[ ]:


South_Korea_data=populations[populations['Country Code']=="KOR"]
South_Korea_data.head()


# Function to transform the data from World Bank codes to age groups.

# In[ ]:


def WorldBank_to_df(data):

    ages=range(0,80,5)

    code={}

    for c in data['Series Code'].values:
        code[c]=-1

    for a in ages:
        code["SP.POP.{:02d}{:02d}.MA".format(a,a+4)]=a
        code["SP.POP.{:02d}{:02d}.FE".format(a,a+4)]=a
    code["SP.POP.{:02d}UP.MA".format(80)]=80
    code["SP.POP.{:02d}UP.FE".format(80)]=80
    
    population_data=data[['Series Code','2020 [YR2020]']].copy()
    population_data=population_data.reindex()
    population_data['age']=population_data['Series Code'].apply(lambda x: code[x])
    population_data['sex']=population_data['Series Code'].str[-2:]
    population_data=population_data.drop(population_data[population_data['age']==-1].index)
    population_data=population_data.drop(columns='Series Code')
    population_data=population_data.rename(columns={"2020 [YR2020]" : "population"})
    
    return population_data
    
    


# In[ ]:


South_Korea_population_data=WorldBank_to_df(South_Korea_data)
South_Korea_population_data.head()


# Function to change the age groups from multiples of 5 (World Bank) into multiples of 10 (As reported by countries)

# In[ ]:


def population_data_to_age10(data):
    
    pop_data=data.copy()
    
    pop_data['age10']=pop_data['age'].apply(lambda x: (x//10)*10)
    pop_data=pop_data[['population','age10']].groupby(['age10']).sum()
    pop_total=pop_data.sum()
    pop_data['fraction']=pop_data['population'].apply(lambda x: 100*x/pop_total)

    
    return pop_data


# In[ ]:


South_Korea_pop_data=population_data_to_age10(South_Korea_population_data)


# Import data on the distribution of the number of cases per age group as a function of time. We use only the latest time-stamp.

# In[ ]:


timeage=pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv', index_col="date")
timeage.head()


# In[ ]:


South_Korea_latest_timeage=timeage[timeage.index.max()==timeage.index][['age','confirmed','deceased']]
South_Korea_latest_timeage['age']=South_Korea_latest_timeage['age'].apply(lambda x: int(x[:-1]))
South_Korea_latest_timeage=South_Korea_latest_timeage.set_index('age')

South_Korea_age_stat=pd.concat([South_Korea_pop_data,South_Korea_latest_timeage],axis=1)

South_Korea_age_stat['confirmed_per_million']=1000000*South_Korea_age_stat['confirmed']/South_Korea_age_stat['population']
South_Korea_age_stat['confirmed_percentage']=100*South_Korea_age_stat['confirmed']/South_Korea_age_stat['confirmed'].sum(axis=0)

South_Korea_age_stat.to_csv('South_Korea_final.csv')
South_Korea_age_stat


# # Iceland

# Apply the same operations for Iceland.

# In[ ]:


TimeAge2=pd.read_csv('../input/covid19-confirmed-cases-by-country-and-age/COVID-19_Age.csv', index_col="Date")


# In[ ]:


def latest_age_data(df,country):
    tmp=df[df['Country']==country].copy()
    tmp_timeage=tmp[tmp.index.max()==tmp.index][['Age_start','Positive']]
    tmp_timeage=tmp_timeage.rename(columns={'Positive':'confirmed','Age_start':'Age'})
    tmp_timeage=tmp_timeage.set_index('Age')
    if 90 in tmp_timeage.index:
        tmp_timeage.loc[80]=tmp_timeage.loc[80]+tmp_timeage.loc[90]
        tmp_timeage.drop(90,inplace=True)
    if 100 in tmp_timeage.index:
        tmp_timeage.loc[80]=tmp_timeage.loc[80]+tmp_timeage.loc[100]
        tmp_timeage.drop(100,inplace=True)
    return tmp_timeage
    


# In[ ]:


def get_age_stat(country,country_code,populations,TimeAge2):
    tmp_data=populations[populations['Country Code']==country_code]
    tmp_age_stat=pd.concat([population_data_to_age10(WorldBank_to_df(tmp_data)),
                            latest_age_data(TimeAge2,country)],axis=1)
    tmp_age_stat['confirmed_per_million']=1000000*tmp_age_stat['confirmed']/tmp_age_stat['population']
    tmp_age_stat['confirmed_percentage']=100*tmp_age_stat['confirmed']/tmp_age_stat['confirmed'].sum(axis=0)
    tmp_age_stat.to_csv(country+'_final.csv')
    
    return tmp_age_stat 
    
    


# In[ ]:


Iceland_age_stat=get_age_stat("Iceland","ISL",populations,TimeAge2)
Iceland_age_stat


# # Norway
# 
# Apply same operations for Norway.

# In[ ]:


Norway_age_stat=get_age_stat("Norway","NOR",populations,TimeAge2)
Norway_age_stat


# # Italy
# 
# Apply the same operations for Italy. 

# In[ ]:


Italy_age_stat=get_age_stat("Italy","ITA",populations,TimeAge2)
Italy_age_stat


# # Danmark 
# Apply the same operations for Danmark.

# In[ ]:


Danmark_age_stat=get_age_stat("Danmark","DNK",populations,TimeAge2)
Danmark_age_stat


# # Switzerland
# 
# Apply the same operations for Switzerland

# In[ ]:


Switzerland_age_stat=get_age_stat("Switzerland","CHE",populations,TimeAge2)
Switzerland_age_stat


# # Comparison

# Percentage of cases per age group

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Norway', x=Norway_age_stat.index, y=Norway_age_stat['confirmed_percentage']),
    go.Bar(name='Iceland', x=Iceland_age_stat.index, y=Iceland_age_stat['confirmed_percentage']),
    go.Bar(name='South Korea', x=South_Korea_age_stat.index, y=South_Korea_age_stat['confirmed_percentage']),
    go.Bar(name='Italy', x=Italy_age_stat.index, y=Italy_age_stat['confirmed_percentage']),
    go.Bar(name='Danmark', x=Danmark_age_stat.index, y=Danmark_age_stat['confirmed_percentage']),
    go.Bar(name='Switzerland', x=Switzerland_age_stat.index, y=Switzerland_age_stat['confirmed_percentage'])
])

fig.update_layout(barmode='group',title_text='Confirmed by age group',
                 xaxis_title="Age group",
                 yaxis_title="Percentage of cases")

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 10, 20, 30, 40, 50, 60 , 70, 80],
        ticktext = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    )
)


fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = [0, 5, 10, 15, 20, 25],
        ticktext = ['0%', '5%', '10%', '15%', '20%', '25%']
    )
)

fig.show()


# What can be somewhat more informative, is the distribution that takes into account the demographics i.e. information on the total number of people in a given age group.

# In[ ]:




fig = go.Figure(data=[
    go.Bar(name='Norway', x=Norway_age_stat.index, y=Norway_age_stat['confirmed_per_million']),
    go.Bar(name='Iceland', x=Iceland_age_stat.index, y=Iceland_age_stat['confirmed_per_million']),
    go.Bar(name='South Korea', x=South_Korea_age_stat.index, y=South_Korea_age_stat['confirmed_per_million']),
    go.Bar(name='Italy', x=Italy_age_stat.index, y=Italy_age_stat['confirmed_per_million']),
    go.Bar(name='Danmark', x=Danmark_age_stat.index, y=Danmark_age_stat['confirmed_per_million']),
    go.Bar(name='Switzerland', x=Switzerland_age_stat.index, y=Switzerland_age_stat['confirmed_per_million'])
])

fig.update_layout(barmode='group',title_text='Confirmed cases per million people by age group',
                 xaxis_title="Age group",
                 yaxis_title="Number of cases per million")

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 10, 20, 30, 40, 50, 60 , 70, 80],
        ticktext = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    )
)

fig.show()



# We find that Iceland has the highest number of detected positive cases per million, then come Norway and Danmark, and only after that South Korea and Italy. 
# 
# We also see the shapes of the age distributions for Danmark, Iceland and Norway look quite alike: there is a mode around 40-49 and 50-59 respectively, and the rest is lower. However, Norway and Danmark have a second peak for 80+, that Iceland doesn't have. 
# 
# The data for Italy seems to have a mode at 80. This could be due to the fact that there, mostly people with severe symptoms were tested, whereas in the other cases the coverage was much more varied.
# 
# Finally, the data for South Korea seems close to Norway, if we forget the peak for ages 20-29. Yet the peak is there an obviously requires further investigation to understand its origin. 
