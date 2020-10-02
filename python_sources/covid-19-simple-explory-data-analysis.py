#!/usr/bin/env python
# coding: utf-8

# ## Import Data

# #### first of all we import the covid_19 data and we prepare the packages that we are going to use later

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np 
import pandas as pd 
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# import train data and understand the data
df_train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
df_train.head()


# In[ ]:


df_train.sample(6)


# In[ ]:


# Number of rows and columns
df_train.shape


# In[ ]:


# Columns names
df_train.columns


# In[ ]:


# Type of columns
df_train.dtypes


# In[ ]:


# Do the same thing on the test data
df_test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


df_test.columns


# In[ ]:


# Names of the countries
df_train["Country_Region"].unique()


# In[ ]:


len(df_train["Country_Region"].unique())


# In[ ]:


df_train["Country_Region"].value_counts()


# ### From the first interaction with the data, we can affirm that it includes 187 countries for the past 78 days. We can note that there are some countries that have more precise visualization for each State such as US and France.

# ## Explory Data Analysis 

# ### Confirmed Covid-19 by country

# In[ ]:


fig_reg = px.bar(df_train[df_train['Date']=="2020-04-08"], x='Country_Region', y='ConfirmedCases')

fig_reg.show()


# In[ ]:


df = df_train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum().groupby(['Country_Region','Province_State']).max().sort_values().groupby(['Country_Region']).sum().sort_values(ascending=False)
top_conf_count = pd.DataFrame(df)
top_conf_count1 = pd.DataFrame(df.head(10))
fig_reg = px.bar(top_conf_count1,x=top_conf_count1.index, y='ConfirmedCases',color='ConfirmedCases')
fig_reg.update_layout(
    title="Confirmed Cases by Country",
    xaxis_title=" Countries",
    yaxis_title="numbre of Confirmed Cases ",
    )
fig_reg.show()


# In[ ]:


top_conf_count.head(10)


# ## From the last graphs and visualisation, we can say that US and the European continent are the most country affected by Covid-19. Meanwhile, China,where the virus began,become at rank 6 by the number of confirmed cases   

# ### Fatalities Covid-19 by country

# In[ ]:


df_d = df_train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['Fatalities'].sum().groupby(['Country_Region','Province_State']).max().sort_values().groupby(['Country_Region']).sum().sort_values(ascending=False)
top_death_count = pd.DataFrame(df_d)
top_death_count


# In[ ]:


top_death_count1 = pd.DataFrame(df_d.head(10))
fig_reg_fat = px.bar(top_death_count1,x=top_death_count1.index, y='Fatalities',color='Fatalities')
fig_reg_fat.update_layout(
    title="Fatalities by Country",
    xaxis_title=" Countries",
    yaxis_title="numbre of Fatalities ",
    )
fig_reg_fat.show()


# From the above graph, we can say that Italy and Spain has until now the most Fatalities of Covid-19. we can also notice that the number of fatalities in US has increased exponentially from tha last week

# In[ ]:


# Ordrening the countries by number of fatalities
top_count = pd.concat([top_conf_count , top_death_count],axis=1)
top_count = top_count.sort_values(['ConfirmedCases'],ascending=False)[:10]
top_count


# ### Confirmed Cases and Fatalities Covid-19 by country

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Bar(name='ConfirmedCases',x=top_count.index, y=top_count['ConfirmedCases']),
    go.Bar(name='Fatalities',x=top_count.index, y=top_count['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='group',title="Confirmed Cases and Fatalities by Country",
    xaxis_title=" Countries",
    yaxis_title="number of Confirmed Cases and Fatalities ",)
fig.show()


# ### Focus on each country case

# ## Tunisia Case

# In[ ]:


# Visualize tunisia dataframe
df_train[df_train["Country_Region"]=="Tunisia"]


# In[ ]:


fig_tun_fatal = px.line(df_train[df_train["Country_Region"]=="Tunisia"], x="Date", y="Fatalities", title='Tunisia Covid-19 Fatalities')
fig.update_layout(barmode='group',
    xaxis_title=" Date ",
    yaxis_title=" Fatalities ",)
fig_tun_fatal.show()


# In[ ]:


fig_tun_confirmed = px.line(df_train[df_train["Country_Region"]=="Tunisia"], x="Date", y="ConfirmedCases", title='Tunisia Covid-19 confirmed cases')
fig.update_layout(
    xaxis_title=" Date ",
    yaxis_title=" Confirmed Cases",)
fig_tun_confirmed.show()


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="Tunisia"]["ConfirmedCases"],
    x=df_train[df_train["Country_Region"]=="Tunisia"]["Date"],
    name = 'ConfirmedCases', 
    connectgaps=True 
))
fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="Tunisia"]["Fatalities"],
    x=df_train[df_train["Country_Region"]=="Tunisia"]["Date"],
    name='Fatalities',
))
fig.update_layout(title=' ConfirmedCases & Fatalities Covid-19 in Tunisia', xaxis_title=" Date ",yaxis_title=" Confirmed Cases & Fatalities",)
fig.show()


# ## United states Case

# In[ ]:


df_train[df_train["Country_Region"]=="US"]


# In[ ]:


sort=df_train[df_train["Country_Region"]=="US"].sort_values(by=["ConfirmedCases"],ascending=False)[:400]
sort_fat=df_train[df_train["Country_Region"]=="US"].sort_values(by=["Fatalities"],ascending=False)[:400]


# In[ ]:


fig = px.line(sort, x="Date", y="ConfirmedCases",color='Province_State', title='US confirmed cases by state')
fig.update_layout( xaxis_title=" Date ",yaxis_title=" Confirmed Cases & Fatalities",)

fig.show()


# In[ ]:


fig_fat = px.line(sort_fat, x="Date", y="Fatalities",color='Province_State', title='US Fatalities cases by state')
fig.update_layout(xaxis_title=" Date ",yaxis_title="Fatalities",)
fig_fat.show()


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="US"].fillna('NA').groupby(['Date'])["ConfirmedCases"].sum(),
    x=df_train[df_train["Country_Region"]=="US"]["Date"],
    name = 'ConfirmedCases', 
    connectgaps=True 
))
fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="US"].fillna('NA').groupby(['Date'])['Fatalities'].sum(),
    x=df_train[df_train["Country_Region"]=="US"]["Date"],
    name='Fatalities',
))
fig.update_layout(title=' ConfirmedCases & Fatalities in USA')
fig.show()


# According to the graphs above, we can affirm that the state new york is the most afffected so far comparing by the other states and then we find New jersey

# ## Italy Case

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="Italy"]["ConfirmedCases"],
    x=df_train[df_train["Country_Region"]=="Italy"]["Date"],
    name = 'ConfirmedCases', 
    connectgaps=True
))
fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="Italy"]["Fatalities"],
    x=df_train[df_train["Country_Region"]=="Italy"]["Date"],
    name='Fatalities',
))
fig.update_layout(title=' ConfirmedCases & Fatalities in Italy')
fig.show()


# ## France Case

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="France"].fillna('NA').groupby(['Date'])["ConfirmedCases"].sum(),
    x=df_train[df_train["Country_Region"]=="France"]["Date"],
    name = 'ConfirmedCases', 
    connectgaps=True 
))
fig.add_trace(go.Scatter(
    y=df_train[df_train["Country_Region"]=="France"].fillna('NA').groupby(['Date'])['Fatalities'].sum(),
    x=df_train[df_train["Country_Region"]=="France"]["Date"],
    name='Fatalities',
))
fig.update_layout(title=' ConfirmedCases & Fatalities in France')
fig.show()


# In[ ]:


country_df = df_train.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
country_df.tail()


# In[ ]:


data = (
    df_train.groupby(["Country_Region","Date"])
    .agg({"ConfirmedCases": "sum", "Fatalities": "sum"})
    .reset_index()
)
test_data = (
    df_test.groupby(["Date", "Country_Region"])
    .last()
    .reset_index()[["Date", "Country_Region"]]
)


# In[ ]:


data["Date"] = pd.to_datetime(data.Date)
test_data["Date"] = pd.to_datetime(test_data.Date)


# In[ ]:


countries = data["Country_Region"].unique()
test_countries = test_data["Country_Region"].unique()


# In[ ]:


df_train1 = df_train.fillna('NA').groupby(['Country_Region','Date']).sum()
df_train1
df_train1.reset_index(inplace=True)
df_train1


# In[ ]:


import plotly.graph_objects as go

from plotly.offline import iplot


for i in range(1, len(countries)):
  
    _data = df_train1[df_train1["Country_Region"] == countries[i - 1]]
    trace1 = go.Scatter(
        x=_data.Date,
        y=_data.ConfirmedCases,
        name= "Confirmed Cases"
    )
    trace2 = go.Scatter(
        x=_data.Date,
        y=_data.Fatalities,
        name="Confirmed Fatalities"
        )
    data1 = [trace1, trace2]
    layout = go.Layout(title = countries[i - 1], xaxis = {'title':'Date'}, yaxis = {'title':'value'})
    fig = go.Figure(data=data1,layout=layout)
    iplot(fig)


#  ## Covid-19 spread over time

# In[ ]:


country_df['Date'] = country_df['Date'].apply(str)

fig = px.scatter_geo(country_df, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='ConfirmedCases', hover_name="Country_Region",
                     hover_data=['ConfirmedCases', 'Fatalities'],
                     range_color= [0, top_count['ConfirmedCases'].max()], 
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Confirmed cases spread Over Time', color_continuous_scale="portland" , size_max=80)
fig.show()


# In[ ]:





# # **Ridge**

# In[ ]:


def getColumnInfo(df):
    #n_province =  df['Province_State'].nunique()
    n_country  =  df['Country_Region'].nunique()
    n_days     =  df['Date'].nunique()
    start_date =  df['Date'].unique()[0]
    end_date   =  df['Date'].unique()[-1]
    return  n_country, n_days, start_date, end_date

n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(df_train1)
n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(df_test)


df_test = df_test.loc[df_test.Date > '2020-04-10']
overlap_days = n_test_days - df_test.Date.nunique()


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


countries_pred = 'US'

# Take the 1st day as 2020-02-23
df = df_train1.loc[df_train1.Date >= '2020-02-23']
n_days_europe = df.Date.nunique()

#for i in range(1, len(countries_pred)): 
df_country_train = df_train1[df_train1['Country_Region']==countries_pred ]
df_country_test = df_test[df_test['Country_Region']==countries_pred]  
df_country_train = df_country_train.reset_index()[df_country_train.reset_index().Date > '2020-02-22']

x_train = np.arange(1, n_days_europe+1).reshape((-1,1))
x_test  = (np.arange(1,n_test_days+1+overlap_days)).reshape((-1,1)) 

# ****************** Fatalities ****************************
y_train_f = df_country_train['Fatalities']
model_f = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_f = model_f.fit(x_train, y_train_f)
y_predict_f = model_f.predict(x_test) 

# ******************* Cases ******************************
y_train_c = df_country_train['ConfirmedCases'] 
model_c = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_c = model_c.fit(x_train, y_train_c)
y_predict_c = model_c.predict(x_test)

# ***************** Figure **************************
plt.rcParams.update({'font.size': 12})
fig,(ax0,ax1) = plt.subplots(2,1,figsize=(20, 20))



ax0.plot(x_test, y_predict_c,linewidth=2, label='predict_Cases_'+countries_pred)
ax0.plot(x_train, y_train_c, linewidth=2, color='r', linestyle='dotted', label='train_Cases_'+countries_pred)
ax0.set_title( " Predicted vs Confirmed Cases : " +countries_pred)
ax0.set_xlabel("Number of days")
ax0.set_ylabel("Confirmed Cases")
ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

ax1.plot(x_test, y_predict_f,linewidth=2, label='predict_Fatalities_'+countries_pred)
ax1.plot(x_train, y_train_f, linewidth=2, color='r', linestyle='dotted', label='train_Fatalities_'+countries_pred)
ax1.set_title("Predicted vs Confirmed Fatalities : " + countries_pred)
ax1.set_xlabel("Number of days")
ax1.set_ylabel("Fatalities")
ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))


# In[ ]:


countries_pred = 'Italy'

# Take the 1st day as 2020-02-23
df = df_train1.loc[df_train1.Date >= '2020-02-23']
n_days_europe = df.Date.nunique()

#for i in range(1, len(countries_pred)): 
df_country_train = df_train1[df_train1['Country_Region']==countries_pred ]
df_country_test = df_test[df_test['Country_Region']==countries_pred]  
df_country_train = df_country_train.reset_index()[df_country_train.reset_index().Date > '2020-02-22']

x_train = np.arange(1, n_days_europe+1).reshape((-1,1))
x_test  = (np.arange(1,n_test_days+1+overlap_days)).reshape((-1,1)) 

# ****************** Fatalities ****************************
y_train_f = df_country_train['Fatalities']
model_f = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_f = model_f.fit(x_train, y_train_f)
y_predict_f = model_f.predict(x_test) 

# ******************* Cases ******************************
y_train_c = df_country_train['ConfirmedCases'] 
model_c = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_c = model_c.fit(x_train, y_train_c)
y_predict_c = model_c.predict(x_test)

# ***************** Figure **************************
plt.rcParams.update({'font.size': 12})
fig,(ax0,ax1) = plt.subplots(2,1,figsize=(20, 20))



ax0.plot(x_test, y_predict_c,linewidth=2, label='predict_Cases_'+countries_pred)
ax0.plot(x_train, y_train_c, linewidth=2, color='r', linestyle='dotted', label='train_Cases_'+countries_pred)
ax0.set_title( " Predicted vs Confirmed Cases : " +countries_pred)
ax0.set_xlabel("Number of days")
ax0.set_ylabel("Confirmed Cases")
ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

ax1.plot(x_test, y_predict_f,linewidth=2, label='predict_Fatalities_'+countries_pred)
ax1.plot(x_train, y_train_f, linewidth=2, color='r', linestyle='dotted', label='train_Fatalities_'+countries_pred)
ax1.set_title("Predicted vs Confirmed Fatalities : " + countries_pred)
ax1.set_xlabel("Number of days")
ax1.set_ylabel("Fatalities")
ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))


# In[ ]:


countries_pred = 'Tunisia'

# Take the 1st day as 2020-02-23
df = df_train1.loc[df_train1.Date >= '2020-02-23']
n_days_europe = df.Date.nunique()

#for i in range(1, len(countries_pred)): 
df_country_train = df_train1[df_train1['Country_Region']==countries_pred ]
df_country_test = df_test[df_test['Country_Region']==countries_pred]  
df_country_train = df_country_train.reset_index()[df_country_train.reset_index().Date > '2020-02-22']

x_train = np.arange(1, n_days_europe+1).reshape((-1,1))
x_test  = (np.arange(1,n_test_days+1+overlap_days)).reshape((-1,1)) 

# ****************** Fatalities ****************************
y_train_f = df_country_train['Fatalities']
model_f = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_f = model_f.fit(x_train, y_train_f)
y_predict_f = model_f.predict(x_test) 

# ******************* Cases ******************************
y_train_c = df_country_train['ConfirmedCases'] 
model_c = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 
model_c = model_c.fit(x_train, y_train_c)
y_predict_c = model_c.predict(x_test)

# ***************** Figure **************************
plt.rcParams.update({'font.size': 12})
fig,(ax0,ax1) = plt.subplots(2,1,figsize=(20, 20))



ax0.plot(x_test, y_predict_c,linewidth=2, label='predict_Cases_'+countries_pred)
ax0.plot(x_train, y_train_c, linewidth=2, color='r', linestyle='dotted', label='train_Cases_'+countries_pred)
ax0.set_title( " Predicted vs Confirmed Cases : " +countries_pred)
ax0.set_xlabel("Number of days")
ax0.set_ylabel("Confirmed Cases")
ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

ax1.plot(x_test, y_predict_f,linewidth=2, label='predict_Fatalities_'+countries_pred)
ax1.plot(x_train, y_train_f, linewidth=2, color='r', linestyle='dotted', label='train_Fatalities_'+countries_pred)
ax1.set_title("Predicted vs Confirmed Fatalities : " + countries_pred)
ax1.set_xlabel("Number of days")
ax1.set_ylabel("Fatalities")
ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))


# In[ ]:




