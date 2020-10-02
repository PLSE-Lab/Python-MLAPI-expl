#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import git
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


if os.path.isdir('./exports'):
    pass
else:
    os.mkdir('exports')
    
if os.path.isdir('exports/daily-data-per-country/'):
    pass
else:
    os.mkdir('exports/daily-data-per-country/')
    
if os.path.isdir('img/'):
    pass
else:
    os.mkdir('img/')
    


# In[ ]:


"""
This module is for classes and methods that we will need.

We need 2 classes:
	- country
	- Timecountries_series
"""

import pandas as pd
import numpy as np

country_column = 'Country/Region'
drop_cols = ['Lat','Long','Province/State']

class TimeSeries:

	def __init__(self, data, country_name=None):

		# fix the columns with dates
		df = data.copy()
		df.drop(drop_cols, axis=1, inplace=True)
		cols = df.columns.to_list()	
		date_cols = [date.split()[0] for date in list(map(str,pd.to_datetime(cols[1:])))]
		cols[1:] = date_cols
		df.rename(dict(zip(df.columns.to_list(),cols)),axis=1, inplace=True)
		df.sort_values(by=country_column,inplace=True)
		df.index=df[country_column]
		df.drop(country_column,axis=1,inplace=True)

		# unique contries list ordered alphabetically.
		countries_list = list(df.index.unique())
		countries_list.sort()


		self.country_name = country_name

		# if the user throws a country_name name
		if country_name != None:

			# check if country name is a string or a list
			if type(country_name)==str:

				# verify if its a valid country
				if country_name in countries_list:

					self.country_name = country_name

					df = df[df.index==self.country_name].groupby(country_column).sum()
					df.replace({0:np.nan},inplace=True)
					df.dropna(axis=1,inplace=True)					

					self.data = df

				# if country not valid
				else:
					err = """
Please enter a valid country_name.

valid country names: {}
					""".format(countries_list)
					print(err)
					raise

			# If it is a list
			elif type(country_name)==list:

				# check if all are valid countries
				if all(i in countries_list for i in country_name):
					self.country_name = country_name

					# get all countries series of values
					countries_series = []
					for i in range(len(country_name)):
						countries_series.append(
							df[df.index==self.country_name[i]].groupby(country_column).sum()
							)

					# consolidate series in a df
					df = pd.concat(countries_series)
					df.replace({0:np.nan},inplace=True)
					df.dropna(axis=1,inplace=True,thresh=1)

					self.data = df

				else:
					err = """
Please enter a valid country_name.

valid country names: {}
					""".format(countries_list)
					print(err)
					raise

		else:
			self.data = df.groupby(country_column).sum()



	## Methods ##

	def get_data_frame(self, transpose=False):
		"""
		Gets data frame of Time Series.
		"""
		if transpose:
			df = self.data.copy()
			return df.transpose()

		else:
			df = self.data.copy()
			return df

	def get_last_update_date(self):
		"""
		Gets last update date of the time series
		"""
		df = self.data.copy()
		dates = pd.to_datetime(df.columns.to_list())
		max_date = str(dates.max()).split()[0]
		return max_date

	def get_last_values(self):
		"""
		"""
		df = self.data.copy()
		last_date = self.get_last_update_date()
		last_values = df[last_date]
		return last_values

	def get_diff(self,percentages=False):
		"""
		"""
		if percentages:
			df = self.data.copy()
			df = df.diff(axis=1)/df
			return df.dropna(axis=1,thresh=1)

		else:
			df = self.data.copy()
			return df.diff(axis=1).dropna(axis=1,thresh=1)


# In[ ]:


# read our data
confirmed_df =pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

labels = ['Confirmed','Deaths','Recovered']
dfs = [confirmed_df,deaths_df,recovered_df]
data_dict = dict(zip(labels,dfs))

# get last update date:
last_update = TimeSeries(confirmed_df).get_last_update_date()

## Actual Information per Country ##
actual_confirmed = TimeSeries(confirmed_df).get_last_values()
actual_deaths = TimeSeries(deaths_df).get_last_values()
actual_recovered = TimeSeries(recovered_df).get_last_values()

actual_data = pd.DataFrame({'Confirmed':actual_confirmed.values,
                            'Deaths':actual_deaths.values,
                            'Recovered':actual_recovered.values},index=actual_confirmed.index)


# In[ ]:


print("Last update: "+ str(last_update))


# # Welcome to this Coronavirus Global Visualizations and Information notebook.
# 
# The data source of this notebook is the [repository](https://github.com/CSSEGISandData/COVID-19) of the [Johns Hopkins University](https://www.jhu.edu/). I would like to thank them for making this data available and public.
# 
# 
# In this notebook we will be answering the following questions:
# 
# * **How is the current situation COVID-19 in the world? (numbers and visualizations)**
# 
# 
# * **How does the number of Confirmed/Deaths/Recovered cases behave through time? How does the Mortality Rate evolves through time?**
# 
# 
# * **Which are the countries with more confirmed/deaths/recovered cases and also which are the ones with the highest and lowest Mortality Rates?**
# 
# 
# * **How the daily new confirmed/deaths/recovered cases are behaving through time? Are we managing the situation? Does the daily increases are decresing or increasing over time? In which date there have been more daily confirmed/deaths/recovered cases?**
# 
# 
# * **How can we compare country by country growth?**
# 
# 
# * **What is the real form of the cumulated confirmed cases curve, exponential/logistic?**
# 
# 
# * **How can we visualize if a country is still in an exponential curve or if it has decreased significantly the new confirmed cases by now?**
# 
# 
# > **Note:** In this notebook I'll be using a self-made python module called **models**, for cleaning and displaying the Time Series of the data source for every country and make it easy to work with.

# # Current global and country by country information
# The countries are sorted by confirmed cases in a descending mode

# In[ ]:


def get_doubling_times(countries,kind='Confirmed'):
    data = TimeSeries(data_dict[kind]).data

    doubling_times = []
    for country in countries:
        last_week_cases = data.loc[country].values[-8]
        current_cases = data.loc[country].values[-1]

        if last_week_cases>0:
            ratio = current_cases/last_week_cases
            if ratio != 1:
                doubling_time = 7*np.log(2)/np.log(ratio)
                doubling_times.append(doubling_time)
            else:
                doubling_times.append(0)
        else:
            doubling_times.append(0)
        
    return doubling_times

actual_data['Mortality Rates']= actual_data['Deaths']/actual_data['Confirmed']
actual_data.to_csv('exports/daily-data-per-country/'+last_update+'.csv')

print('Global: ',last_update)
print('\tConfirmed: {:,}'.format(actual_data['Confirmed'].sum()))
print('\tDeaths: {:,}'.format(actual_data['Deaths'].sum()))
print('\tRecovered: {:,}'.format(actual_data['Recovered'].sum()))
print('\tMortality Rate: {:.2f}%'.format(100*actual_data['Deaths'].sum()/actual_data['Confirmed'].sum()))
print('--------------------------------')

actual_data.sort_values(by='Confirmed',ascending=False,inplace=True)

for i in range(len(actual_data)):
    print(str(i+1)+'.- '+actual_data.index.to_list()[i]+':')
    print('\tConfirmed: {:,}'.format(actual_data['Confirmed'][i]))
    dbt = get_doubling_times([actual_data.index.to_list()[i]],kind='Confirmed')[0]
    print('\tConfirmed Cases Doubling Time: {:.2f}'.format(dbt),' (Considering the last week increment)')
    print('\tDeaths: {:,}'.format(actual_data['Deaths'][i]))
    print('\tRecovered: {:,}'.format(actual_data['Recovered'][i]))
    print('\tMortality Rate: {:.2f}%'.format(actual_data['Mortality Rates'][i]*100))
    print('--------------------------------')


# # Global cumulated Confirmed/Deaths/Recovered and Actual Infected people evolution through the time

# In[ ]:


def plot_global_data():
    
    confirmed = TimeSeries(confirmed_df).data.groupby(lambda x: True).sum().transpose()
    deaths = TimeSeries(deaths_df).data.groupby(lambda x: True).sum().transpose()
    recovered = TimeSeries(recovered_df).data.groupby(lambda x: True).sum().transpose()
    actual_infected = confirmed - recovered - deaths
    df = pd.concat([confirmed,deaths,recovered,actual_infected],axis=1)
    df.columns = ['Confirmed','Deaths','Recovered','Actual Infected']
    
    fig = go.Figure()
    annotations=[]
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines+markers'))
        annotations.append(dict(x=df.index.to_list()[-1],y=df[col][-1],xref="x",
                                yref="y",text='{:,}'.format(int(df[col][-1])),
                                showarrow=True,arrowhead=7))
    
    fig.update_layout(title="<b>Global Cases</b>",xaxis_title="Date",yaxis_title=None,
                      font=dict(family="Arial, bold",size=14),annotations=annotations,
                      legend=dict(x=.01, y=.99))

    fig.show()
    
plot_global_data()


# # Global Mortality Rate
# 
# This plot shows the evolution of the **Global Mortality Rate** since January 22th, 2020.

# In[ ]:


def plot_line_mortality_rate():
    
    confirmed = TimeSeries(confirmed_df).data.groupby(lambda x: True).sum().transpose()
    deaths = TimeSeries(deaths_df).data.groupby(lambda x: True).sum().transpose()
    mort_rates = deaths/confirmed
    mort_rates.columns = ['Mortality Rate']
    n = len(mort_rates)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mort_rates.index,y=mort_rates['Mortality Rate'],
                            mode='lines+markers',name='Mortality Rate'))
    fig.add_trace(go.Scatter(x=mort_rates.index,
                             y=[mort_rates['Mortality Rate'].mean()]*len(mort_rates),
                            mode='lines',name='Mean'))
    annotations = []
    for i in range(len(mort_rates['Mortality Rate'])):
        if (i)%3==0:
            annotations.append(dict(x=mort_rates.index.to_list()[n-i-1],
                                    y=mort_rates['Mortality Rate'][n-i-1],
                                    xref="x",yref="y",
                                    text='{:.1f}%'.format(mort_rates['Mortality Rate'][n-1-i]*100),
                                    yshift=0))

    annotations.append(dict(x=mort_rates.index.to_list()[-1],
                            y=mort_rates['Mortality Rate'].mean(),
                            xref="x",yref="y",
                            text='{:.2f}%'.format(mort_rates['Mortality Rate'].mean()*100),
                            yshift=0))
    
    fig.update_layout(title="<b>Global Mortality Rate</b>",
                      xaxis_title="Date",yaxis_title=None,
                      annotations=annotations,
                      legend=dict(x=.01, y=.99))

    fig.show()

plot_line_mortality_rate()


# # Global COVID-19 Top Countries
# 
# Here are the bar plots, showing the countries with the highest number of the following:
# 
# * Confirmed cases
# * Deaths cases
# * Mortality rates
# * Recovered cases

# In[ ]:


def plot_bar_top(n, kind, ascending=False):
    
    def plot_mortality_rate():
        confirmed = TimeSeries(data_dict['Confirmed']).get_last_values()
        deaths = TimeSeries(data_dict['Deaths']).get_last_values()
        df = pd.DataFrame({'Confirmed':confirmed,'Deaths':deaths,'Mortality Rate':deaths/confirmed})
        df = df[df['Confirmed']>250]
        mortality_rates = df['Mortality Rate']
        top_values = mortality_rates.sort_values(ascending=ascending).head(n)
        top_values = top_values[::-1]
        
        title = "Top {} {}'s  by {}".format(n,kind,last_update)
        xaxis_title = '{}'.format(kind)
        
        fig = px.bar(x=top_values.values,y=top_values.index,color=top_values.index,
                     orientation='h',text=top_values.values)
        fig.update_layout(showlegend=False,title=title,xaxis_title=xaxis_title,
                         yaxis_title='Country')
        fig.update_traces(textposition='inside',texttemplate='%{x:.3p}')
        fig.show()
    
    if kind == 'Mortality Rate':
        plot_mortality_rate()
        
    else:
        data_source = data_dict[kind]
        data = TimeSeries(data_source).get_last_values()

        top_values = list(data.sort_values(ascending=ascending).head(n).values)
        others_sum = data.sort_values().head(len(data)-n).sum()
        top_values.append(others_sum)

        y = data.sort_values(ascending=ascending).head(n).index.to_list()
        y.append('Others')
        y=y[::-1]
        x = top_values[::-1]

        title = 'Top {} {} cases by {}'.format(n,kind,last_update)
        xaxis_title = '{} Cases'.format(kind)

        fig = px.bar(x=x,y=y,color=y,orientation='h',text=x)
        fig.update_layout(showlegend=False,title=title,xaxis_title=xaxis_title,
                         yaxis_title='Country')
        fig.update_traces(textposition='inside',texttemplate='%{x:,}')
        fig.show()

plot_bar_top(15,'Confirmed')
plot_bar_top(15,'Deaths')
plot_bar_top(15,'Mortality Rate')
plot_bar_top(15,'Recovered')


# # Countries with the lowest mortality rates
# 
# > **Note:** Taking in consideration only the countries that have 250 or more confirmed cases

# In[ ]:


# Countries with the lowest mortality rates
plot_bar_top(15,'Mortality Rate',ascending=True)


# # Global Daily New Cases
# 
# This are the global new cases through the time since January 22th, 2020. This are divided in 3:
# 
# * Confirmed
# * Deaths
# * Recovered

# In[ ]:


def get_diff(array):
    out_arr = []
    for i in range(len(array)):
        if i==0:
            out_arr.append(array[i])
        
        else:
            out_arr.append(array[i]-array[i-1])
    
    return np.array(out_arr)

def plot_daily_diff(kind='Confirmed', country=None):
    
    # if we want to plot global daily new cases or
    # daily new cases of a specific country
    if country==None:
        cases = TimeSeries(data_dict[kind]).data.groupby(lambda x: 'Global').sum().loc['Global'].values
        dates = TimeSeries(data_dict[kind]).data.columns.to_list()
        last_date = max(dates)
        title = 'Global '+kind + " Daily New Cases" + ' by '+ str(last_date)
    else:
        cases = TimeSeries(data_dict[kind],country).data.loc[country].values
        dates = TimeSeries(data_dict[kind],country).data.columns.to_list()
        last_date = max(dates)
        title = kind + ' Daily New Cases in ' + country + ' by '+ str(last_date)
    
    daily_cases = get_diff(cases)
    
    
    fig = px.bar(x=dates,y=daily_cases,color=daily_cases)
    
    annotations = [dict(x=dates[-1],y=daily_cases[-1],xref='x',yref='y',
                       text='{:,}'.format(int(daily_cases[-1])))]
    
    max_value = daily_cases.max()
    max_value_ix = np.where(daily_cases == daily_cases.max())[0][0]
    
    if max_value > daily_cases[-1]:
        annotations.append(dict(x=dates[max_value_ix],y=max_value,xref='x',yref='y',
                                text='Max: {:,}<br>{}'.format(int(max_value),dates[max_value_ix])))
    
    xaxis_title = 'Dates'
    yaxis_title = kind + ' new cases'
    fig.update_layout(title=title,annotations=annotations,xaxis_title=xaxis_title,
                     yaxis_title=yaxis_title)
    
    fig.show()

    
plot_daily_diff(kind='Confirmed')
plot_daily_diff(kind='Deaths')
plot_daily_diff(kind='Recovered')
plot_daily_diff(kind='Confirmed',country='Mexico')
plot_daily_diff(kind='Deaths',country='Mexico')
plot_daily_diff(kind='Confirmed',country='US')
plot_daily_diff(kind='Deaths',country='US')


# # Country by country cumulated confirmed cases trajectories
# Of the 10 countries with more confirmed cases in the world, and Mexico.

# In[ ]:


def plot_countries_trajectories(countries, kind='Confirmed'):
    
    data = TimeSeries(data_dict[kind]).get_data_frame(transpose=True)
    data = data[data>100]
    
    cases = []
    for country in countries:
        values = list(data[country].dropna().values)
        if len(values)>90:
            cases.append(values[:89])
        else:
            cases.append(values)
    
    cases_dict = dict(zip(countries,cases))
    
    fig = go.Figure()
    annot = []
    for country in countries:
        fig.add_trace(go.Scatter(y=cases_dict[country],name=country,mode='lines+markers',marker_size=4, opacity=0.8))
        annot.append(dict(x=cases_dict[country].index(cases_dict[country][-1]),
                          y=np.log10(cases_dict[country][-1]),text=country,
                          showarrow=False,yshift=8))
    
    every_1_days = [(2**x)*100 for x in range(1,9)]
    every_2_days = [(2**(x/2))*100 for x in range(1,23)]
    every_3_days = [(2**(x/3))*100 for x in range(1,40)]
    every_8_days = [(2**(x/8))*100 for x in range(1,50)]
    dbt = [every_1_days,every_2_days,every_3_days,every_8_days]
    dbt_names = ['Every Day','Every 2 Days','Every 3 Days', 'Every week']
    dbt_dict = dict(zip(dbt_names,dbt))
    
    fig.add_trace(go.Scatter(y=every_1_days,name='Every Day',
                             opacity=0.5,mode='lines',marker=dict(color='black')))
    fig.add_trace(go.Scatter(y=every_2_days,name='Every 2 Days',
                             opacity=0.5,mode='lines',marker=dict(color='black')))
    fig.add_trace(go.Scatter(y=every_3_days,name='Every 3 Days',
                             opacity=0.5,mode='lines',marker=dict(color='black')))
    fig.add_trace(go.Scatter(y=every_8_days,name='Every week',
                             opacity=0.5,mode='lines',marker=dict(color='black')))
    
    for label in dbt_names:
        text = 'Cases Double<br>'+label
        annot.append(dict(x=dbt_dict[label].index(dbt_dict[label][-1]),y=np.log10(dbt_dict[label][-1]),
                               text=text,showarrow=False,yshift=15))
    title='<b>Country by Country: How COVID-19 '+kind+' cases trajectories compare</b>'
    fig.update_layout(title=title,yaxis_type='log',xaxis_type='linear',annotations=annot,showlegend=False,
                     xaxis_title='Number of days since 100th case',yaxis_title='Number of cumulated confirmed cases')
    fig.show()

countries = actual_data.head(10).index.to_list()
countries.append('Mexico')
plot_countries_trajectories(countries=countries, kind='Confirmed')


# # How can we measure the velocity of the growth in every country? and how we can tell if a country is still in an exponential growth curve or not?
# 
# ### The answer to this question is:
# We plot **new confirmed cases** vs **cumulated confirmed cases** on a logaritmic scale.
# 
# 
# * **In a pandemic, how do the growth of confirmed cases behaves?**
# <br><br>
# Nowadays we hear a lot in the news or articles in the web that the confirmed cases of COVID-19 are growing exponentially in several countries and in all of the world. But, we can't have an exponential growth forever, just because of the simple fact that the number of people, that can be infected in all the world, is limited. That is, if in one point in time all of the humans get infected with the virus, there wont be more confirmed cases because there aren't more people to infect.
# <br><br>
# I'm not saying that we have to get to that point in order to don't have any more confirmed cases, we can also make a vaccine or the countries can implement social distancing policies in order to decrease the growth of the confirmed cases.
# <br><br>
# Having said that, I can introduce you the real curve that the confirmed cases growth follows: **Logistic Curve**.
# This logistic curve at the beginning is like an exponential curve but, at some point, there is an inflection point where the curve passes from exponential to a kind of logaritmic curve, as you can see in the image below.
# 
# ![Logistic Curve Example](https://xaktly.com/Images/Mathematics/LogisticDiffEq/LogisticExponentialComparison.png)
# 
# 
# * **Why logaritmic scale?**
# <br><br>
# In the first stages of the pandemic in every country, the growth of the cumulative is most likely exponential, also we have that ``log`` is the inverse function of the ``exp`` function, so if we apply ``log`` to the exponential growth we get a 'linear' curve.
# <br><br>
# This help us to **visualize and compare better the growths between the countries**, regardless of the big difference of number of cases between one country and another.
# 
# 
# * **Is my country still going through an exponential growth?**
# <br><br>
# When we plot the new confirmed cases vs. cumulated confirmed cases, if the growth is exponential, then in the logaritmic scale we will have a linear curve. So, **if a country has a linear curve that means that the growth is still exponential.**<br><br>
# In the other hand, **when a country curve is not a straight line** in this plot; then, we have that the country has gone through that inflection point when the curve passes from exponential to logaritmic. So, in this case, we have that the growth of confirmed cases is no longer an exponential growth.
# 
# 
# 
# > **Note:** In this plots, we take the weekly increments of confirmed cases, not the daily, in order to be a pessimist plot and not take in consideration small decrements of confirmed cases. This means, that a country line will be no longer exponential only if in the last week, the new confirmed cases had decreased significantly compared with the previous week.

# # New Confirmed Cases vs. Cumulated Confirmed Cases
# Of the 10 countries with more confirmed cases in the world, and Mexico.

# In[ ]:


def plot_trajectory(countries,days=8,cases_limit=50,log=True):
    """
    """
    fig = go.Figure()
    
    annotations=[]
    for country_name in countries:
        confirmed = TimeSeries(confirmed_df,country_name).get_data_frame(transpose=True)
        confirmed = confirmed[confirmed[country_name]>cases_limit][country_name].values[::-1]
        confirmed = confirmed[::days]
        confirmed = confirmed[::-1]
        deltas = get_diff(confirmed)

        fig.add_trace(go.Scatter(x=confirmed,y=deltas,name=country_name))
        
        if log:
            annotations.append(dict(x=np.log10(confirmed[-1]),
                                   y=np.log10(deltas[-1]),
                                   text=country_name,showarrow=False,
                                   xshift=10,yshift=10))
            
        
        else:
            annotations.append(dict(x=(confirmed[-1]),
                                   y=(deltas[-1]),
                                   text=country_name,showarrow=False,
                                   xshift=10,yshift=10))
    if log:
        subtitle ='On a logaritmic scale to normalize the exponential growth. Last Update: '+last_update
        xaxis_type="log"
        yaxis_type="log"
    else:
        subtitle ='On a linear scale. Last Update: '+last_update
        xaxis_type="linear"
        yaxis_type="linear"
    
    annotations.append(dict(xref='paper',yref='paper',x=-.04, y=+1.10,showarrow=False,
                            text =subtitle))

    fig.update_layout(xaxis_type=xaxis_type, yaxis_type=yaxis_type,annotations=annotations,showlegend=False,
                     title = '<b>Trayectory of COVID-19 Confirmed Cases</b>',
                     yaxis_title = 'New Confirmed Cases (in the last {} days)'.format(days),
                     xaxis_title = 'Total Confirmed Cases')
    fig.show()

    
top_confirmed_countries = actual_confirmed.sort_values(ascending=False).head(10).index.to_list()
top_confirmed_countries.append('Mexico')
plot_trajectory(top_confirmed_countries,days=8,cases_limit=50,log=True)


# # New Confirmed Cases vs. Cumulated Confirmed Cases
# Of the countries of **Latin America**

# In[ ]:


latin_america_countries = ['Brazil','Mexico','Argentina','Colombia','Chile','Ecuador','Peru',
                          'Dominican Republic','Costa Rica','Uruguay']

plot_trajectory(latin_america_countries,days=8,cases_limit=10,log=True)


# # Which countries are recently increasing faster in COVID-19 confirmed cases?
# 
# ### Doubling Times
# A **doubling time** is the time that takes to a country to double the confirmed cases given the increment of confirmed cases between two dates.
# 
# > **Note:** In this case I'll use the weekly increments to calculate the doubling times.

# In[ ]:


countries = TimeSeries(confirmed_df).data.index.to_list()
db_times = get_doubling_times(countries=countries,kind='Confirmed')
print('Days that take a country to double the confirmed COVID-19 cases:\n')
for i in range(len(countries)):
    print(str(countries[i])+":\n\t"+"{:.2f}".format(db_times[i]))
    print("-----------------")


# ## Countries that double confirmed cases faster (in the last week).
# This countries are increasing their confirmed cases at a faster rate.
# 
# This may be explained by the lack of a good health system or not implementing enough social distancing policies.
# 
# > **Note:** I take only in consideration those countries with more than 200 confirmed cases up to date.

# In[ ]:


countries = actual_data[actual_data['Confirmed']>200].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Confirmed')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=True).head(15)


# ## Countries that double confirmed cases slower (in the last week).
# This countries are the ones that actually have achieved to decrease the growth rate of confirmed cases.
# 
# This may be explained by the fact that theu have been implementing good policies in order to manage the pandemic situation. Or also, this countries have had the virus for a long period of time and is managing to improve the situation.
# 
# > **Note:** I take only in consideration those countries with more than 200 confirmed cases up to date.

# In[ ]:


countries = actual_data[actual_data['Confirmed']>200].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Confirmed')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=False).head(15)


# # Which countries are recently increasing faster in COVID-19 deaths cases?

# ## Countries that double deaths cases in less time than others. (In the last week)
# This are countries that actually are having deaths cases at a faster rate.
# 
# This may be explained because of the health system of the countries or if a country is managing to control the disease propagation. Also, if a country is starting to have more people infected of older age or with an underlying chronic illness, it can increase significantly it's growth rate of deaths.
# 
# > **Note:** Taking in consideration only countries that actually have registrated more than 50 deaths.

# In[ ]:


countries = actual_data[actual_data['Deaths']>50].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Deaths')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=True).head(15)


# ## Countries that double deaths cases in more time than others. (In the last week)
# This are countries that actually are having deaths cases at a slower rate.
# 
# This may be explained because of the health system of the countries or if a country is managing to control the disease propagation. Also if a country is starting to have more people infected of older age or with an underlying chronic illness, it can increase significantly it's growth rate of deaths.
# 
# > **Note:** Taking in consideration only countries that actually have registrated more than 50 deaths.

# In[ ]:


countries = actual_data[actual_data['Deaths']>50].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Deaths')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=False).head(15)


# # Which countries are recently increasing faster in COVID-19 recovered cases?

# ## Countries that double recovered cases in less time than others. (In the last week)
# This are the countries that actually are recovering people at a faster rate.
# 
# > **Note:** Taking in consideration only countries that actually have registrated more than 50 recovered cases.

# In[ ]:


countries = actual_data[actual_data['Recovered']>50].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Recovered')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=True).head(15)


# ## Countries that double recovered cases in more time than others. (In the last week)
# This are the countries that actually are recovering people slower rate.
# 
# 
# If a country is in this list, not necesarly is because the people there doesn't recover, it can be that they have recovered almost all the people so, that's why they recover less people now.
# 
# > **Note:** Taking in consideration only countries that actually have registrated more than 50 recovered cases.

# In[ ]:


countries = actual_data[actual_data['Recovered']>50].index.to_list()
db_times = get_doubling_times(countries=countries,kind='Recovered')
df = pd.DataFrame(data={'Doubling Time':db_times},index=countries)
df = df[df>0].dropna()
df.sort_values(by='Doubling Time',ascending=False).head(15)

