#!/usr/bin/env python
# coding: utf-8

# ## Correcting under-reported COVID-19 case numbers
# 
# The COVID-19 virus has spread worldwide in a matter of a few month. Healthcare systems struggle to monitor and report current cases. Limited capabilities in testing result in difficult to guide policies and mitigate lack of preparation. Since severe cases, which more likely lead to fatal outcomes, are detected at a higher percentage than mild cases, the reported death rates are likely inflated in most countries. Such under-estimation can be attributed to under-sampling of infection cases and results in systematic death rate estimation biases.
# 
# The method proposed here utilizes a benchmark country (South Korea) and its reported death rates in combination with population demographics to correct the reported COVID-19 case numbers. By applying a correction, we predict that the number of cases is highly under-reported in most countries. In the case of China, it is estimated that more than 700.000 cases of COVID-19 actually occurred instead of the confirmed 80,932 cases as of 3/13/2020.
# 
# Manuscript:
# https://www.medrxiv.org/content/10.1101/2020.03.14.20036178v1

# This workflow utilizes population demographics and public COVID-19 case reports to estimate the true case progression workwide.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io
import requests

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Download and initialize datasets

# In[ ]:



confirmed_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
#recovered_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

url = confirmed_url
s = requests.get(url).content
cases = pd.read_csv(io.StringIO(s.decode('utf-8')))

url = deaths_url
s = requests.get(url).content
deaths = pd.read_csv(io.StringIO(s.decode('utf-8')))

#url = recovered_url
#s = requests.get(url).content
#recovered = pd.read_csv(io.StringIO(s.decode('utf-8')))


# Homogenize some country names, to make them compatible accross datasets

# In[ ]:


cases['Country/Region'] = cases['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
cases['Country/Region'] = cases['Country/Region'].replace('Taiwan*', 'Taiwan')
cases['Country/Region'] = cases['Country/Region'].replace('Criuse Ship', 'Diamond Princess')
cases['Country/Region'] = cases['Country/Region'].replace('Korea, South', 'South Korea')

deaths['Country/Region'] = deaths['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
deaths['Country/Region'] = deaths['Country/Region'].replace('Taiwan*', 'Taiwan')
deaths['Country/Region'] = deaths['Country/Region'].replace('Criuse Ship', 'Diamond Princess')
deaths['Country/Region'] = deaths['Country/Region'].replace('Korea, South', 'South Korea')

#recovered['Country/Region'] = recovered['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')
#recovered['Country/Region'] = recovered['Country/Region'].replace('Taiwan*', 'Taiwan')
#recovered['Country/Region'] = recovered['Country/Region'].replace('Criuse Ship', 'Diamond Princess')
#recovered['Country/Region'] = recovered['Country/Region'].replace('Korea, South', 'South Korea')

countries = cases.iloc[:,1].unique()
countries.sort()


# 1. Plot the case progression of reported COVID-19 cases by country and the respective deaths. These numbers are cummulative case counts. There are two scales for the y-axis. The left scale is for the case numbers and recovered cases. The right scale in green is the number of deaths. Only countries with more than 1000 cases are plotted.

# In[ ]:


counter = 0

plt.figure(figsize=(16,30))
plt.tight_layout(pad=3.0)

plt.rcParams.update({'font.size': 26})

for country in countries:
    ma = np.where(cases['Country/Region'] == country)[0]
    c_count = cases.iloc[ma, 4:].sum(axis=0)
    d_count = deaths.iloc[ma, 4:].sum(axis=0)
    #r_count = recovered.iloc[ma, 4:].sum(axis=0)
    
    if np.max(c_count) > 1000:
        sig_cases = np.where(c_count > 30)[0]
        counter = counter + 1
        ax1 = plt.subplot(10, 3, counter)
        res1, = plt.plot(c_count[sig_cases], 'ro-', linewidth=3, label="reported cases")
        #res2, = plt.plot(r_count[sig_cases], 'b^-', linewidth=3, label="recovered")
        plt.xticks(rotation=90)
        ax2 = ax1.twinx()
        res3, = ax2.plot(d_count[sig_cases], 'gs-', linewidth=3, label="deaths")
        
        plt.title(country)
        ax1.set_ylabel('cases', color="black")
        color = 'green'
        ax2.set_ylabel('deaths', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.legend(handles=[res1, res3])

plt.subplots_adjust(left=0, right=2, top=2, bottom=0.5, hspace=1)
plt.show()


# ### User input required
# Select two countries to be analyzed. 
# 
# **country1** = target country (as descripted in manuscript)
# 
# The target country case progression will be corrected using a benchmark country. Currently South Korea has the most accurate information available.
# 
# **country2** = benchmark country (as described in manuscript)
# 
# The benchmark country death-rate and population demographics are used to correct the case progression of **country1**

# In[ ]:


country1 = "Germany"
country2 = "South Korea"


# Plot the death-rate of **country1** and **country2** over time. The depicted time interval is limited to days where **country1** reported at least 4 deaths.

# In[ ]:


country = country1
ma = np.where(cases["Country/Region"] == country)[0]
country_count = cases.iloc[ma, 4:].sum(axis=0)
country_deaths = deaths.iloc[ma, 4:].sum(axis=0)
dr_c1 = country_deaths / country_count

d_cases = np.where(country_deaths > 3)[0]

country = country2
ma = np.where(cases["Country/Region"] == country)[0]
country_count = cases.iloc[ma, 4:].sum(axis=0)
country_deaths = deaths.iloc[ma, 4:].sum(axis=0)
dr_c2 = country_deaths / country_count
d_cases_c2 = np.where(country_deaths > 3)[0]

plt.figure(figsize=(10,6))
plt.rc('ytick', labelsize=26) 
plt.rc('xtick', labelsize=13)

res1, = plt.plot(dr_c1[d_cases]*100, 'b.-', label="Death rate ("+country1+")")
res2, = plt.plot(dr_c2[d_cases]*100, 'r.-', label="Death rate ("+country2+")")

plt.ylabel("death rate")
plt.xticks(rotation=90)
plt.legend(handles=[res1, res2])
plt.show()


# Collapse the line plots to show the distibution of death-rate as a barplot.

# In[ ]:


plt.figure(figsize=(5,6))
x = [dr_c1[d_cases]*100,
     dr_c2[d_cases_c2]*100]
plt.boxplot(x)
plt.xticks([1, 2], [country1, country2])
plt.show()


# Load population demographics. The data covers 2007 to 2019. Some countries may not be included in this time frame. 

# In[ ]:


demo = pd.read_csv("/kaggle/input/world-population-demographics-by-age-2019/world_demographics.csv")

demo[:1]

demo["Country or Area"] = demo["Country or Area"].replace("Viet Nam", "Vietnam")
demo["Country or Area"] = demo["Country or Area"].replace("United States of America", "US")
demo["Country or Area"] = demo["Country or Area"].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
demo["Country or Area"] = demo["Country or Area"].replace("Republic of Korea", "South Korea")
demo["Country or Area"] = demo["Country or Area"].replace("Venezuela (Bolivarian Republic of)", "Venezuela")
demo["Country or Area"] = demo["Country or Area"].replace('Iran (Islamic Republic of)', 'Iran')

duc = demo["Country or Area"].unique()
duc.sort()


# Aggregate demographic informaton for the countries. 

# In[ ]:


ma = np.where(demo["Country or Area"] == country1)[0]
demo_c1 = demo.iloc[ma,:]
max_year = np.where(demo_c1["Year"] == demo_c1["Year"].max())[0]
demo_c1 = demo_c1.iloc[max_year]

ma = np.where(demo["Country or Area"] == country2)[0]
demo_c2 = demo.iloc[ma,:]
max_year = np.where(demo_c2["Year"] == demo_c2["Year"].max())[0]
demo_c2 = demo_c2.iloc[max_year]

ll = list(range(0,120))
ll = [str(i) for i in ll]

ma = np.where(np.isin(demo_c1["Age"], np.array(ll)))[0]
demo_c1 = demo_c1.iloc[ma,:]

ma = np.where(np.isin(demo_c2["Age"], np.array(ll)))[0]
demo_c2 = demo_c2.iloc[ma,:]


# Plot the demographic distribution of the countries by year of birth.

# In[ ]:


p_c1 = demo_c1["Value"]
p_c2 = demo_c2["Value"]

y = list(range(0, 100, 1))

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[-(p_c1+p_c2).max(), (p_c1+p_c2).max()],
                       tickvals=[-1000000, -500000, 0, 500000, 1000000],
                       ticktext=["1M", "0.5M", "0", "0.5M", "1M"],
                       title='Number'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=y,
               x=-p_c1,
               orientation='h',
               name=country1,
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=y,
               x=p_c2,
               orientation='h',
               name=country2,
               text=-1 * p_c2.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]

fig = py.iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid')


# This information is published by The Center for Disease Control of South Korea. The data is based on a data release from 3/18/2020.

# In[ ]:


deathrate_age = list(range(0,120))

deathrate_age[0:30] = [0]*30
deathrate_age[30:40] = [0.0011]*10
deathrate_age[40:50] = [0.0009]*10
deathrate_age[50:60] = [0.0037]*10
deathrate_age[60:70] = [0.0151]*10
deathrate_age[70:80] = [0.0535]*10
deathrate_age[80:120] = [0.1084]*40


# The vulnerable vector is the number of expected fatal outcomes if everybody in the population is infected.

# In[ ]:


vulnerable_c1 = [a * b for a, b in zip(demo_c1["Value"], deathrate_age)]
vulnerable_c2 = [a * b for a, b in zip(demo_c2["Value"], deathrate_age)]


# Plot the demographic information for predicted fatalities.

# In[ ]:


p_c1 = np.array(vulnerable_c1)
p_c2 = np.array(vulnerable_c2)

y = list(range(0, 100, 1))

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[-max(list(p_c1)+list(p_c2)), max(list(p_c1)+list(p_c2))],
                       tickvals=[-50000, -25000, 0, 25000, 50000],
                       ticktext=["50k", "25k", "0", "25k", "50k"],
                       title='Number'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=y,
               x=-p_c1,
               orientation='h',
               name=country1,
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=y,
               x=p_c2,
               orientation='h',
               name=country2,
               text=-1 * p_c2.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]

py.iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid')


# Calculate the Vulnerability Factor.

# In[ ]:


v_c1 = sum(vulnerable_c1)/sum(demo_c1["Value"])
v_c2 = sum(vulnerable_c2)/sum(demo_c2["Value"])

exp_diff = v_c1/v_c2
print(exp_diff)


# Plot the number of adjusted cases vs the reported cases.

# In[ ]:


country = country1
ma = np.where(cases["Country/Region"] == country)[0]
country_count = cases.iloc[ma, 4:].sum(axis=0)
country_deaths = deaths.iloc[ma, 4:].sum(axis=0)
dr_c1 = country_deaths / country_count

d_cases = np.where(country_deaths > 0)[0]
print(country_deaths[d_cases])

country = country2
ma = np.where(cases["Country/Region"] == country)[0]
country_count = cases.iloc[ma, 4:].sum(axis=0)
country_deaths = deaths.iloc[ma, 4:].sum(axis=0)
dr_c2 = country_deaths / country_count

d_cases_c2 = np.where(country_deaths > 3)[0]
# instead of average death rate choose latest reported death rate from benchmark country
# 
#scaling_factor = dr_c1[d_cases] / ((exp_diff*sum(dr_c2[d_cases_c2])/len(d_cases_c2)))

scaling_factor = dr_c1[d_cases] / (exp_diff*dr_c2[d_cases_c2][-1])

country = country1
ma = np.where(cases["Country/Region"] == country)[0]

country_count = cases.iloc[ma, 4:].sum(axis=0)
country_deaths = deaths.iloc[ma, 4:].sum(axis=0)

potential_cases = country_count[d_cases] * scaling_factor

plt.figure(figsize=(10,6))
res1, = plt.plot(country_count[d_cases], 'bo-', label="Reported Cases ("+country1+")")
res2, = plt.plot(potential_cases, 'ro-', label="Adjusted Cases ("+country1+")")

plt.ylabel("cases")
plt.xticks(rotation=90)
plt.legend(handles=[res1, res2])
plt.show()


# Print the reported cases, adjusted cases, increase in adjustment (%)

# In[ ]:


print(country_count[d_cases][-1])
print(round(potential_cases[-1]))
print(round(100*round(potential_cases[-1])/country_count[d_cases][-1]))


# In[ ]:




