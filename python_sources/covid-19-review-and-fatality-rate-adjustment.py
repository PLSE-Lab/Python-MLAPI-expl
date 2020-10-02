#!/usr/bin/env python
# coding: utf-8

# Yet another COVID-19 data visualization notebook. It was mainly inspired by abhinand05's [project](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper).
# Due to the data being updated on daily basis, some of the comments below the charts could not be 100% accurate by the date you read, but I will try to update the notebook regularly.
# 
# # Main conclusions
# * Each country has its own testing policy, which can underestimate the number of new confirmed cases or the recoveries. Therefore, most of the country-wise comparisons might be inappropriate; country dynamics might be more useful
# * The data shows a very high number of deaths in Italy, but this could be result of many factors - aged population, pre-existing medical conditions (comorbidities) of infected people, high antibiotic resistance of the population (highest in the EU), as well as using too conservative methodology of classifying the victims of SARS-CoV-2
# * The data for the new confirmed cases in China since early February seems to be unreliable
# * The most popular approach to calculate the fatality rate as death cases divided by total confirmed cases is ***misleading and deflates the real rate***. It would have been correct only at the end of the epidemic period. There is a lag (assuming of about 7 days on average) between the case confirmation and the death, therefore the real fatality rate should compare ***today's death cases to the total confirmed from 7 days ago***.
# * Several countries from Middle East, South Africa and Far East have reported high recovery rates
# * There is some slight evidence about positive effect of imposing national quarantine, but this is something difficult to be measured (long length and high standard deviation of the incubation period means a lag between the quarantine date and the first evident effects; also it's more than clear that the measures are definitely helping to stop the spread of the disease)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input daty_chinaa files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "plotly"
import datetime

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates = ['ObservationDate'])
data.head()
data.info()
data.describe()


# In[ ]:


data['ObservationDate'] = data['ObservationDate'].apply(lambda x: x.date())
data = data.sort_values(by=['ObservationDate','Country/Region'])
data.rename(columns={'ObservationDate':'Date', 'Province/State':'Province', 'Country/Region':'Country', 'Deaths':'Death'}, inplace=True)
data['Active'] = data.Confirmed - data.Death - data.Recovered
data.head()


# In[ ]:


print(f"First entry: {data.Date.min()}")
print(f"Last entry: {data.Date.max()}")
print(f"Time length: {data.Date.max() - data.Date.min()}")


# # Dynamics of cases over time

# In[ ]:


features = ['Confirmed', 'Active', 'Death', 'Recovered']
totals = data.groupby('Date')[features].sum().reset_index()


# In[ ]:


fig = go.Figure().add_trace(go.Scatter(x=totals.Date, y=totals.Confirmed, mode='lines', name='Confirmed'))    .add_trace(go.Scatter(x=totals.Date, y=totals.Death, mode='lines', name='Deaths'))    .add_trace(go.Scatter(x=totals.Date, y=totals.Recovered, mode='lines', name='Recovered'))    .add_trace(go.Scatter(x=totals.Date, y=totals.Active, mode='lines', name='Active')).update_layout(title='Cases over time')
fig.show()


# In February, the number of active cases slowed the tempo and even began to decrease in the second half of the month, mainly due to the increasing number of recoveries in China.
# After 5th March, however, the disease spreads almost all over the world.
# 
# Although the spread of the desease is currently having momentum higher than ever, let's take a closer look at the recoveries and death cases dynamic.

# In[ ]:


data_nochina = data[data.Country!='Mainland China'].groupby('Date')[features].sum().reset_index()


# In[ ]:


go.Figure().add_scatter(x=data_nochina.Date, y=data_nochina.Death, mode='lines', name='Deaths', line=dict(color='black')).add_scatter(x=data_nochina.Date, y=data_nochina.Recovered, mode='lines', name='Recovered', line=dict(color='green')).update_layout(title='Deaths vs Recoveries (excl. China)', yaxis_type='log')


# As more than half of world's recoveries are registed in China, we will exclude it from the chart for better visibility. Although the recovered people are more than the death cases, they both share similar momentum; despite the difference in the figures, people are being healed as quickly as other people are dying. In the early February, the number of recovered people has been 20 times higher than the number of deaths; by the end of March this difference has been narrowed down to about 2.5 times (again, these figures don't include China). By mid-April this ratio is 3.3

# In[ ]:


latest = data.loc[data.Date == max(data['Date']),:]
latestByCountry = latest.groupby('Country')[features].sum().reset_index()


# ## Top 10 countries by confirmed cases, active cases, deaths and recoveries

# In[ ]:


for feature in features:
    px.bar(latestByCountry.sort_values(by=feature,ascending=True).tail(10).reset_index(), x=feature, y='Country',
           orientation='h', title='Top 10 '+feature+' cases')


# In[ ]:


data_noprovinces = data.groupby(['Country','Date'])[features].sum().reset_index()


# In[ ]:


for feature in features:
    px.line(data_noprovinces[data_noprovinces['Country'].isin(
                data_noprovinces.groupby('Country')[features].sum().reset_index().
                sort_values(by=feature,ascending=False).head(10).Country)],
        x='Date', y=feature, color='Country', title=feature + ' cases over time by country')


# Several conclusions:
# * The number of new cases in China has stalled since the end of February - the restrictions their government has made have had a good effect on the disease spread, or the data provided is totally unreliable
# * The death cases in Italy have really exploded since the beginning of March and have surpassed China's deaths in less than 3 weeks. Spain is the other country with a hugely increasing number of deaths, while the next could potentially be US and France.
# * The number of deaths in Iran and especially in Spain and France has alarmingly increased by mid-March, while in USA, Spain and Germany - in the 2nd half of March. Iran also has increasing number of cases, but the increase looks more balanced and steady, unlike some other exponential-looking curves for other countries.
# * On the other hand, South Korea seems to have the situation under control for the entire March.

# In[ ]:


data_noprovinces['Daily new'] = data_noprovinces.groupby('Country').Confirmed.diff()
data_noprovinces['Daily death'] = data_noprovinces.groupby('Country').Death.diff()
data_noprovinces['Daily recovered'] = data_noprovinces.groupby('Country').Recovered.diff()


# # Fatality Rate Adjustment

# ## Important notes when analyzing the fatality rate
# 
# The most popular formula to calculate the fatality rate is dividing the death cases by the total confirmed ones. However, this would be correct only when the epidemic period is ended. It doesn't take into account the current infected patients and the length of the whole process - from the infection, through the incubation period, then the first received symptoms and finally - the death. Therefore, **most of the official statistics about the fatality rate are incorrect as they naively deflate the real fatality rate**.
# 
# As it's believed that the detection tests are only effective after the incubation period, we can then assume that a case confirmation comes when the first symptoms appear. Therefore, we should take into account the time between the first symptom and the death, which varies greatly, but according to [this source](https://www.worldometers.info/coronavirus/coronavirus-death-rate/) is about 7 days on average. Therefore, to calculate the fatality rate, we will **take the latest death cases figures and divide them by the total confirmed cases from 7 days ago**. This will cover the period when the infection develops and we'll assume that the victims will not die in less than 7 days after their first symptoms.
# 
# Even this correction might not be enough - similar approach is suggested by the medical science journal [The Lancet](http://tiny.cc/7u78lz), where the suggested correction is even more conservative and goes up to 14 days. However, it is only a month past the first serious increase in number of cases globally, so it wouldn't allow us to analyze how the fatality rate changes over time for more than a few countries. Therefore, we'll use the 7-days approach, although not being the most precise one. Also, due to the high volatility in day-to-day confirmed/death/recovered cases, we will not use the daily values, but the cummulative figures.
# 
# 
# Another important note - the comparison of rate by country is difficult and sometimes not appropriate due to several reasons listed by the [Centre of evidence-based medicine](https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/):
# * The number of cases detected by testing will vary considerably by country;
# * The selection bias varies - there are countries where severe disease are preferentially tested, while other people won't be tested, hence they would stay out of the official statistics;
# * There may be delays between symptoms onset and deaths  which  can lead to underestimation of the CFR;
# * There may be factors that account for increased death rates such  as coinfection, more inadequate healthcare, patient demographics (i.e., older patients might be more prevalent in countries such as Italy);
# * There may be increased rates of smoking or comorbidities amongst the fatalities.
# * Differences in how deaths are attributed to Coronavirus: dying with the disease (association) is not the same as dying from the disease (causation).
# 
# Due to these factors, more appropriate would be to just focus on the volatility of rates for a single country and avoid comparing one country to another.

# In[ ]:


data_noprovinces['Death_rate'] = 100*data_noprovinces['Death'] / data_noprovinces['Confirmed']
data_noprovinces['Recovery_rate'] = 100*data_noprovinces['Recovered'] / data_noprovinces['Confirmed']
data_noprovinces['Active_rate'] = 100*data_noprovinces['Active'] / data_noprovinces['Confirmed']


# In[ ]:


from datetime import timedelta
from datetime import date

countries10plusdeaths = data_noprovinces.loc[
    data_noprovinces.Country.isin(
        data_noprovinces.loc[data_noprovinces.Death > 10, 'Country']) &  (data_noprovinces.Country != 'Diamond Princess'),'Country'].unique()
fatality_adj_list = []
for country in countries10plusdeaths:
    for i in reversed(range(21)):
        step = []
        step.append(country)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Date'].values)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Death'].replace(np.nan, 0).values) # was daily death
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Recovered'].replace(np.nan, 0).values) # was daily recovered
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=14+i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values) # was daily new
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=7+i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values)
        fatality_adj_list.append(step)


# In[ ]:


for i in range(len(fatality_adj_list)):
    # convert numpy datetime64 to pandas timestamp
    fatality_adj_list[i][1] = np.append(fatality_adj_list[i][1], date.today())
    fatality_adj_list[i][1] = pd.Timestamp(fatality_adj_list[i][1][0])
    # add 0 to all 4th element arrays, then get their first sub-element - this is a workaround to avoid the issue when a country didn't have data so far back
    for j in range(2,7):
        fatality_adj_list[i][j] = np.append(fatality_adj_list[i][j],0)
        fatality_adj_list[i][j] = fatality_adj_list[i][j][0]
        fatality_adj_list[i][j] = int(fatality_adj_list[i][j])

fatality_adj = pd.DataFrame(fatality_adj_list, columns=['Country', 'Date', 'Deaths', 'Recovered', 'Confirmed14daysago',
                                                        'Confirmed7daysago', 'ConfirmedToday'])
fatality_adj['Fatality Rate (Basic)'] = round(100*fatality_adj.Deaths / fatality_adj.ConfirmedToday, 2)
#fatality_adj['Fatality Rate (Adjusted) 14 days'] = round(100*fatality_adj.Deaths / fatality_adj.Confirmed14daysago, 2)
fatality_adj['Fatality Rate (Adjusted) 7 days'] = round(100*fatality_adj.Deaths / fatality_adj.Confirmed7daysago, 2).replace(np.nan,0).replace(np.inf,0)
fatality_adj['Recovery Rate (Adjusted) 7 days'] = round(100*fatality_adj.Recovered / fatality_adj.Confirmed7daysago, 2).replace(np.nan,0).replace(np.inf,0)
fatality_adj.Date = fatality_adj.Date.apply(lambda x: x.date())
fatality_adj = fatality_adj.sort_values(by=['Date','Country'])
fatality_adj.replace(np.nan,0,inplace=True)


# In[ ]:


latest_adj = fatality_adj[fatality_adj.Date == fatality_adj[fatality_adj.Country != 'Others'].Date.max()]
avg_fatality = round(100*latest_adj.Deaths.sum() / latest_adj.Confirmed7daysago.sum(),2)

print(f"The world's average fatality rate by yesterday is: {avg_fatality}%.")

temp = []
for i in range(10):
    temp.append(avg_fatality)
    
px.bar(latest_adj.sort_values(by='Fatality Rate (Adjusted) 7 days', ascending=False).head(10),
       x='Country', y='Fatality Rate (Adjusted) 7 days', color='Deaths', color_continuous_scale='Burg')\
.add_scatter(x=latest_adj.sort_values(by='Fatality Rate (Adjusted) 7 days', ascending=False).head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_fatality}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with highest fatality rate by yesterday')


# According to the data, about one in every 10 infected people in the world dies. This is really scary.
# 
# Althoug Italy hardly makes it in the top 10 countries with highest fatality rate nowadays, the rate is still higher than the average and, what's more important, the base is already very high.
# There are several reasons for the high number of death cases in Italy:
# * demographic factors - the Italian population is second oldest in the world (by proportion of people aged 65+) [[source](https://iiasa.ac.at/web/home/research/researchPrograms/WorldPopulation/PublicationsMediaCoverage/ModelsData/AgingDemDataSheet2018_web.pdf)]
# * Italy is the EU country with highest **antibiotic resistance** [[source](http://www.ansa.it/english/news/science_tecnology/2019/11/19/italy-top-in-eu-in-antibiotic-resistance_369e0123-0107-445e-8c17-f11932c9d27c.html)] [1]
# * Comorbidities - there was evidence of high proportion of infected people who had pre-existing medical conditions [[source](https://www.epicentro.iss.it/coronavirus/bollettino/Report-COVID-2019_17_marzo-v2.pdf)]
# * **The methodology of recording the fatality rate - all death cases in hospitals where there were people with confirmed COVID-19, have been recorded as fatalities caused by COVID-19** [[source](https://www.stuff.co.nz/national/health/coronavirus/120443722/coronavirus-is-covid19-really-the-cause-of-all-the-fatalities-in-italy)]
# * According to some specialists, only 12% of the deaths in Italy are really caused by the coronavirus [[source](https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/)]
# 
# Other countries with high fatality percentage and also a high number of total deaths are Spain, Iran and France, although they are some way off the figure in Italy.
# 
# *[1] I do believe that antibiotic resistance will become the main topic of discussions in the near future as it has the potential to be a real issue for the health of the humanity. Farmers are using enormous quantities of antibiotics in animal husbandry, which we are taking in our body with the food that we eat. People are taking antibiotics by their own decision when feeling ill, often not completing a full 5-days course. We all are getting antibiotics in small quantities when we don't really need them, so the bacterias are getting used to this and in some point they become antibiotic-resistant. There is a huge risk of reaching a moment when we'll really need antibiotics, but they won't be useful for us anymore.*

# In[ ]:


print(f"The world's average fatality rate by yesterday is: {avg_fatality}%.")

px.bar(latest_adj[(latest_adj.Deaths>0)].sort_values(by='Fatality Rate (Adjusted) 7 days').head(10),
       x='Country', y='Fatality Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Deaths', color_continuous_scale='Blugrn')\
.add_scatter(x=latest_adj[(latest_adj.Deaths>0)].sort_values(by='Fatality Rate (Adjusted) 7 days').head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_fatality}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with lowest fatality rate by yesterday')


# The chart above shows the top 10 countries that have registered more than 500 infected people and have the lowest fatality rate - for all of them the rate is below 0.5%.
# 
# Personally, I tend to trust this data more than the data for highest fatality rate, because:
# * Some countries might report exaggerated death numbers due to methodology of classifying a death as caused by coronavirus.
# * The number of confirmed cases in some countries might be much lower than the real number of infected people, due to testing policy (e.g. some countries test only people with severe symptoms, while others are performing large-scale testing). The number of confirmed cases is a denominator in the fatality rate formula, so when it is lower, it could overestimate the fatality rate.
# 
# Having said that, in my opinion the lowest fatality ratios are more reliable than the highest ones.

# In[ ]:


avg_recovery = round(100*latest_adj.Recovered.sum() / latest_adj.Confirmed7daysago.sum(),2)
print(f"The world's average recovery rate by yesterday is: {avg_recovery}%.")

temp = []
for i in range(10):
    temp.append(avg_recovery)
    
px.bar(latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days', ascending=False).head(10),
       x='Country', y='Recovery Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Recovered', color_continuous_scale='Blugrn')\
.add_scatter(x=latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days', ascending=False).head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_recovery}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with highest recovery rate by yesterday', yaxis_type='log')


# China reports a very high recovery rate of over 90%, which is impressive. It also seems that the cruise ship Diamond Princess also has a high recovery rate - but this might be an expected result as many of the passengers were ones of the first quarantined, sent to their home countries and looked after. A few other countries from the Far East - South Korea, Japan, Singapore and Hong Kong - also can be clustered, although not being on this chart every day (the chart changes on daily basis).
# Another interesting note - Iran, Egypt, France and Spain appear here sometimes, but they can be seen on the chart with the highest fatality rate, too.

# In[ ]:


print(f"The world's average recovery rate by yesterday is: {avg_recovery}%.")

px.bar(latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days').head(10),
       x='Country', y='Recovery Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Recovered', color_continuous_scale='Burg')\
.add_scatter(x=latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days').head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_recovery}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with lowest recovery rate by yesterday')


# Netherlands fails to report any significant number of healed people despite the 8k+ number of confirmed cases, but an important note is that these might depend on the willingness and resources to re-test already infected people. If there are many people quarantined at home, some of them may have been recovered, but not tested yet, and hence not included in the official recoveries data. Czech Republic and Israel are other examples of countries with a relatively low number of healed people reported, compared to the total number of infections.

# ## Rates over time

# In[ ]:


totals_adj = fatality_adj.groupby('Date')['Confirmed7daysago', 'ConfirmedToday', 'Deaths', 'Recovered'].sum()
totals_adj['Fatality rate'] = round(100*totals_adj.Deaths / totals_adj.Confirmed7daysago,2)
totals_adj['Fatality rate (WHO methodology)'] = round(100*totals_adj.Deaths / totals_adj.ConfirmedToday,2)
totals_adj['Recovery rate'] = round(100*totals_adj.Recovered / totals_adj.Confirmed7daysago,2)
totals_adj['Recovery rate (WHO methodology)'] = round(100*totals_adj.Recovered / totals_adj.ConfirmedToday,2)


# In[ ]:


median_fatality_adj = []
for i in range(len(totals_adj)):
    median_fatality_adj.append(totals_adj['Fatality rate'].median())

median_fatality_who = []
for i in range(len(totals_adj)):
    median_fatality_who.append(totals_adj['Fatality rate (WHO methodology)'].median())


# In[ ]:


go.Figure().add_scatter(x=totals_adj.index, y=totals_adj['Fatality rate'], name='Fatality rate (adj)', line_shape='spline', mode='lines').add_scatter(x=totals_adj.index, y=totals_adj['Fatality rate (WHO methodology)'], name='Fatality rate (WHO)', line_shape='spline', mode='lines').add_scatter(x=totals_adj.index, y=median_fatality_adj, line=dict(dash='dash', color='royalblue'), name='Median fatality rate (adj)').add_scatter(x=totals_adj.index, y=median_fatality_who, line=dict(dash='dash', color = 'firebrick'), name='Median fatality rate (WHO)').update_layout(title='Fatality rate over time - adjusted vs. WHO methodology')


# The chart perfectly illustrates how naive is the WHO methodology - when the increase of the deaths is slightly higher than the increase of confirmed cases for the same day, it shows only a little increase in the fatality rate - in fact, it has been incorrectly deflated by today's high number of new (daily) confirmed cases. Such methology is inappropriate during the epidemy - it could be used only when the disease has been eradicated and there are not any infected people left in the world. 
# 
# In reality, today's deaths should actually be attributed to the confirmed cases value at some past moment (in our assumption that's 7 days ago). Therefore, with the adjusted rate, we can easily spot a day when the death rate actually starts to increase. A horizontal line here would mean that people are dying with the same rate as new people have been infected a week ago. An increase in today's death rate has no relation to today's new confirmed cases.

# In[ ]:


median_recovery_adj = []
for i in range(len(totals_adj)):
    median_recovery_adj.append(totals_adj['Recovery rate'].median())

median_recovery_who = []
for i in range(len(totals_adj)):
    median_recovery_who.append(totals_adj['Recovery rate (WHO methodology)'].median())


# In[ ]:


go.Figure().add_scatter(x=totals_adj.index, y=totals_adj['Recovery rate'], name='Recovery rate (adj)', line_shape='spline', mode='lines', line=dict(color='green')).add_scatter(x=totals_adj.index, y=totals_adj['Recovery rate (WHO methodology)'], name='Recovery rate (WHO)', line_shape='spline', line=dict(color='yellow')).add_scatter(x=totals_adj.index, y=median_recovery_adj, name='Median recovery rate (adj)', line=dict(dash='dash', color='green')).add_scatter(x=totals_adj.index, y=median_recovery_who, name='Median recovery rate (WHO)', line=dict(dash='dash', color='yellow')).update_layout(title='Recovery rate over time - adjusted vs. WHO methodology')


# In[ ]:


top10_fatality_median = fatality_adj.groupby('Country')['Fatality Rate (Adjusted) 7 days'].median().sort_values(ascending=False).head(10).reset_index().Country
top10_recovery_median = fatality_adj.groupby('Country')['Recovery Rate (Adjusted) 7 days'].median().sort_values(ascending=False).head(10).reset_index().Country


# In[ ]:


fatality_adj.loc[fatality_adj['Fatality Rate (Adjusted) 7 days'] > 100, 'Fatality Rate (Adjusted) 7 days'] = 100
fatality_adj.loc[fatality_adj['Recovery Rate (Adjusted) 7 days'] > 100, 'Recovery Rate (Adjusted) 7 days'] = 100
fatality_adj.loc[fatality_adj['Fatality Rate (Adjusted) 7 days'] < 0, 'Fatality Rate (Adjusted) 7 days'] = 0
fatality_adj.loc[fatality_adj['Recovery Rate (Adjusted) 7 days'] < 0, 'Recovery Rate (Adjusted) 7 days'] = 0


# In[ ]:


rates = ['Fatality Rate (Adjusted) 7 days', 'Recovery Rate (Adjusted) 7 days']

for rate in rates:
    px.line(fatality_adj[fatality_adj['Country'].isin(top10_fatality_median)],
        x='Date', y=rate, color='Country', title = (rate[:14] .replace('Rate','rate') + 'over time by country (Top 10 median)'),
           line_shape='spline')


# These two charts show the rates over time, but they are more generous - include countries with 100 or more confirmed cases. Half of the countries from top 10 with highest mortality rate are from Middle East or North Africa.
# The same can be said about the countries with highest recovery rate. Many developed European countries are experiencing issues with high mortality rates, but could not get a good position on the recovery rate table yet. Middle East ,North Africa and Far East countries are generally doing good on the recovery part (this can be confirmed when having a look outside of top 10s, too).

# ## Maps

# In[ ]:


for feature in features:
    px.choropleth(latestByCountry, locations = 'Country', locationmode = 'country names', color=feature,
             color_continuous_scale = 'Portland', title = 'World map of ' + feature + ' cases',
             range_color = [1,1000 if feature!='Confirmed' else 2000])


# # New cases

# In[ ]:


data_noprovinces['Daily new pct'] = data_noprovinces.groupby('Country').Confirmed.pct_change()*100
data_noprovinces['Daily death pct'] = data_noprovinces.groupby('Country').Death.pct_change()*100
data_noprovinces['Daily recovered pct'] = data_noprovinces.groupby('Country').Recovered.pct_change()*100


# In[ ]:


px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily new', ascending=False).head(10),
       x='Country', y='Daily new', title="Last day's Top 10 by number of new confirmed cases", color='Confirmed', color_continuous_scale='Portland')\
.update_layout(legend_title='<b>Total Confirmed</b>')


# The chart above shows that there's a huge increase in the new confirmed cases in USA for the last day. The new cases in Italy are still too many, but they have been surpassed by Spain and Germany. From the countries with fewer cases there are increases for UK and Turkey, which are now becoming an issue.

# In[ ]:


px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily death', ascending=False).head(10),
       x='Country', y='Daily death', title="Last day's Top 10 by number of new death cases", color='Death', color_continuous_scale='matter')\
.update_layout(legend_title='Total Deaths>')


# Having so many deaths already and still leading by number of new deaths per day shows that something really worrying is happening in Italy,but now Spain has taken the first place for deaths. Despite the lag in the disease spread in the USA, the number of deaths there is increasing and soon may reach the levels of Italy and Spain. The deaths in UK for the last few days were more than in Iran.

# In[ ]:


px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily recovered', ascending=False).head(10),
       x='Country', y='Daily recovered', title="Last day's Top 10 by number of new recovered cases", color='Recovered', color_continuous_scale='RdBu')


# The high number of recovered people in top European countries and USA for the last day is good news, as well as that the recoveries in Switzerland are finally starting to grow (not only the new cases and the deaths).

# In[ ]:


top10conf = data_noprovinces.groupby('Country')['Confirmed'].sum().reset_index().sort_values(by='Confirmed',ascending=False).head(10).Country
top10deaths = data_noprovinces.groupby('Country')['Death'].sum().reset_index().sort_values(by='Death',ascending=False).head(10).Country
top10recov = data_noprovinces.groupby('Country')['Recovered'].sum().reset_index().sort_values(by='Recovered',ascending=False).head(10).Country
top10active = data_noprovinces.groupby('Country')['Active'].sum().reset_index().sort_values(by='Active',ascending=False).head(10).Country


# In[ ]:


px.line(data_noprovinces[data_noprovinces.Country.isin(top10conf)], x='Date', y='Daily new', color='Country', title='Daily new cases', line_shape='spline')


# Something very interesting can be seen here - **a big portion of China's confirmed cases (15.1k) were reported in a single day** (13th Feb, Thursday), while in the majority of dates around that date the new daly cases were between 2000-3000. This raises a big question over the reliability of the data reported by the Chinese government. This was followed by a instant decrease of cases, which can now see only a double-digit figures since 7th March. Could this be a result of the strict quarantine rules, or just unreliable statistics were provided? Could we spectate in the future another case pf multi-thousand figure in a single day?
# A good approach to dive deeper into this could be if we find results about the number of tested patients per day.
# 
# Apart from China, very similar are the trends of new cases per day in Italy, France, Germany, and, to some extent, France. 

# In[ ]:


px.line(data_noprovinces[data_noprovinces.Country.isin(top10deaths)], x='Date', y='Daily death', color='Country', title='Daily death cases', line_shape='spline')


# In[ ]:


px.line(data_noprovinces[data_noprovinces.Country.isin(top10recov)], x='Date', y='Daily recovered', color='Country', title='Daily recovered cases', line_shape='spline')


# Although China seems to be a standout performer on this chart, there are some positives seen for Italy, Iran, but most interestingly - France and South Korea. Despite the lower reported number of confirmed cases than for other countries, in the past day the absolute number of healed people in France has surpassed the same number in Italy by twice. This, along with the several rumours about different medicines with a positive impact on the infected people, is a portion of good news coming from France.
# 
# *Half-humour opinion: After putting some great pressure over the smaller countries' agriculture for decades, by getting them addicted to the EU funding programmes, could France redeem some guilt and save Europe from the COVID-19 disease?*
# 
# Update: By 27th March for several days four-digit figures of recovered people were registered in Germany and Spain
# 
# Update: By the early April, countries such as Germany and Spain have been the recent leaders in recovered cases. Also, US, Italy and Iran have reported a decent number of recovered people.

# In[ ]:


last7days = data_noprovinces.sort_values(by=['Date','Country'], ascending=True).tail(len(data_noprovinces.Country.unique())*7)


# In[ ]:


top10highConf7days = last7days[last7days.Confirmed > 500].groupby('Country')['Daily new pct'].median().reset_index().replace(np.inf,0).sort_values(by='Daily new pct', ascending=False).head(10).Country
top10highDeath7days = last7days[last7days.Death > 50].groupby('Country')['Daily death pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily death pct', ascending=False).head(10).Country
top10highRecov7days = last7days[last7days.Recovered > 50].groupby('Country')['Daily recovered pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily recovered pct', ascending=False).head(10).Country


# In[ ]:


px.line(data_noprovinces.loc[(data_noprovinces.Country.isin(top10highConf7days)) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))], x='Date', y='Daily new', color='Country',
        title='Top 10 countries with highest average daily growth of Confirmed new cases for the last 7 days', line_shape='spline')\
.update_layout(yaxis_type='log')

px.bar(last7days[last7days.Confirmed > 500].groupby('Country')['Daily new pct'].median().reset_index().replace(np.inf,0).sort_values(by='Daily new pct', ascending=False).head(10),
      x='Country', y='Daily new pct', title='Top 10 countries with highest average daily growth of Confirmed new cases for the last 7 days')


# The chart above represents the top 10 countries that are experiencing the highest average daily growth of new confirmed cases for the last week. The countries with less than 500 confirmed cases are excluded from the chart.
# 
# As of 24th of March, the situation in the USA seems to be totally out of control, while other countries with a smaller base values, such as Turkey, are experiencing a huge increase of cases during the last week.

# In[ ]:


px.line(data_noprovinces[(data_noprovinces.Country.isin(top10highDeath7days)) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))],
        x='Date', y='Daily death pct', color='Country', title='Top 10 countries with highest median daily growth of new Death cases for the last 7 days', line_shape='spline')#\
#.update_layout(yaxis_type='log')

px.bar(last7days[last7days.Death > 50].groupby('Country')['Daily death pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily death pct', ascending=False).head(10),
      x='Country', y='Daily death pct', title='Top 10 countries with highest median daily growth of new Death cases for the last 7 days')


# All of the countries on the charts above have a reason to worry the newly reported deaths are increasing quickly there. Countries with less than 50 deaths are excluded.
# 
# As of 24th March, eight out of top 10 countries with highest recent daily growth of death cases are from Western Europe, with 7 of them having rate higher than 25% and the other one being Italy with "only" 15%, but with a huge base compared to the other countries.

# In[ ]:


px.line(data_noprovinces[data_noprovinces.Country.isin(top10highRecov7days) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))],
        x='Date', y='Daily recovered pct', color='Country', title='Top 10 countries with highest median daily growth of Recoveries for the last 7 days', line_shape='spline')
px.bar(last7days[last7days.Recovered > 50].groupby('Country')['Daily recovered pct'].mean().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily recovered pct', ascending=False).head(10),
       x='Country', y='Daily recovered pct', title='Top 10 countries with highest median daily growth of Recoveries for the last 7 days')


# Finally, some positive news - a huge increase in the recovered patients in Belgium and France for the last week (as of 24th March) - daily average of about 60%! As stated earlier, it would be no surprise if the mass healing starts from France and spreads all over the world!
# 
# It's good to see Belgium not only on the deaths chart, but on this one as well. I might be just speculating, but the geographic closeness between France and Belgium could be the reason for these results.
# 
# Unfortunately, the good news stop here. The other countries are far, far behind.
# 
# Iceland, however, as being the third one here, is worth mentioning that they claim to have the highest tests per capita ratio, as a result of a mass testing programme ([link](https://www.government.is/news/article/2020/03/15/Large-scale-testing-of-general-population-in-Iceland-underway/?fbclid=IwAR0SWmnub7BxfuqSAbIBWVUH2e_XnasxcrsqsUax_EqO3nc7sBlWXqex4kg)).

# # Tests per country

# In[ ]:


tests_input = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
tests_input.head()
tests_input.describe()


# In[ ]:


tests = tests_input[tests_input.tests.isna() == False]
tests = tests.loc[:,['country','pop','tests','density','medianage', 'urbanpop', 'gatheringlimit', 'hospibed', 'smokers', 'lung', 'healthexp', 'fertility']]
tests = tests.rename(columns={'country': 'Country', 'pop': 'Population', 'density': 'Density', 'medianage': 'Median Age', 'tests': 'Tests'})


# In[ ]:


continents = pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding='latin-1')
continents.head()
continents = continents.loc[:,['country','continent','sub_region']]
continents = continents.rename(columns={'country':'Country', 'continent':'Continent', 'sub_region':'Region'})


# In[ ]:


tests_merged = pd.merge(latestByCountry, tests, on='Country')
tests_merged.Population = tests_merged.Population.apply(lambda x: int(x.replace(',','')))
tests_merged = pd.merge(tests_merged, continents, on='Country', how='left')


# In[ ]:


tests_merged[tests_merged.Continent.isna()]
tests_merged.loc[tests_merged.Country=='Russia','Continent'] = 'Europe'
tests_merged.loc[tests_merged.Country=='Russia','Region'] = 'Eastern Europe'
tests_merged.loc[tests_merged.Country=='US','Continent'] = 'Americas'
tests_merged.loc[tests_merged.Country=='US','Region'] = 'Northern America'
tests_merged.loc[tests_merged.Country=='Vietnam','Continent'] = 'Asia'
tests_merged.loc[tests_merged.Country=='Vietnam','Region'] = 'South-Eastern Asia'


# In[ ]:


tests_merged['Tests1m'] = round(1000000*tests_merged.Tests/tests_merged.Population,2)
tests_merged['Confirmed1m'] = round(1000000*tests_merged.Confirmed/tests_merged.Population,2)
tests_merged['Deaths1m'] = round(1000000*tests_merged.Death/tests_merged.Population,2)
tests_merged['Recovered1m'] = round(1000000*tests_merged.Recovered/tests_merged.Population,2)


# In[ ]:


corrmatrix=tests_merged.corr()

go.Figure(data=go.Heatmap(z=corrmatrix, x=corrmatrix.index, y=corrmatrix.columns)).update_layout(title='Tests Heatmap')


# There is some fair amount of correlation between the total number of tests and the number of deaths (0.52), which is higher than the correlation between the number of tests and number of confirmed cases (0.38) or recoveries (0.46).
# Some weak correlation can be noticed between the number of deaths (and recoveries) and the population median age. Interestingly, there's no correlation between country density and number of confirmed cases or deaths.

# In[ ]:


px.scatter(x=tests_merged.Tests1m, y=tests_merged.Confirmed1m, size=tests_merged['Median Age'], text=tests_merged.Country, color=tests_merged['Continent']).update_traces(textposition='top center', textfont_size=10).update_layout(title='Tests conducted vs Confirmed cases by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Confirmed cases per 1 million population')


# The high number of tests per 1 million people in Iceland has identified a high number of infections too, but that's not always the case - there is some correlation, but it's not as strong at it was initially thought. There are some countries from the Middle East who had the highest tests/million rate in the world, but have lower confirmed/million rate than many Western European countries. Also, there are counries like Belarus, who had 100 times more tests per million than Brazil, but have identified population-proportionally the same number of infections.

# In[ ]:


px.scatter(x=tests_merged.Tests1m, y=tests_merged.Deaths1m, size=tests_merged['Median Age']-22, text=tests_merged.Country, color=tests_merged['Continent']).update_traces(textposition='top center', textfont_size=10).update_layout(title='Tests conducted vs Deaths by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Deaths per 1 million population')


# In[ ]:


px.scatter(x=tests_merged.Tests1m, y=tests_merged.Recovered1m, size=tests_merged['Median Age']-22, text=tests_merged.Country, color=tests_merged['Continent']).update_traces(textposition='top center', textfont_size=10).update_layout(title='Tests conducted vs Recoveries by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Recoveries per 1 million population')


# As said earlier, the strongest correlation (although not absolutely strong) for the number of test cases is with the number of recovered people. This makes some sense - if the availability of tests helps the countries to identify the infected people earlier, so a proper care could be imposed at the earlier disease stages.

# # Quarantine effect

# To measure the quarantine effect, we will plot the cummulative number of confirmed cases before and after imposing a quarantine in some selected countries.

# In[ ]:


restrictions = tests_input.loc[tests_input.quarantine.isna() == False,['country','quarantine','schools', 'publicplace', 'gathering', 'nonessential']]
restrictions = restrictions.groupby('country').first().reset_index()


# In[ ]:


def to_date(x):
    converttodate = datetime.date(int(x.split('/')[2]), int(x.split('/')[0]), int(x.split('/')[1]))
    return converttodate

restrictions.quarantine = restrictions.quarantine.apply(lambda x: to_date(x))


# In[ ]:


restrictions.groupby(['quarantine','country']).count()


# In[ ]:


go.Figure().add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Confirmed'], mode='lines', name='Italy before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Confirmed'], mode='lines', name='Italy after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Confirmed'], mode='lines', name='Germany before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Confirmed'], mode='lines', name='Germany after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Confirmed'], mode='lines', name='France before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Confirmed'], mode='lines', name='France after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Confirmed'], mode='lines', name='Spain before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Confirmed'], mode='lines', name='Spain after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Confirmed'], mode='lines', name='Belgium before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Confirmed'], mode='lines', name='Belgium after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Iran before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Iran after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Bulgaria before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Bulgaria after')\
.update_layout(title='Confirmed cases before and after imposing national quarantine', yaxis_type='log')


# The quarantine effect can hardly be estimated due to the long incubation period.
# 
# As shown above, the imposing of quarantine had no immediate impact on the number of confirmed cases. In some countries (Spain, France, Belgium) there was an increase in the growth of the confirmed cases, but this can be easily explained - the long incubation period (up to 2 weeks) means that the cases that were confirmed shortly after the quarantine date have actually been infected at some point before that date. Therefore, the impact of the quarantine should become evident about 2 weeks later - and indeed, it can be seen on the logarithmic scale above that the lines for most of the countries are not as "straight" as they were before and are very slowly starting to decrease - meaning that the exponential increase has been slightly interrupted.
