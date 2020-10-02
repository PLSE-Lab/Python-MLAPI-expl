#!/usr/bin/env python
# coding: utf-8

# ### Approach
# The covid data sets captured the spread and affected counts for all the countries. But the date on which COVID entered a country and when community spread started is different from country to country. Conducting an analysis with all the countries together will lead to confusion. So the main goal is to identify the countries in pandemic state and find out the common features which are main factors for covid spread. These factors are then used to identify country/group more prone for infection or in high risk state.
# 
# Note: Data available only till the end of March 2020

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[ ]:


# Importing necessary library
import pandas as pd
import numpy as np
import glob  
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()
import seaborn as sns


# In[ ]:


covid = pd.read_csv('../input/johns-hopkins/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv').rename(columns={'country_region':'Country'})
#info = pd.read_csv('countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})


# Along with johns hopkins data, country information like population, weather, density are added for better analysis

# In[ ]:


info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})
info = info[info.region.isnull()]
info['pop'] = info[~info['pop'].isnull()]['pop'].str.replace(',','').astype('int64')
info['totalcases'] = info[~info['totalcases'].isnull()]['totalcases'].str.replace(',','').astype('int64')
info['casediv1m'] = info[~info['casediv1m'].isnull()]['casediv1m'].str.replace(',','').astype('float')
# info['healthexp'] = info[~info['healthexp'].isnull()]['healthexp'].str.replace(',','').astype('float')
# info['gdp2019'] = info[~info['gdp2019'].isnull()]['gdp2019'].str.replace(',','').astype('float')


# In[ ]:


covid = covid.merge(info[['Country', 'pop']], how='left', on='Country')


# In[ ]:


covid.Country.nunique()


# In[ ]:


covid['confirmed_per_1000']= covid.confirmed*1000/covid['pop']


# In[ ]:


covid[covid.Country.isin(['Iran','France','Italy', 'Spain'])][['last_update', 'confirmed','Country']].pivot(index='last_update', columns='Country', values='confirmed').plot(figsize=(20,5))
plt.axhline(y=300, color='r', linestyle='dashed')
axes = plt.gca()
axes.set_ylim([0,5000])
plt.ylabel('Confirmed cases')
plt.xlabel('Date')


# 
# Looking at the chart of cases over time, we can see the exponential pattern started after a critical mass of 300 (red dash line). In fact I plotted for different countries at different time periods to arrive at this minimum possible number

# ### Identify Pandemic Countries
# Check if the confirmed case takes exponential trend in a weeks time to confirm if a country is in pandemic state. For example find exponential factor of 8-march comparing with 1-march, compare 15-march with 8-march and so on. If the number of cases almost doubled (increase by atleast 80%) for atleast 2 weeks, then the country is termed as pandemic. 
# 
# Also find the rate of spread at the time of exponential spread. It is used to identify countries that are more prone to exponential spread

# In[ ]:


pandemic_country = []
rate_at_exp = []

for cntry in covid.Country.unique():
    country = covid[covid["Country"]==cntry]
    country = country.sort_values("confirmed",ascending=True)
    #By plotting the confirmed cases over time,
    #the confirmed cases takes exponential shape after critical mass of 300 confirmed cases
    country = country[country.confirmed>300]
    country.reset_index(drop=True, inplace=True)
    spread_rate=country.confirmed.pct_change(7).values
    spread_double_counter=0
    tmplst=[]
    #Check if the exponential happened after a week
    for i in range(7,len(spread_rate),7):
        if spread_rate[i] > 0.8:
            spread_double_counter+=1
            tmplst.append(country.confirmed_per_1000[i])
            
    #Term a country pandemic if doubling effect continued for more than a week        
    if spread_double_counter >1:
        pandemic_country.append(cntry)
        rate_at_exp.extend(tmplst)
print("Pandemic Countries:")
pandemic_country


# In[ ]:


pandemic_country = pd.DataFrame(pandemic_country, columns={'Country'})
pandemic_country['risk_level']='Pandemic'


# Find Median value of confirmed_per_1000 in the pandemic country. It is used to find countries in the verge of pandemic

# In[ ]:


median_rate= np.quantile(rate_at_exp,0.5)


# ### Countries at the risk of Pandemic based on high rate of confirmed cases

# In[ ]:


risk_country = covid[(covid.confirmed_per_1000>=median_rate) & (covid.confirmed>300) ].Country.unique()
risk_country = [country for country in risk_country if country not in pandemic_country.Country.values]
print('High Risk Countries :\n', risk_country)

risk_country = pd.DataFrame(risk_country, columns={'Country'})
risk_country['risk_level'] = 'High Risk'


# In[ ]:


risk = risk_country.append(pandemic_country, ignore_index=True)
risk = risk.merge(info, how='left', on='Country')
#Sort from low to high intensity
risk = risk.sort_values(by=['risk_level', 'totalcases'], ascending=True).reset_index(drop=True).reset_index()


# In[ ]:


risk.Country.nunique()


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = risk['alpha3code'],
    z = risk['index'],
    text = risk['Country'],
    customdata=risk[['risk_level', 'totalcases']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>Level : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
    #color=risk['risk_level']
))

fig.update_layout( title='Pandemic and High risk countries')

fig.show()


# ### Finding Correlation Between Variables

# In[ ]:


## Keeping a copy of the same to be used later
info_copy = info.copy()


# In[ ]:


# Taking Relevant fields and changhing the data types of few columns

info = info[['Country','medianage','lung','hospibed','avgtemp','healthexp','avghumidity','casediv1m','gdp2019','smokers','totalcases']]
info["casediv1m"] = info["casediv1m"].astype("float")
info["healthexp"] = info["healthexp"].str.replace(",","").astype("float")
info["gdp2019"] = info["gdp2019"].str.replace(",","").astype("float")


# In[ ]:


#Correlation Study based on the pandemic countries
info_pandemic = info[info["Country"].isin(risk.Country)]


# In[ ]:


print("Correlation of different variables with the cases")
info_pandemic.corr()[["casediv1m","totalcases"]]


# In[ ]:


import seaborn as sns
pairplot =sns.pairplot(info_pandemic)
pairplot.fig.set_size_inches(15,15)


# Lets look at the correlation of totalcases with respect to other features (Bottom row of pair plot).
# 
# Median age as seen above has higher correlation with the total cases and as is accepted universally,implies aged population is more vulnerable with highly affected countries with median age more than 40.
# 
# Lung Patients and smokers are seen to be positively correlated with cases, indicating people with deteriorated respiratory conditions are more vulnerable. Lower cases are seen with death rate from lung diseases varying in the lower levels. On the contrary we should not ignore that higher number of cases are also seen to fall in that range. 
# 
# Owing to the highly contagious nature of the pandemic, even countries with high healthcare systems are not likely to escape adverse effect, as shown by hospital bed per/1000 and health expenditure plots in the figure above. 
# 
# Temperature has a positive correlation and humidity has negative correlation indicating higher temperature and low humid countries could also be vulnerable. But there are no strong evidience against it and could'nt be strongly advocated.
# 
# Vulnereablity of the pandemic is not limited to countries with lower income. Higher income countries also tend to be exposed to  high threat. 

# ## Age Analysis

# In[ ]:


# Box plot showing the median age distribution
sns.boxplot(risk.medianage)


# In[ ]:


sns.kdeplot(risk.medianage)


# In[ ]:


#Calculating the potential range for median age
age_lower_limit = round(risk.medianage.round().quantile(0.25))
age_upper_limit = round(risk.medianage.round().quantile(1))


# We have a smooth distribution above fair vicinity of 25th percentile of the distribution of the median age. Hence 25th percentile is taken as the lower limit and age groups below could be ignore. 
# 
# As we have seen before as age increases the vulerablitiy to the sickness also increases. Therfore it is thought that capping the upper limit makes ambiguity and some countries with higher median age group in their population could be missed out, for whom it is highly likely to get adversely affected. 

# In[ ]:


print("The potential range for countries, of median age which have higher risk of contract "
     ,age_lower_limit,"--",age_upper_limit)


# In[ ]:


print("List of countries within the potential range:")
age_countries = info_copy[(info_copy["medianage"]>=age_lower_limit)][["Country","totalcases","medianage","alpha3code"]]
age_countries = age_countries[~age_countries["Country"].isin(risk.Country)]
age_countries = age_countries.sort_values(by=['medianage', 'totalcases'], ascending=True).reset_index(drop=True).reset_index()
age_countries["Country"].unique()


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = age_countries['alpha3code'],
    z = age_countries['index'],
    text = age_countries['Country'],
    customdata=age_countries[['medianage', 'totalcases']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>Median Age : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
))

fig.update_layout( title='Countries which might be of risk from higher median age.')

fig.show()


# In[ ]:




