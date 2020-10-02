#!/usr/bin/env python
# coding: utf-8

# ### Approach
# The covid data sets captured the spread and affected counts for all the countries. But the date on which COVID entered a country and when community spread started is different from country to country. Conducting an analysis with all the countries together will lead to ambiguous findings. So the main goal is to identify the countries in pandemic state and find out the common features which are main factors for covid mortality. These factors are then used to identify country/group more vulnerable to mortality or in high risk state.
# 
# Note: Data available only till the end of March 2020

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing necessary modules
import pandas as pd
import numpy as np
import glob  
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# In[ ]:


covid = pd.read_csv('../input/johns-hopkins/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv').rename(columns={'country_region':'Country'})
#info = pd.read_csv('countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})


# In[ ]:


info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})
info = info[info.region.isnull()]
info['pop'] = info[~info['pop'].isnull()]['pop'].str.replace(',','').astype('int64')
info['totalcases'] = info[~info['totalcases'].isnull()]['totalcases'].str.replace(',','').astype('int64')
info['casediv1m'] = info[~info['casediv1m'].isnull()]['casediv1m'].str.replace(',','').astype('float')
#info['deathdiv1m'] = info[~info['deathdiv1m'].isnull()]['deathdiv1m'].str.replace(',','').astype('float')
info['deaths'] = info[~info['deaths'].isnull()]['deaths'].str.replace(',','').astype('float')

# info['healthexp'] = info[~info['healthexp'].isnull()]['healthexp'].str.replace(',','').astype('float')
# info['gdp2019'] = info[~info['gdp2019'].isnull()]['gdp2019'].str.replace(',','').astype('float')


# In[ ]:


covid = covid.merge(info[['Country', 'pop']], how='left', on='Country')


# In[ ]:


covid.Country.nunique()


# In[ ]:


covid['confirmed_per_1000']= covid.confirmed*1000/covid['pop']
covid["deaths_per_confirmed"] = covid["deaths"]/covid["confirmed"]


# Function to get the pandemic countries is given at the last cell of the notebook.You can go to our previous notebook at the contract impact where this is discussed in detalis, at : https://www.kaggle.com/ideas2it/analysis-of-population-at-higher-risk-of-covid-19

# In[ ]:


pandemic_country = get_pandemic_countries(covid)
pandemic_country


# In[ ]:


pandemic_country = pd.DataFrame(pandemic_country, columns={'Country'})
pandemic_country['risk_level']='Pandemic'


# In[ ]:


risk = pandemic_country.merge(info, how='left', on='Country')
risk = risk.sort_values(by=['risk_level', 'totalcases'], ascending=True).reset_index(drop=True).reset_index()


# In[ ]:


#Finding the top ten countries with highest mortalities.
top_10_high_mortality_country = risk[["Country","deaths","risk_level","deathdiv1m"]].sort_values("deaths",ascending=False)[0:10]


# In[ ]:


# Top ten countries with highest mortalities
top_10_high_mortality_country.Country.unique()


# In[ ]:


# Plotting deaths over time for high mortality countries
covid[covid.Country.isin(top_10_high_mortality_country.Country)][['last_update', 'deaths','Country']].pivot(index='last_update', columns='Country', values='deaths').plot(figsize=(20,5))
axes = plt.gca()
axes.set_ylim([0,5000])
plt.ylabel('Deaths')
plt.xlabel('Date')


# Above is the overtime increase of the fatalities from the top ten countries with highest mortalities with time. All of these countries are
# pandemic stricken and as is expected would have higher mortalities. Only China, shown by the orange line shows a plateaued the death
# curve. Some countries are on exponential increase of deaths. Countries like Germany, Netherlands and UK are seen to be catching up the
# exponential effect lately.

# # Correlation Study

# In[ ]:


## Keeping a copy of the same to be used later
info_copy = info.copy()


# In[ ]:


# Taking Relevant fields and changhing the data types of few columns

info = info[['Country','medianage','lung','hospibed','avgtemp','healthexp','avghumidity','casediv1m','gdp2019','smokers','totalcases',"deaths","malelung","femalelung","deathdiv1m","healthperpop"]]
info["casediv1m"] = info["casediv1m"].astype("float")
info["healthexp"] = info["healthexp"].str.replace(",","").astype("float")
info["gdp2019"] = info["gdp2019"].str.replace(",","").astype("float")


# In[ ]:


#Correlation Study based on the pandemic countries
info_pandemic = info[info["Country"].isin(risk.Country)][["hospibed","Country","healthexp","smokers","lung","malelung","femalelung","gdp2019","healthperpop","deaths","deathdiv1m"]]


# In[ ]:


print("Correlation of different variables with the cases")
corr = info_pandemic.corr()[["deaths"]].reset_index()
corr


# In[ ]:


sns.barplot(corr['index'],corr.deaths)
plt.xticks(rotation="vertical")


# ### Insights
# Both female lung patients and male lung patients have positive correlation with mortality. But males are worse off than females. 
# 
# Higher healthcare infrastructure,indicated by hospibed, plays an important role in reducing mortality. We see a negative correlation between deaths and the same. Also higer per capita expenditure improves healthcare infrastructure and per person availablity of health resources could reduce tackling mortality. 
# 
# Again, mortality doesn't depend on the income level of the country and how much they invest in healthcare expenditure, but how the expenditure in distributed among the population and how well people could avail them,as discussed above. As we see the top ten countries with highest mortalities, most of the world superpowers including (US and China) are present in that.

# # Scatter Plot Analysis on Variables

# In[ ]:


import seaborn as sns
pairplot =sns.pairplot(info_pandemic[["hospibed","healthexp","healthperpop","smokers","lung","malelung","femalelung","gdp2019","deaths"]])
pairplot.fig.set_size_inches(15,15)


# ### Insights
# Lets look at the correlation of deaths with respect to other features (Bottom row of pair plot).
# 
# Most of the countries which have lower hospital beds per 1000 population have seen the highest mortalities. Although there are still points where the hospital beds are lower with lower mortality, but we also see lowest mortalities where the hospital beds are more per thousand population.
# 
# The scatter of the smokers shows that highest deaths numbers falls above 20 smokers per 1000. 
# 
# Highest cases are reported for countries in the lower ranges of the gdp plot. However higher GDP countries like the US and China also have high mortality from the sickness. 

# # Healthcare Infrastructre Impact on Mortality

# In[ ]:


# Box plot showing the hospital beds and per capita healthcare expenditure distribution
f , axes= plt.subplots(1,2,figsize=(15,5))
sns.boxplot(risk.hospibed,orient="h",ax=axes[0]).set_title("hospibed distribution")
sns.boxplot(risk.healthperpop,orient="h",ax =axes[1]).set_title("Healthperpop distribution")


# As we see the density is highest in the lower levels of the per capita health expenditure, among the pandemic countries. 
# We have noted before that higher healthcare infrastructure could help tackle the mortality rate. We must keep in mind that
# the spread of the sickness could depend on various factors including travels, measures of the government for isolation and lockdowns etc, but as we see the recovery and tackling the mortality highly depends on how the health care system stands stong 
# amidist this. Also, as argued before respiratory comorbidities is also likely to shoot up mortality rates.

# In[ ]:


#Calculating the potential range for median age
hopibed_lower_limit = round(risk.hospibed.round().min())
hospibed_upper_limit = round(risk.hospibed.round().quantile(0.75))
healthperpop_lower_limit = round(risk.healthperpop.round().min())
healthperpop_upper_limit = round(risk.healthperpop.round().quantile(0.75))


# In[ ]:


print("The range of hospital beds per 1k population at high risk is",hopibed_lower_limit,"--", hospibed_upper_limit)


# From the most affected countries, we could see that the number of hospital beds per 1k population varies between 2 - 5 beds per 1k population and country with maxium number of hospital beds per 1k population ofhas lower death cases as compared to the ones in the given range.This could further imply the more the population level at given health expenditure the pie gets smaller and hence this could be seen.

# #### Classiflying Higher Risk Countries, which are vulnerable to catch up with the Pandemic countries with respect to mortality based on healthcare information.

# In[ ]:


print("High risk countries which are probable to catch up with the Pandemic countries on mortality based on health care expenditure per capita ")
mortality_catching_up_country = info_copy[(info_copy["healthperpop"]>healthperpop_lower_limit)&(info_copy["healthperpop"]<=healthperpop_upper_limit)&(info_copy["totalcases"]>300)]
mortality_catching_up_country = mortality_catching_up_country[~mortality_catching_up_country["Country"].isin(pandemic_country["Country"])]
mortality_catching_up_country = mortality_catching_up_country.sort_values(by=['healthperpop', 'deaths'], ascending=True).reset_index(drop=True).reset_index()
mortality_catching_up_country["Country"].unique()


# These are the countries where the healthcare per capita falls in the critical range and has number of confirmed cases greater than 300.

# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = mortality_catching_up_country['alpha3code'],
    z = mortality_catching_up_country['index'],
    text = mortality_catching_up_country['Country'],
    customdata=mortality_catching_up_country[['healthperpop', 'deaths']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Deaths : %{customdata[1]}<br>Healthexppop : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
))

fig.update_layout( title='Countries at high mortality risk from lower per capital healthcare expenditure.')

fig.show()


# # Analysis of Smokers and incidence of death

# In[ ]:


sns.kdeplot(info_pandemic.smokers,info_pandemic.deaths,n_levels=12).set_title("Bivariate Density plot for deaths and smokers in Pandemic Countries")


# In[ ]:


sns.regplot(info_pandemic[info_pandemic["deaths"]<10000].smokers,(info_pandemic[info_pandemic["deaths"]<10000].deaths),order=1).set_title("Incidence of Death and smokers in Pandemic Countries")


# In[ ]:


info_non_pandemic = info[~info["Country"].isin(info_pandemic.Country)]


# In[ ]:


sns.regplot(info_non_pandemic[info_non_pandemic["deaths"]<50].smokers,info_non_pandemic[info_non_pandemic["deaths"]<50].deaths).set_title("Incidence of Death and smokers in Least Mortality Affected Countries")


# In[ ]:


#info_non_pandemic.deaths.describe()


# There has been a world wide study on whether smokers are more vulnerable to the sickness. Although there hasn't been any solid concluded ground in which we could put forth this as an true argument. Researhers have come up with mixed findings related to this.
# 
# In the density plot we see that although the plot is significantly dense in the areas where we have lower than median smokers with lower mortality, a fair portion of higher mortality is observed at lower range of smokers and sometime lower mortality at higher range of smoker is also seen. 
# 
# This is evidient from the incidence of the deaths and smokers in the pandemic countries that deaths and somkers are negatively correlated. The line of average is negatively sloped. A study based in France, says that nicotine might prevent smokers from contracting the sickness and potentially blocks the virus from attaching to the cells. But if a patient catches the sickness, a smoker is more likely to have aggravated health situation. 
# However, this argument doesn't advocate intake of nicotine based products as an preemtive action against coronavirus, owing to the universally accepted fact about the harmful health-effects of nicotine consumption.
# 
# However, in the countries with least mortality, in the third graph above, we see an opposite picture. Mortality is higher with higher somkers. 
# 
# Since, the final conclusion is slightly ambiguous in nature, a further suggestive study as we have done with healthcare expenditure per capita, may be misleading in various aspect. 

# In[ ]:


## Function to get the pandemic countries
def get_pandemic_countries(df):
    pandemic_country = []
    rate_at_exp = []
    for cntry in df.Country.unique():
        country = df[df["Country"]==cntry]
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
    return pandemic_country


# In[ ]:




