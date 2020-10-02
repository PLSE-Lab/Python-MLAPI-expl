#!/usr/bin/env python
# coding: utf-8

# # Visualization of Covid-19 pandemic

# **Coronavirus(Covid_19) is an illeness caused by virus that spreads from person to person. This virus has spread throughout the world. A novel coronavirus disease emerged in a seafood and poultry market in the Chinese city of Wuhan in 2019. Cases have been detected in most countries worldwide, and on March 11, 2020, the World Health Organization characterized the outbreak as a pandemic.**
# 
# 
# **I have taken the dataset covid_19_data.csv from Kaggle in order to visualize the effect of this desease worldwide.**

# In[ ]:


import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
import numpy as np


# In[ ]:


Covid_19 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col=0) #reading the file


# In[ ]:


Covid_19.head(5)


# **Dataset has all the above infromation. let's change the name of some columns and it's values for our convenience.**

# In[ ]:


Covid_19.rename(columns = {'Province/State':'Province', 'Country/Region':'Country', 'Last Update': 'Last_Update','ObservationDate':'Observation_Date'}, inplace = True) 
   


# In[ ]:


Covid_19.Country.unique()


# In[ ]:


Covid_19.loc[Covid_19['Country']=='Mainland China', 'Country'] = 'China'


# In[ ]:


Covid_19.loc[Covid_19['Country']=="('St. Martin',)", 'Country'] = 'St.Martin'


# In[ ]:


Covid_19.Country.nunique()


# **There are 223 countires in the dataset. Since we have columns of confirmed cases, recovered cases and deaths, if we subtract recovered cases and deaths from confirmed cases we can have number of active cases.**
# 
# **Active case =Confirmed-Case-Recovered**

# In[ ]:


active_cases = Covid_19['Confirmed']-Covid_19['Deaths']-Covid_19['Recovered']
Covid_19['Active_Cases'] = active_cases


# **New dataset including column of Active_Cases is as follows**

# In[ ]:


Covid_19.head(5)


# **Now let's check the timeline of the dataset.**

# In[ ]:


max_obs_date = Covid_19.Observation_Date.max()
max_obs_date


# In[ ]:


min_obs_date = Covid_19.Observation_Date.min()
min_obs_date


# **The time frame of this data is from 22 January to 7 May 2020**
# ### Now let's checkout the world wide scenario of Confirmed cases, Deaths, and Recovered Cases

# In[ ]:


list_Confirmed = Covid_19[['Confirmed']] #maniuplating orginial dataframe
Confirmed_df = DataFrame(list_Confirmed,columns=['Confirmed'])
list_Deaths = Covid_19[['Deaths']]
Deaths_df =  DataFrame(list_Deaths,columns=['Deaths'])
list_Recovered = Covid_19[['Recovered']]
Recovered_df =  DataFrame(list_Recovered,columns=['Recovered'])
world_confirmed = Confirmed_df[Confirmed_df.columns[-1:]].sum()
world_deaths = Deaths_df[Deaths_df.columns[-1:]].sum()
world_recovered = Recovered_df[Recovered_df.columns[-1:]].sum()
sizes = [world_confirmed,world_deaths,world_recovered]
labels = ['Confirmed_Cases', 'Deaths', 'Recovered_Cases']
my_colors = ['lightblue','lightcoral','silver']
explode = (0.1, 0.1, 0.1) 
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=60, shadow = True, colors=my_colors, explode=explode) # performing visualization
plt.title('World wide scenario of Confirmed cases,Deaths, and Recovered Cases')
plt.axis('equal')
plt.legend(loc="best")
fig = plt.gcf()
fig.set_size_inches(8,8)
plt.show()



# ### To dig deep we can visualize the scenario of confirmed cases, active cases, recovered, andcases and deaths according to dates from Jan 22 to May 7 2020 in the form of scatter chart.

# In[ ]:


#grouping coulmns in original dataframe
data_over_time= Covid_19.groupby(['Observation_Date'])[['Confirmed', 'Active_Cases', 'Recovered', 'Deaths']
                                                  ].sum().reset_index().sort_values('Observation_Date',ascending=True).reset_index(drop=True)

fig1 = go.Figure()

# Add traces
fig1.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Confirmed,
                    mode='lines',
                    name='Confirmed',
                    marker_color='#A9A9A9'))
fig1.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Deaths,
                    mode='lines',
                    name='Deaths',
                    marker_color='#BDB76B'))
fig1.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Recovered,
                    mode='lines',
                    name='Recovered',
                    marker_color='#45CE30'))

fig1.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Active_Cases,
                    mode='lines',
                    name='Active_Cases',
                    marker_color='#FFA07A'))

fig1.show()


# **We can see that as the days are passing from January 22 to July 8, Confirmed cases have increased very rapidly from 653 to 12.04 mliion. Recovery also seems to have increased in the month of June and July.**
# 
# 
# **To make more sense in terms of recovery and mortality it's good to have general idea about mortality rate and recovery rate worldwide**
# 
# ### Mortality and Recovery Rate worldwide

# In[ ]:


#adding column in manipulated dataframe and performing calculation
data_over_time['Mortality_Rate'] = np.round(100*data_over_time['Deaths']/data_over_time['Confirmed'],2)
data_over_time['Recovery_Rate'] = np.round(100*data_over_time['Recovered']/data_over_time['Confirmed'],2)

#visualization
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Mortality_Rate,
                    mode='lines',
                    name='Mortality_Rate',
                    marker_color='#A9A9A9'))
fig2.add_trace(go.Scatter(x=data_over_time.Observation_Date, y=data_over_time.Recovery_Rate,
                    mode='lines',
                    name='Recovery_Rate',
                    marker_color='#BDB76B'))

fig2.show()


# **From the above graph highest recovery rate was in the beginnig of March, then it strated decreasing from around 10th of March and again started increasing from April to July. In the July we again have highest recovery rate.**
# 
# **Death rate seems to have highest in the month of April and May.**

# ### More interactive way of showing confirmed cases increases world wide from January to July using choropleth is following

# In[ ]:


#grouping in original dataframe
df_countrydate = Covid_19[Covid_19['Confirmed']>0]
df_countrydate = df_countrydate.groupby(['Observation_Date','Country']).sum().reset_index()
df_countrydate

#visualization
fig3 = px.choropleth(df_countrydate, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country", 
                    animation_frame="Observation_Date"
                   )
fig3.update_layout(
    title_text = 'Global Spread in terms of confirmed cases of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig3.show()


# **We can also create same kind of same choropleth map for Recovered Cases, Active Cases and Deaths.**

# **From the above graphs we can see that number of confirmed cases has been increasing worldwide. It also shows date wise increase in the number of confirmed cases of corona virus in different countries.**
# 
# **Now let's check exactly which countries have highest number of Confirmed cases, Deaths, Active and Recovered Cases worldwide**

# ### Confirmed cases worldwide

# In[ ]:



Covid_19["World"] = "World" # in order to have a single root node

fig4 = px.treemap(Covid_19, path=['World','Country'], values='Confirmed', color='Country',
                  hover_data=['Confirmed'],
                  color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.sum(Covid_19['Confirmed']))
fig4.show()


# **Countries US, Brazil, Russia, and Spain have highest confirmed cases**
# 
# ### Deaths worldwide

# In[ ]:


fig5 = px.treemap(Covid_19, path=['World','Country'], values='Deaths', color='Country',
                  hover_data=['Deaths'],
                  color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.sum(Covid_19['Deaths']))
fig5.show()


# **Countries with highest number of deaths are US, UK, Italy, and Spain.**
# 
# ### Recovered cases worldwide

# In[ ]:


fig6 = px.treemap(Covid_19, path=['World','Country'], values='Recovered', color='Country',
                  hover_data=['Recovered'],
                  color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.sum(Covid_19['Recovered']))
fig6.show()


# **Countries with highest recovered cases are US, Brazil, Russia, and Germany**
# 
# ### Active cases worldwide

# In[ ]:


fig7 = px.treemap(Covid_19, path=['World','Country'], values='Active_Cases', color='Country',
                  hover_data=['Active_Cases'],
                  color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.sum(Covid_19['Active_Cases']))
fig7.show()


# **Top countries with active cases are US, Brazil, UK, and Russia**

# **Top countries with highest confirmed cases, deaths, revodered cases, and active cases are US, Brazil, Russia, UK, Spain, Germany, Italy and France**

# **Let's dig deep and visualize country wise scenario. Since we have provinces, we can visualize confirmed cases, active cases, recovered cases and deaths region wise for given countries**
# 
# ### Countrywise visualization

# In[ ]:



#defining function
def analysis_of_province(Country_name):
    
    country = Covid_19.groupby('Country') #grouping dataframe countrywise
    
    country_name_group = country.get_group(str(Country_name)) # country name
    
    country_name_province =  country_name_group.groupby('Province') # province group country wise
    
    province = country_name_group['Province'].unique() #list of province
    
    total_confirmed = [] #totalconfirmed cases by province
    
    total_deaths = [] #total deaths by province
    
    total_recovered = [] #total recovered cases by province
    
    total_active = [] #total active cases by province
    
    for i in province:
        total_confirmed.append(country_name_province.get_group(i).Confirmed.sum())
    
    for i in province:
        total_deaths.append(country_name_province.get_group(i).Deaths.sum())  
        
    for i in province:
        total_recovered.append(country_name_province.get_group(i).Recovered.sum())  
     
    for i in province:
        total_active.append(country_name_province.get_group(i).Active_Cases.sum()) 
        
#     total_confirmed.sort(reverse = True)
#     total_deaths.sort(reverse = True)
    #creating dataframe from lists
    df = pd.DataFrame(data={'Province': province,
                            'Confirmed_Cases':total_confirmed,
                            'Deaths':  total_deaths,
                            'Recovered_Cases': total_recovered,
                            'Active_Cases': total_active})
    #performing visualization
    fig8 = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Province,df.Confirmed_Cases,df.Deaths, df.Recovered_Cases, df.Active_Cases],
               align='left'))])
    fig8.show()
   


# In[ ]:


analysis_of_province('US')


# **As we can see areas  Illinois, California, New York, Massachusetts, Georgia, Florida, New Jersey, Texas, and  Pennsylvania are most infected. Now Let's check mortality and recovery rate over the given timeline in US.**

# In[ ]:


#defining user defined function
def rates_country(country_name):
    
    country = Covid_19.groupby('Country') #grouping dataframe countrywise
    
    country_name_group = country.get_group(str(country_name)) # country name
    
    #further grouping
    rates_country= country_name_group.groupby(['Observation_Date'])[['Confirmed', 'Active_Cases', 'Recovered', 'Deaths']
                                                  ].sum().reset_index().sort_values('Observation_Date',ascending=True).reset_index(drop=True)
    #adding columns and performing calculations
    rates_country['Mortality_Rate'] = np.round(100*rates_country['Deaths']/rates_country['Confirmed'],2)
    rates_country['Recovery_Rate'] = np.round(100*rates_country['Recovered']/rates_country['Confirmed'],2)
    
    
    #visualization
    fig9 = go.Figure()

    # Add traces
    fig9.add_trace(go.Scatter(x=rates_country.Observation_Date, y=rates_country.Mortality_Rate,
                    mode='lines',
                    name='Mortality_Rate',
                    marker_color='#A9A9A9'))
    fig9.add_trace(go.Scatter(x=rates_country.Observation_Date, y=rates_country.Recovery_Rate,
                    mode='lines',
                    name='Recovery_Rate',
                    marker_color='#BDB76B'))

    fig9.show()

    
   


# In[ ]:


rates_country('US')


# **We can see from the graph that recovery rate is highest in the month of June and July in US and mortality rate was high in May but getting lower in June and July. That means situation is getting better in US for last two months.**
# 
# **We can check the same for other countries using the user defined functions code analysis_of_province, and rates_country in the given code.**
