#!/usr/bin/env python
# coding: utf-8

# # Analysis of effects of Lockdown due to COVID-19 on Pollution levels in India
# #### Lockdown in India began on 25th March 2020, when the Indian Goverment put all 1.3 billion of its citizens under mandatory lockdown, as a precautionary measure in order to slowdown the spread of COVID-19 in the country. Only essential services such as medical, water, electricity and grocery services remained active.
# 

# # Objective
# #### In this project we will attempt to determine the effects of the lockdown on the pollution levels in various cities in India. We have been reading reports of clear skies from all across India and will attempt to see if this claim is supported by the data. We have access to a large database containing pollution data dating from 1st January 2015 to 1st May 2020. 

# # Importing necessary libraries

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
sns.set(style='darkgrid',context='notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas_profiling import ProfileReport

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')


# # Importing the Data

# In[ ]:


city_day=pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')
print(city_day.shape)
city_day.head()


# # Exploring the data

# ### Explanation of some features of the dataset:
# 1. PM2.5 - It refers to particles that have diameter less than 2.5 micrometres and remain suspended for longer. These particles are formed as a result of burning fuel and chemical reactions that take place in the atmosphere.
# 2. PM10 - They are very small particles found in dust and smoke. They have a diameter of 10 micrometres (0.01 mm) or smaller. PM10 particles are a common air pollutant
# 3. CO - Carbon monoxide is a temporary atmospheric pollutant in some urban areas, chiefly from the exhaust of internal combustion engines (including vehicles, portable and back-up generators, etc.), but also from incomplete combustion of various other fuels (including wood, coal, natural gas, and trash).
# 4. NH3 - Ammonia is a chemical found in trace quantities in nature, being produced from nitrogenous animal and vegetable matter.
# 5. SO2 - Sulfur dioxide is a noticeable component in the atmosphere, especially following volcanic eruptions. It is a major air pollutant and has significant impacts upon human health.
# 6. NOx - In atmospheric chemistry, NOx is a generic term for the nitrogen oxides that are most relevant for air pollution, namely nitric oxide (NO) and nitrogen dioxide (NO2). These gases contribute to the formation of smog and acid rain, as well as affecting tropospheric ozone. NOx gases are usually produced from the reaction among nitrogen and oxygen during combustion of fuels, such as hydrocarbons in air.
# 7. AQI - It stands for Air Quality Index. It is the most important metric in the measurement of air pollution and is considered as the overall pollution level at a location at any given point of time. It is calculated by transforming the weighted values of 8 individual air pollutantion related parameters (PM10, PM2.5, NO2, SO2, CO, O3, Pb and NH3) into a single number or set of numbers. 
# **The index has 6 categories: Good (0-50), Satisfactory (51-100), Moderately polluted (101-200), Poor (201-300), Very poor (301-400) and Severe (> 401).**
# 

# In[ ]:


city_day.info()


# In[ ]:


city_day.describe()


# * As we can see, many of the colums have a significant amount of null values present in them, especially PM10, NH3 and Xylene.
# * Date column exists as an object dtype and is not in datetime format, we will have to rectify that.

# # Analysing missing values

# In[ ]:


def missing_values_table(df):
        # Total missing values
        missing_val = df.isnull().sum()
        
        # Percentage of missing values
        missing_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        missing_val_table = pd.concat([missing_val, missing_val_percent], axis=1)
        
        # Rename the columns
        missing_val_table_ren_columns = missing_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing values in descending order, 
        # ignoring the colums with no missing values.
        missing_val_table_ren_columns = missing_val_table_ren_columns[
            missing_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)
        
        # Print some summary information
        print ("The DataFrame has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(missing_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        plt.subplots(figsize=(12,8))
        sns.barplot(y=missing_val_table_ren_columns['% of Total Values'],x=missing_val_table_ren_columns.index)
        plt.show()
        # Return the dataframe with missing information
        return missing_val_table_ren_columns
    
missing_values_table(city_day)


# * NH3 and PM10 have missing values in more than 30% of the entries, whereas Xylene has missing values in a massive 64% of rows.
# * As a large percentage of values in Benzene, Xylene and Toulene columns are zero, we can combine them with each other to simplify the data and make visualization easier.

# In[ ]:


city_day1=city_day.copy()


# In[ ]:


city_day['BTX']=city_day['Benzene'] + city_day['Toluene'] + city_day['Xylene']
city_day.drop(['Benzene','Toluene','Xylene'],axis=1,inplace=True)
city_day[city_day['City']=='Gurugram'].tail(7)


# ### The missing values can be present due to
# * Accidental Deletion of data
# * Faulty Equipment
# * Values not being recorded

# # Converting Date column to datetime format

# In[ ]:


city_day['Date'] = pd.to_datetime(city_day['Date'])
print('Data is available for the period {} to {} '.format(city_day['Date'].min(),city_day['Date'].max()))


# In[ ]:


cities_all=city_day['City'].value_counts()
print('We have data for the following cities:')
print(list(cities_all.index))


# # Visualising Data

# #### We group the data by cities and sort them by taking the mean of the levels of all the pollutants.

# In[ ]:


def city_wise_pollution(pollutant):
    i=city_day[[pollutant,'City']].groupby('City').mean().sort_values(by=pollutant,ascending=False).reset_index()
    return i[:10].style.background_gradient(cmap='PuBu')


# In[ ]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.render()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# #### This code cell allows us to create multiple tables side by side in order for us to effectively compare the magnitude of different pollutants in different cities.

# In[ ]:


pm25=city_wise_pollution('PM2.5')
pm10=city_wise_pollution('PM10')
btx=city_wise_pollution('BTX')
so2=city_wise_pollution('SO2')
no2=city_wise_pollution('NO2')
co=city_wise_pollution('CO')
nh3=city_wise_pollution('NH3')
AQI=city_wise_pollution('AQI')

pollutants=['PM2.5','PM10','SO2','BTX','CO','NH3','NO2']

display_side_by_side(pm25,pm10,btx,so2,no2,co,nh3,AQI)


# * Maximum level of pollutants are observed in Delhi, Patna, Gurugram and Ahmedabad.
# * Ahemdabad has an unsually high mean AQI level.
# 

# # Let's visualize these values in the form of a bar plot to get a better understanding.

# In[ ]:


def barplot(df,pollutant):
    bar=df[[pollutant,'City']].groupby(['City']).mean().sort_values(by=pollutant,ascending=False).reset_index()
    ax,fig=plt.subplots(figsize=(16,6))
    sns.barplot(x='City',y=pollutant,data=bar,palette='viridis')
    plt.xlabel('City',fontsize=16)
    plt.ylabel(pollutant,fontsize=16)
    plt.xticks(rotation=45,horizontalalignment='center',fontsize=12)


# # 1. PM2.5

# In[ ]:


barplot(city_day,'PM2.5')


# # 2. PM10

# In[ ]:


barplot(city_day,'PM10')


# # 3. SO2

# In[ ]:


barplot(city_day,'SO2')


# # 4. NO2

# In[ ]:


barplot(city_day,'NO2')


# # 5. CO

# In[ ]:


barplot(city_day,'CO')


# # 6. NH3

# In[ ]:


barplot(city_day,'NH3')


# # 7. AQI

# In[ ]:


barplot(city_day,'AQI')


# * As expected we can see maximum amounts of pollutants in metropolitan cities with large populations and significant commercial and maunfacturing activities.
# * The level of pollutants decrease as the size of the city decreases.

# ## Now we will check the pollution levels with repect to time

# In[ ]:


city_day.set_index('Date',inplace=True)  # only run this line once, second time it will give out error
pollutants=['PM2.5','PM10','SO2','BTX','CO','NH3','NO2']

axes=city_day[pollutants].plot(marker='.',figsize=(16,20),subplots=True,alpha=0.3)
for axes in axes:
    axes.set_xlabel('Years',fontsize=16)
    axes.set_ylabel('Concentration',fontsize=12)
plt.show()


# * There are indications of seasonality towards the increase in PM2.5 and PM10 levels.
# * We should take a closer look to determine the exact nature of the trend.
# * We can also observe a clear increase in SO2, CO and BTX levels from 2018 onwards. 

# In[ ]:


def seasonality(df,val):
    df['year']=[d.year for d in df.Date]   
    df['months']=[d.strftime('%m') for d in df.Date]
    
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    
    sns.boxplot(x='year',y=val, data=df,ax=axes[0])
    sns.lineplot(x='months',y=val,data=df.loc[~df.year.isin([2020]),:]) # not including 2020 as data is available
    axes[0].set_title('Yearly Boxplot',fontsize=14)                     # onlt till May
    axes[1].set_title('Monthly Lineplot',fontsize=14)
    plt.show()


# * Now let's explore the seasonality of the pollution levels in depth.
# * We will be seeing the trends in each type of pollutant individually with the help of boxplots and lineplots

# # Yearly and Monthly Distributions

# # 1. PM10

# In[ ]:


city_day2=city_day.copy()  # df with date as index
city_day.reset_index(inplace=True)  # only run this line once, second time it will give out error
seasonality(city_day,'PM10')


# # 2.PM2.5

# In[ ]:


seasonality(city_day,'PM2.5')


# # 3. SO2

# In[ ]:


seasonality(city_day,'SO2')


# # 4. NO2

# In[ ]:


seasonality(city_day,'NO2')


# # 5. BTX

# In[ ]:


seasonality(city_day,'BTX')


# # 6.O3

# In[ ]:


seasonality(city_day,'O3')


# # 7. AQI

# In[ ]:


seasonality(city_day,'AQI')


# * The general trend amongst all pollutants is that their levels drop in the summer and monsoon months between April and September, after which they experience a sharp rise during the winters.
# * From the boxplots we see that SO2 and NO2 levels increased 2018 onwards.
# * We can notice that the 3rd quantile values for all pollutants as well as AQI for 2019 are genreally lower than those of 2018, this can be due to the fact that 2019 was an extremely wet year for the Indian Subcontinent. The rainfall amounts during both the Southwest Monsoon (June to September) and Northeast Monsoon (October to December) remained **109 per cent** of the Long Period Average (LPA).

# # AQI Comparison between major cities
# #### We will be selecting major cities from various parts of the country for the comaprison, as that will give us a clear interpretation of the overall effect of the lockdown. 

# In[ ]:


cities=['Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata']

city_day_cities=city_day[city_day['Date'] >= '2019-01-01']
AQI_table=city_day_cities[city_day_cities['City'].isin(cities)]
AQI_table=AQI_table[['Date','City','AQI','AQI_Bucket']]
AQI_table


# In[ ]:


AQI_table_pivot=AQI_table.pivot(index='Date',columns='City',values='AQI')
AQI_table_pivot.head(7)


# In[ ]:


AQI_table_pivot.fillna(method='bfill',inplace=True)
AQI_table_pivot.describe()


# In[ ]:


fig=make_subplots(rows=5,cols=1,subplot_titles=('Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata'))

fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Ahmedabad']
                     ,marker=dict(color=AQI_table_pivot['Ahmedabad']
                                  ,coloraxis="coloraxis")),1,1)

fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Bengaluru']
                     ,marker=dict(color=AQI_table_pivot['Bengaluru']
                                  ,coloraxis="coloraxis")),2,1)

fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Chennai']
                     ,marker=dict(color=AQI_table_pivot['Chennai']
                                  ,coloraxis="coloraxis")),3,1)

fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Delhi']
                     ,marker=dict(color=AQI_table_pivot['Delhi']
                                  ,coloraxis="coloraxis")),4,1)

fig.add_trace(go.Bar(x=AQI_table_pivot.index,y=AQI_table_pivot['Kolkata']
                     ,marker=dict(color=AQI_table_pivot['Kolkata']
                                  ,coloraxis="coloraxis")),5,1)

fig.update_layout(coloraxis=dict(colorscale='Temps'),showlegend=False,title_text="AQI Levels")

fig.update_layout( width=1000,height=1200,shapes=[dict(type= 'line',yref= 'paper'
                                                       ,y0= 0,y1= 1,xref= 'x',x0= '2020-03-25',x1= '2020-03-25')])

fig.show()


# * The graph is seperated into 'Before Lockdown' and 'After Lockdown' by the vertical black line placed on the date 25-03-2020  
# * We can see a clear decrease in AQI levels in all cities after the lockdown came into effect.

# # Comparison of Lockdown on individual pollutants

# In[ ]:


cities=['Ahmedabad','Bengaluru','Chennai','Delhi','Kolkata']

pollutants_2019=city_day[(city_day['Date'] >= '2019-01-01') & (city_day['Date'] <= '2019-05-01')]
pollutants_2019.fillna(method='bfill',inplace=True)
pollutants_2019.set_index('Date',inplace=True)

pollutants_2020=city_day[(city_day['Date'] >= '2020-01-01') & (city_day['Date'] <= '2020-05-01')]
pollutants_2020.fillna(method='bfill',inplace=True)
pollutants_2020.set_index('Date',inplace=True)

pollutants_2019=pollutants_2019[pollutants_2019['City'].isin(cities)][['City','SO2','CO','PM2.5','NO']]
pollutants_2020=pollutants_2020[pollutants_2020['City'].isin(cities)][['City','SO2','CO','PM2.5','NO']]


# In[ ]:


def comparison(city):
    # Figure for 2019
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['SO2']
                             ,line=dict(dash='dash',color='blue'),name='SO2'))
    
    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['CO']
                             ,line=dict(dash='solid',color='green'),name='CO'))
                  
    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['PM2.5']
                             ,line=dict(dash='solid',color='rosybrown'),name='PM2.5'))
                  
    fig.add_trace(go.Scatter(x=pollutants_2019.index,y=pollutants_2019[pollutants_2019['City']==city]['NO']
                             ,line=dict(dash='dashdot',color='slategrey'),name='NO'))
                  
    fig.update_layout(title_text=city+' 2019 ',width=800,height=500)
    fig.show()
    
    # Figure for 2020
    
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['SO2']
                             ,line=dict(dash='dash',color='blue'),name='SO2'))
    
    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['CO']
                             ,line=dict(dash='solid',color='green'),name='CO'))
                  
    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['PM2.5']
                             ,line=dict(dash='solid',color='rosybrown'),name='PM2.5'))
                  
    fig.add_trace(go.Scatter(x=pollutants_2020.index,y=pollutants_2020[pollutants_2020['City']==city]['NO']
                             ,line=dict(dash='dashdot',color='slategrey'),name='NO'))
                  
    fig.update_layout(title_text=city+' 2020 ',width=800,height=500)
    fig.show()


# In[ ]:


comparison('Delhi')


# In[ ]:


comparison('Ahmedabad')


# In[ ]:


comparison('Chennai')


# In[ ]:


comparison('Kolkata')


# In[ ]:


comparison('Bengaluru')


# #### There is a clear decrease in the concentrations of all pollutants after the Lockdown came into effect, especially in Ahmedabad, Delhi and Bengaluru.
# #### This decline can be due to the following reasons:
# * Reduced Commercial and Industrial activity
# * Drastic decrease in level of vehicular movement.
# * Suspension of all construction projects.
