#!/usr/bin/env python
# coding: utf-8

# <h1> CovidStrat : COVID-19 Strategy Tactics Resources and Analysis Tools </h1>
# 
# <h2> How is the implementation of existing strategies affecting the rates of COVID-19 infection? </h2>

# <H3> Task Details </H3>
# 
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management. - COVID19 Uncover Challenge

# <h3> Implementation of the Problem </h3>
# 
# The General Question which we require to study in this notebook is how the implementation of existing strategies affecting the rate of COVID-19 Infection. For the sake of this notebook we proceed with the following analogies ans steps.
# 
# 1. Checking which countries are somewhat successful in controlling the rate of COVID-19 Spreads.
# 2. Figuring out the various measures adopted by that very countries to understand how did it affected spreads.
# 3. Checking the countries that displayed a much higher COVID-19 Infections and the growth is exponential as of now.
# 4. Figuring out where the country lacked in implementation tools.
# 
# Finally we can compare how the country that controlled the COVID-19 Proceeded with the community measures than that to the countries that currently exhibit a near to exponential growth of the COVID-19 Infections.

# <h3> Importing the Essential Libraries for the notebook</h3>

# In[ ]:


#Importing Libraries for data manipulation and loading files.
import pandas as pd                              
import numpy as np       
import json
import datetime

#Importing libraries for graphical analyses.
import matplotlib.pyplot as plt                  
import plotly.express as px                      
import plotly.offline as py                     
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns                            

#Other essential libraries to import.
import glob                             
import os     
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')

#Required Libraries for analyses
get_ipython().system('pip install pivottablejs')
from pivottablejs import pivot_ui


# <h3> Reading the Files and Datasets </h3>
# 
# 1. We read the Novel-Corona-Virus-2019-dataset managed by SRK into this notebook. The dataset hold s information about the cumulative case counts of COVID-19 Across the world. The dataset can be viewed and download from [here](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
# 
# 2. A dataset for COVID-19 Cases in China created by me is uploaded for this notebook.
# 
# 3. The UNCOVER Dataset as a part of UNCOVER COVID-19 Challenge is also loaded into this notebook.
# 
# 4. COVID19 Containment and Mitigation Measures Dataset uploaded by Paul Mooney. [See here](//https://www.kaggle.com/paultimothymooney/covid19-containment-and-mitigation-measures)
# 
# 5. China Geo-JSON Document uploaded by sauravmishra1710.

# <h3> Analysis of COVID-19 Confirmed Cases and Deaths for multiple countries </h3>

# In[ ]:


#Reading the cumulative cases dataset
covid_cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#Viewing the dataset
covid_cases.head()

#Grouping the coutries together for further analyses
country_list = covid_cases['Country/Region'].unique()

country_grouped_covid = covid_cases[0:1]

for country in country_list:
    test_data = covid_cases['Country/Region'] == country   
    test_data = covid_cases[test_data]
    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)
    
country_grouped_covid.reset_index(drop=True)
country_grouped_covid.head()


# <h3> Plotting a Running Map for observing the spread of COVID-19 Confirmed Cases </h3>

# <iframe src='https://flo.uri.sh/visualisation/2025509/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2025509/?utm_source=embed&utm_campaign=visualisation/2025509' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'></a></div>

# In[ ]:


#Plotting a bar graph for confirmed cases vs deaths due to COVID-19 in World.

unique_dates = country_grouped_covid['ObservationDate'].unique()
confirmed_cases = []
recovered = []
deaths = []

for date in unique_dates:
    date_wise = country_grouped_covid['ObservationDate'] == date  
    test_data = country_grouped_covid[date_wise]
    
    confirmed_cases.append(test_data['Confirmed'].sum())
    deaths.append(test_data['Deaths'].sum())
    recovered.append(test_data['Recovered'].sum())
    
#Converting the lists to a pandas dataframe.

country_dataset = {'Date' : unique_dates, 'Confirmed' : confirmed_cases, 'Recovered' : recovered, 'Deaths' : deaths}
country_dataset = pd.DataFrame(country_dataset)

#Plotting the Graph of Cases vs Deaths Globally.

fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Deaths'],name='Total Deaths because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Deaths from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
    barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()


fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Recovered'],name='Total Recoveries because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Recoveries from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
    barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()


# <h3> General Observations from the above Running Graph </h3>
# 
# 1. The cases of COVID-19 starts from China as the epicenter with first initial COVID-19 Cases reported in Australia, US, Canada.
# 
# 2. Gradually cases in China increases and the confirmed cases is more than anywhere else in the world.
# 
# 3. Europe emerges later as the new epicenter for the virus, where there is a rapid rise in COVID-19 Cases in European Countries. This outbreak occurs where the confirmed number of COVID-19 Cases in China saturates.
# 
# 4. The confirmed cases of COVID-19 gradually spreads throughout the world, with spike in confirmed cases seen in European regions and US.
# 
# 5. As of April 21st 2020, USA has the highest number of confimed COVID-19 Cases reported, with some European Countries emerging as the 2nd-4th highest cases of COVID-19
# 

# <h3> Digging down further for analyses </h3>
# 
# We observe COVID-19 Confirmed Cases saturation for countries like China, South Korea and Japan. Hence we look much deep into the dataset of Confrimed cases for these countries to understand it in a better context. Also various data measures available from the UNCOVER dataset are taken into consideration.

# <h3> Analysis with the China COVID-19 Cases </h3>

# In[ ]:


#Reading the dataset
search_data_china = country_grouped_covid['Country/Region'] == 'Mainland China'       
china_data = country_grouped_covid[search_data_china]

#Viewing the dataset
china_data.head()


# <H3> Analysis of Spread of COVID-19 in China </h3>

# In[ ]:


with open('../input/china-geo-json/china_geojson.json') as json_file:
    china = json.load(json_file)


# In[ ]:


#Creating the interactive map
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map

formated_gdf = china_data.groupby(['ObservationDate', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['ObservationDate'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['log_ConfirmedCases'] = np.log(formated_gdf.Confirmed + 1)

#Plotting the figure

fig = px.choropleth(formated_gdf,geojson = china,locations='Province/State',scope="asia",
                     color="log_ConfirmedCases", hover_name='Province/State',projection="mercator",
                     animation_frame="Date",width=1000, height=800,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='The Spread of COVID-19 Cases Across China')

#Showing the figure

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# <h3> Understanding the choropleth map of COVID-19 Confirmed Cases in China </h3>
# 
# 1. Hubei Province of China has the highest number of confirmed covid-cases. It emerged as a epicenter in China.
# 2. Gradually over time cases in China increased.
# 3. A stable to negligible growth of newly confirmed cases of COVID-19 is later observed in China.
# 
# <h3> Plotting the graph for Hubei Province (Outbreak Epicenter) </h3>

# In[ ]:


def plot_case_graph(a):
    
    search_city = a

    #Draws the plot for the searched city

    search_data = country_grouped_covid['Province/State'] == search_city       
    search_data = country_grouped_covid[search_data]                           

    x = search_data['ObservationDate']
    y = search_data['Confirmed']
    b = search_data['Confirmed'].values
    
    
    a = b.shape   
    a = a[0]
    growth_rate = []
    
    #Loop to calculate the daily growth rate of cases
    
    for i in range(1,a):                                       
        daily_growth_rate = ((b[i]/b[i-1])-1)*100
        growth_rate.append(daily_growth_rate)                                      

    growth_rate.append(daily_growth_rate)
        
    data = {'Growth' : growth_rate}
    b = pd.DataFrame(data)
    
    #Plotting the chart for confirmed cases vs date     
        
    plt.figure(figsize=(15,5))
    plt.bar(x,y,color="#9ACD32")                              
    plt.xticks(rotation=90)
    
    plt.title('Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Confirmed Cases')

    plt.tight_layout()
    plt.show()
    
    #Plotting the chart daily growth rate in confirmed COVID-19 Cases.
    
    plt.figure(figsize=(15,5))
    
    plt.plot(x,b,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=8,label="Daily Growth Rate of New Confirmed Cases")
    plt.xticks(rotation=90)
    
    plt.title('Percentage Daily Increase of Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Percentage Daily Increase')

    plt.tight_layout()
    plt.show()
    
plot_case_graph('Hubei')


# <h3> Observations from the above Plot </h3>
# 
# We observe a stabel graph post Febraury for Hubei Province in China. Clearly as displayed from the beginning of March, the number of new COVID-19 cases, reported daily has declined sharply. We try to plot the similar dataset for another city to check the same trends.
# 

# In[ ]:


#Plotting Data for Anhui Province (Most affected Province in China after Hubei)
plot_case_graph('Anhui')

#Use can use the above function to plot the cases and grwoth rate of covid-19 cases across China. Feel free to fork this notebook and use.


# <h3> Analogies </h3>
# 
# As observed in the Hubei province a much similar trends are seen over the Last week of Feb - Early March during the period of which the cases stabilizes. We again confirm with plotting of all the provinces in China which I'd already done as saved as image for this dataset.
# 
# <h3> Plotting data for China Provinces </h3>

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Anhui.png')
imgplot = plt.imshow(img)
plt.show()

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Beijing.png')
imgplot = plt.imshow(img)
plt.show()

for i in range(1,17):
    plt.figure(figsize=(15,5))
    img=mpimg.imread('../input/china-covid19-data/Screenshot ({}).png'.format(303+i))
    imgplot = plt.imshow(img)
    plt.show()


# <h3> Deciphering the pattern </h3>
# 
# 1. We observe past 22-02-2012 to 28-02-2020, cases in China stabilizes, post which not new much cases are reported. 
# 2. The pattern goes on same for all the provinces of China.

# <h3> Hunting down for the Possible Causes </h3>
# 
# 1. We look through the entire section of data available under uncover section to search for the events that occured in China during the period.
# 2. I'll update this section regualrly with new analyses to get the best possible reasons out for this saturation curve. 
# 
# <h3> Analysis of Stragegical Factors that Might have contributed to the case stabilization </h3>
# 
# We take help of the the pivotui tool to create a filterable drage and drop dataset to customize the views for mitigation measures taken across world. The tool's install command is uploaded into the libraries section of this notebook.
# 

# In[ ]:


#Importing the dataset
mitigation_policies = pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv")

#Generating the pivoting toolkit
pivot_ui(mitigation_policies)


# <iframe src='https://flo.uri.sh/visualisation/2048862/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2048862/?utm_source=embed&utm_campaign=visualisation/2048862' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>

# 
# <h3> Using the tool generated above. </h3>
# 
# 1. We could select values - (say, country/region add drag it to the 2nd column box 
# 2. The value measure is dragged to the 2nd column to generate country wise view of the mitigation measures taken.
# 
# This tool is highly customizable hence feel free to fork the notebook and incorporate the tool in your analyses.
# 
# <h3> Analysis of Social Measures adopted in China </h3>
# 
# From the above analyses we observe the following trends in China

# <iframe src='https://flo.uri.sh/visualisation/2038617/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2038617/?utm_source=embed&utm_campaign=visualisation/2038617' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>

# In[ ]:


#Analysis for all the important keywords from the dataset

plt.figure(figsize=(15,20))
frame1 = plt.gca()


img=mpimg.imread('../input/china-covid19-data/Word Art.png')
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
imgplot = plt.imshow(img)
plt.show()


# <h3> Understanding the Chinese Government Action Plan </h3>

# In[ ]:


#We load one more dataset avaialable on Mitigation measures to analyze the cases vs mitigation  measures adopted.
mitigation_measures_tot = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

#Generating the pivoting toolkit
pivot_ui(mitigation_measures_tot)


# <h3> What was the amount of measures taken by China during the initial days of COVID-19? </h3>

# <h3> 1. Government Rapid Action Implementation </h3>
# 
# The first case of COVId-19 was reported in 18th December 2020. Though the disease wasn't clear at the moment and it was spelled as "Viral Pneumonia", complusory isolation and confirmed case isolations were adopted and this measure forced by the government was implemented in Hubei province in Wuhan. Domestic Travel restrictions were sooner placed.
# 
# During the initial stages, lockdown imposed. State of emergency was declared in China on January 20th. Further health care systems were rapidly boosted to ensure the curve of COVID-19 stays flat are hospitals arent under stress.
# 
# <h3> 2. Did the mitigation measures in China helped to reduce the COVID-19 Cases which might have been higher? </h3>
# 
# To look forward with this we analyze the day of adoption of a mitigation measure in China vs. the actual case count. For this task the above dataset is concatenated for the daily cases report for that day and further analysis is done over the resultant dataset.
# 
# <iframe src='https://flo.uri.sh/visualisation/2053680/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2053680/?utm_source=embed&utm_campaign=visualisation/2053680' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>

# <h3> Analysis from the graph above </h3>
# 
# We plot the number of social measures and mitigation policies adopted across China and plotted them against the growth rate of new confirmed cases in China. Hubei (the outbreak) location in China was excluded from the analyses as it might saturate the cases. I guess the social measures are taken as to prevent the newer infection across newer locations so that there is no communal spread of the virus. Hence we exclude the outbreak location and plot the graph.
# 
# 1. We observe the Growth of cases in China (excluding Hubei) on Feb 1, 2020 was 20.42%
# 2. Post this mark the level and growth rate of cases declined.
# 3. A high amount of social measures were taken across the China during this period. 
# 4. This might have further contributed to decline of COVID-19 virus to reach newer locations.

# <h3> What the world saw in mitigation measures? </h3>
# 
# We take help of various open source charts and analyses available on Statista as the data for analyses of the trends are insufficient to draw conclusions from it. The following are the details that we would look over.

# <iframe src='https://flo.uri.sh/visualisation/2048271/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2048271/?utm_source=embed&utm_campaign=visualisation/2048271' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>

# <h3> The extent till which lockdown restricted people movement across Eurpoean Unioun </h3>
# 
# <iframe src='https://flo.uri.sh/visualisation/2049461/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2049461/?utm_source=embed&utm_campaign=visualisation/2049461' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>
# 
# <h3> Undersatnding Mitigation Measures Across Europe </h3>
# 
# <h4> Reduction in citizen movement across places during COVID-19 lockdown </h4>
# 
# <img src="https://www.statista.com/graphic/1/1106086/european-city-movements-during-coronavirus-outbreak.jpg" alt="Statistic: Percentage of people moving in selected European cities in the week ending March 22, 2020 compared to typical period prior to coronavirus outbreak* | Statista" style="width: 100%; height: auto !important; max-width:1000px;-ms-interpolation-mode: bicubic;"/></a>
# 
# 

# <h3> The next big steps </h3>
# 
# This notebook would be regular updated by me to check for much newer and diverse data to analyze more trends in spread of COVID-19 and understand it through the terms of health and clinical information. I would love to further test on more datasets across countries. Would update the notebooks with the new findings.
# Contact LinkedIn - https://www.linkedin.com/in/amankumar01/
# 
# Do upvote and comment if you like or wish to suggest something.
