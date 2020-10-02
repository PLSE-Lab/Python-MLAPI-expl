#!/usr/bin/env python
# coding: utf-8

# # The Interactive Application can be found at the end of each Data Mining Algorithm
# # This will cause the notebook to stop running if you decide to 'Run All' for the execution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px #Importing plotly express library
import plotly.graph_objects as go #Importing plotly graph objects library
from plotly.subplots import make_subplots
import seaborn as sns #Importing seaborn library
import matplotlib.pyplot as plt #Importing matplotlib library
import graphviz #Importing graphviz library
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
#import KMeans algorithm
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing Libraries

# ## Importing dataset

# In[ ]:


suicide_rate = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
suicide_rate.head()


# ## Reviewing Dataset

# In[ ]:


suicide_rate.describe() #Getting general overview of dataset


# *Suicide_no & suicides/100k pop show very high maximum values, potential outliers*

# In[ ]:


suicide_rate.columns.tolist() #Getting column names


# *It is noted that columns: " gdp_for_year ($) ", " gdp_per_capita ($) ", & "suicides/100k pop" May require a name change*

# In[ ]:


suicide_rate.dtypes #Getting column types


# *It is noted that column: "gdp_for_year ($)" is of type object and not numeric *

# In[ ]:


suicide_rate.describe() #Counting row data


# In[ ]:


suicide_rate.isnull().sum() #Sum of null/missing data


# *It is noted that HDI column is missing more than 50% of data*

# ## Boxplot being used to show the outliers of the data

# In[ ]:


fig = px.box(suicide_rate, y="suicides_no")
fig.update_layout(title="Boxplot of Number of Suicides")
fig.show()


# In[ ]:


x = suicide_rate['suicides_no']
IQR = x.quantile(0.75) - x.quantile(0.25) #Finding the interquartile range
uFence = x.quantile(0.75) + (1.5*IQR) #Calculating upperfence
(x>uFence).sum() #Total amount of records greater than the upperfence


# *Data indicates there are over 3909(approx. 15%) results which span above the upper fence. There is no apparent outlier/abnormality that needs to be removed*

# ## Boxplot being used to show the outliers of the data

# In[ ]:


fig = px.box(suicide_rate, y="suicides/100k pop") #Create a boxplot of suicides/100k pop to identify values above upper fence
fig.update_layout(title="Boxplot of Suicides per 100k persons.")
fig.show()


# In[ ]:


x = suicide_rate['suicides/100k pop']
IQR = x.quantile(0.75) - x.quantile(0.25) #Finding the interquartile range
uFence = x.quantile(0.75) + (1.5*IQR) #Calculating upperfence
(x>uFence).sum() #Total amount of records greater than the upperfence


# *Data indicates there are over 2046(approx 8%) results which span above the upper fence. There is no apparent outlier/abnormality that needs to be removed*

# ## Cleaning the Data

# In[ ]:


type(suicide_rate[' gdp_for_year ($) '][53])
#showing the data type that is in the column


# In[ ]:


suicide_rate[' gdp_for_year ($) '] = suicide_rate[' gdp_for_year ($) '].replace(", ","",regex=True) #Remove commas with spaces
suicide_rate[' gdp_for_year ($) '] = suicide_rate[' gdp_for_year ($) '].replace(",","",regex=True) #Remove commas
suicide_rate[' gdp_for_year ($) '] = suicide_rate[' gdp_for_year ($) '].astype('float64')
#using .to_numeric function to convert the categorical data to numerical data


# In[ ]:


type(suicide_rate[' gdp_for_year ($) '][92])
#proving that the data has been converted to numerical data


# In[ ]:


suicide_rate[' gdp_for_year ($) '].describe()
#describing the column since the records in the column has now been converted to numerical data


# In[ ]:


new_suicide = suicide_rate [['country',
                          'year',
                          'sex',
                          'age',
                          'suicides_no',
                          'population',
                           'suicides/100k pop',
                          ' gdp_for_year ($) ',
                          'gdp_per_capita ($)',
                          'generation']]
new_suicide

#assigning specific columns from the previous dataframe 'suicide_rate' to the new dataframe 'new_suicide'


# In[ ]:


new_suicide = new_suicide.rename(columns={'suicides/100k pop': 'suicide_100', ' gdp_for_year ($) ':'gdp_year',
                                            'gdp_per_capita ($)':'gdp_capita'}) #Renaming columns with unorthodox names
new_suicide


# ## Data Extraction
# 
# These visualizations compare the suicide rates of the G7 countries, 7 countries from the south pacific region and 7 countries of the caribbean.
# * **G7 Countries include**: *United States, Italy, Japan, Canada, Germany, France, United Kingdom*
# * **Caribbean Countries include**: *Cuba, Puerto Rico, Jamaica, Trinidad and Tobago, Guyana, Suriname*
# * **Indo-Pacific Countries**: *Australia, Singapore, Philippines, New Zealand, Thailand*
# * **Rich European Countries**: *Sweden, Norway, Finland, Denmark, Switzerland, Luxembourg, Ireland, Iceland*
# 

# In[ ]:


def groupCountry(s): #Function to check if country is a part of a group
    if s in (["United States", "Italy", "Japan", "Canada", "Germany", "France", "United Kingdom"]):
        return "G7"
    elif s in (["Cuba", "Bahamas", "Puerto Rico", "Jamaica", "Trinidad and Tobago", "Guyana", "Suriname"]):
        return "Caribbean"
    elif s in (["Australia", "Singapore", "Philippines", "New Zealand", "Thailand"]):
        return "Indo-pacific"
    return "Rich European Countries"


# In[ ]:


g7 =["United States", "Italy", "Japan", "Canada", "Germany", "France", "United Kingdom"] #G7 Countries
cc = ["Cuba", "Bahamas", "Puerto Rico", "Jamaica", "Trinidad and Tobago", "Guyana", "Suriname"] #Caribbean Countries
ip = ["Australia", "Singapore", "Philippines", "New Zealand", "Thailand"] #Indo-pacific countries
rc = ["Sweden", "Norway", "Finland", "Denmark", "Malta", "Netherlands", "Luxembourg", "Ireland", "Iceland"] #Rich European Countries
g7_df = new_suicide[new_suicide["country"].isin(g7)] 
cc_df = new_suicide[new_suicide["country"].isin(cc)]
ip_df = new_suicide[new_suicide["country"].isin(ip)]
rc_df = new_suicide[new_suicide["country"].isin(rc)]
combined_df = pd.concat([g7_df,cc_df,ip_df,rc_df])
combined_df['country_group'] = new_suicide["country"].apply(groupCountry) #Create new column to identify the country-group of a country
combined_df


# ## Visualizations of Data

# In[ ]:


ageXtract = combined_df.groupby(['sex','age','country_group','generation'], as_index=False)['suicide_100','population','gdp_capita'].mean(), #Group and extract data, average suicides, population and GDP, focuses on age
ageXtract[0]['age'].replace("5-14 years", "05-14 years", regex=True, inplace=True)
ageXtract = ageXtract[0].sort_values(by='age', ascending=True)
ageXtract


# In[ ]:


fig = px.bar_polar(ageXtract, r="suicide_100", theta="generation",
                   color="country_group", template="none", color_continuous_scale=px.colors.sequential.Rainbow_r )
fig.update_layout(title="Visualization of suicides per 100k Population per Generation per Region", legend_title="Regions")
fig.show()


# In[ ]:


fig = px.scatter(ageXtract, x="suicide_100", y="gdp_capita", color="sex", hover_data=['country_group'],
                    template="none" )
fig.update_layout(title="Graph of GDP per Capita vs Suicide per 100kPopulation vs Gender", yaxis_title="GDP per Capita",xaxis_title="Suicides per 100k Population", legend_title="Gender")
fig.show()


# In[ ]:


yearXtract = combined_df.groupby(['sex','year','country_group','generation'], as_index=False)['suicide_100','population','gdp_capita','suicides_no'].mean(), #Group and extract data, average suicides, population and GDP, focuses on year
yearXtract = yearXtract[0].sort_values(by=['year','population'], ascending=True)
yearXtract


# In[ ]:


yearXtract2 = yearXtract.groupby(['year'], as_index=False)['suicide_100','population','gdp_capita','suicides_no'].mean()
yearXtract2


# In[ ]:


fig = make_subplots(rows=2, cols=2,
                   subplot_titles=("Average Population trend 1995-2016", "Average Suicide per 100k Population trend 1995-2016", "Average Suicides trend 1995-2016", "Average GDP per Capita trend 1995-2016"))
#fig = px.histogram(x=yearXtract2['year'], y=yearXtract2['population'])
fig.add_trace(go.Scatter(x=yearXtract2['year'], y=yearXtract2['population'], mode='lines', name='Population'), row=1, col=1)
fig.add_trace(go.Scatter(x=yearXtract2['year'], y=yearXtract2['suicide_100'], mode='lines+markers', name='Suicide Rate'), row=1, col=2)
fig.add_trace(go.Scatter(x=yearXtract2['year'], y=yearXtract2['suicides_no'], mode='lines+markers', name='Suicides'), row=2, col=1)
fig.add_trace(go.Scatter(x=yearXtract2['year'], y=yearXtract2['gdp_capita'], mode='lines+markers', name='GDP per Capita'), row=2, col=2)
fig.update_layout(title_text='View of Population, Suicide per 100k Population, Suicides & GDP per Capita Trends 1995-2016',xaxis_title="Year",yaxis_title="Suicides per 100k Population", legend_title="Trace")

fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=1, col=2)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=2, col=2)

fig.update_yaxes(title_text="Average Population", row=1, col=1)
fig.update_yaxes(title_text="Average Suicides Rate", row=1, col=2)
fig.update_yaxes(title_text="Average Suicides", row=2, col=1)
fig.update_yaxes(title_text="Average GDP per Capita", row=2, col=2)
fig.show()


# ## Additional Visualization

# In[ ]:


total_deaths_binned = pd.cut(combined_df['year'],3, 
                             labels=[
                                 'Suicides in the 80s', 'Suicides in the 90s', 'Suicides after 2000 upto 2016'
                             ])
total_deaths_binned


# In[ ]:


total_deaths_binned.value_counts()


# In[ ]:


fig = go.Figure(data=go.Pie(labels=total_deaths_binned.value_counts().index,values=total_deaths_binned.value_counts(), pull=[0.1,0,0]))
fig.update_layout(title="Grouped suicides by time period")
fig.show()


# In[ ]:


suicide_generation = combined_df['generation'].value_counts()
suicide_generation


# In[ ]:


labels = ['Generation X',
          'Silent',
          'Millenials',
          'Boomers',
          'G.I. Generation',
          'Generation Z']
values = suicide_generation

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, pull=[0.1, 0, 0, 0, 0, 0])])
fig.update_layout(title="Observation of the Generation most frequently occuring in the dataset")
fig.show()


# In[ ]:


fig = go.Figure(data=go.Box(
    x=combined_df.year,
    name = 'Total years people committed suicide',
    marker = dict(
        color = 'rgb(12, 128, 128)',        
    )
)
               )
fig.update_layout(title="Boxplot displaying where the majority of the data falls(1993-2008)")
fig.show()


# In[ ]:


#fig = px.line(combined_df, x=combined_df.country, y=combined_df.gdp_year, color=combined_df.generation, height=500)
#fig.update_layout(title="The occurence of GD", xaxis_title="Countries", yaxis_title="GDP per Year", legend_title="Generation")
#fig.show()


# In[ ]:


fig = px.histogram(combined_df.sort_values(by="suicide_100"), x='country_group', y='suicide_100', color='generation', barmode="group", histfunc="sum", title="Country GPD Generation", height=500
            , hover_data=['country'])
fig.update_layout(title="A look at suicides per 100k across 4 regions by the different generations", xaxis_title="Regions", yaxis_title="Suicides per 100k Population", legend_title="Generation")
fig.show()


# * The graph above shows up to 12k suicides per 100k population on the y-axis, however, this may be a bit misleading. That number represents the added stacked total for ever year and all countries within the region*

# In[ ]:


fig = px.histogram(yearXtract, y="population", x="country_group", histfunc="sum", color="sex", barmode="stack")
fig.update_layout(title="Visualization of population by gender per region",  xaxis_title="Regions", yaxis_title="Population", legend_title="Gender")
fig.show()


# * Visualizing the difference between populations across all 4 regions for males and females.
# * Observable that the population is approximately equal in gender
# * The G7 seems to have 4x more persons that the 2nd highest populous region.
# 

# In[ ]:


fig = px.bar(yearXtract.sort_values(by="suicide_100"), y="suicide_100", x="country_group", color="sex", barmode="stack")
fig.update_layout(title="Visualization of suicides per 100k pop by gender per region",  xaxis_title="Regions", yaxis_title="Suicides per 100k Population", legend_title="Gender")
fig.show()


# * Even though the caribbean region is previously seen with the lowest population, they have the second highest suicides per 100k population, suggestive that population size may not directly affect suicide rates.
# * Women are observed to suicide far times less than males accross all regions. A ratio of approximately 1(female):3(males)*

# In[ ]:


fig = px.histogram(yearXtract.sort_values(by="gdp_capita"), x='country_group', y='gdp_capita', histfunc="sum", height=500)
fig.update_layout(title="Visualization of GDP per Capita per Region", xaxis_title="Countries", yaxis_title="GDP per Capita")
fig.show()


# * The caribbean has the lowest gdp per capita, lowest population and the second highest suicide rate across all 4 regions.
# * GDP per capita may not be a good indicator of increase suicide rates as the regions with the lowest and highest gdp per capita have almost equal suicides per 100k population*

# In[ ]:


fig = px.bar(ageXtract, x=ageXtract['age'], y=ageXtract['suicide_100'], color=ageXtract['sex'], barmode='group', hover_data=['country_group','generation'], height=500)
fig.update_layout(title_text='Bar chart Showing comparison of male & female suicide per 100k population for each age group',xaxis_title="Age",yaxis_title="Suicides per 100k Population", legend_title="Gender")
fig.show()


# *Age groups 45-54, 55-74, 75+ show the highest tendency to suicide*

# * As population rises, suicide rate declines, suicide numbers(totals) increase and GDP per Capita increases.

# In[ ]:


fig = px.bar_polar(ageXtract, r="population", theta="generation",
                   color="country_group", template="none",color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title="Visualization of population per generation per country group", legend_title="Regions")
fig.show()


# *Shows the population per generation and the region which they reside within the dataset, quantifying the dataset*

# *The Silent Generation is seen to have the highest suicides per 100k with over 400+*

# In[ ]:


fig = px.scatter(ageXtract.sort_values(by=["population","gdp_capita"]), y="suicide_100", x="population", color="country_group", size="gdp_capita", hover_data=['sex'],
                    template="none" )
fig.update_layout(title="Graph of population vs Suicide per 100k Population vs Region vs GDP per Capita", xaxis_title="Population",yaxis_title="Suicides per 100k Population", legend_title="Region")
fig.show()


# * Graph shows that the caribbean has the lowest GDP & population an average of 30 suicides per 100k Population
# * The highest suicide per 100k population is from a G7 country but a majority of data-points are less than 20 suicides per 100k
# * Indo-pacific countries with a population greater than 1 million and less than 6 million average less than 20 suicides per 100k

# In[ ]:


fig = px.scatter(ageXtract, y="suicide_100", x="population", color="generation", size="gdp_capita",
                    template="none", hover_data=['sex'] )
fig.update_layout(title="Graph of Population vs Suicide per 100k Population vs Generation vs GDP per Capita", xaxis_title="Population",yaxis_title="Suicides per 100k Population", legend_title="Region")
fig.show()


# * G.I generation has the highest suicide rate
# * Boomers are typically below 30 suicides per 100k population

# * Regardless of GDP per Capita women maintain approximately 10 suicides per 100k
# * Males tend to have random distribution from 10 - 30 suicides regardless of GDP however, for suicides greater than 30 per 100k males below 50k GDP per Capita tend to dominate.

# ### Custom dataset to create add a bit of dimension to the story.
# **This data is not a part of the original dataset, it was sourced through various websites**
# 
# This data may cause some triggering in persons who may have had bad experiences in the past in regards to suicide.
# 
# The purpose of this investigation was to i

# In[ ]:


customDataCsv = """Countries,Land Mass(sq Km),Happiness,divorce per 1000,Prisoners,Rapes per Million,Meat Consumption Per Capita kg/person,Annual Cannabis prevalance %,"Government expenditure on education, total (% of GDP)",Natural Disaster Risk Index,Life Expectancy ,Overall BMI
Australia,7686850,7.228,2.19,22492,289.05,108.2,10.6,5.3,4.28,79.2,27.2
Canada,9976140,7.278,2.11,35519,16.88,100.8,12.6,5.5,2.57,78.5,27.2
Cuba,110860,4,2.63,0,0,31.4,6.004,12.9,5.99,76.1,26.2
Denmark,43094,7.6,2.81,3435,72.42,112.1,5.5,7.6,2.86,76.8,25.3
Finland,337030,7.769,2.4,3433,152.52,68.4,3.1,6.9,2.06,76.8,25.9
France,547030,6.592,1.99,56957,156.22,98.4,8.6,5.4,2.76,77.5,25.3
Germany,357021,6.985,2.29,74904,94.45,82.2,4.8,4.8,2.96,77.5,26.3
Guyana,214970,4,0.626,1507,148.83,33.5,2.6,6.3,9.02,63.1,26.3
Iceland,103000,7.595,1.62,104,244.9,84.7,3.4,7.5,1.56,80,25.9
Ireland,70280,7.021,0.735,0,107.05,102.5,6.3,3.7,4.15,77.6,27.5
Italy,301230,6.223,0.895,55670,76.57,92.4,14.6,3.8,4.74,78.9,26
Jamaica,10831,5.89,0.888,4744,247.3,57.3,9.9,5.4,12.89,73.3,27.4
Japan,377835,5.886,1.84,69502,10.11,44.7,0.1,3.6,13.57,79.3,22.6
Luxembourg,2586,7.09,2.46,341,116.65,112.2,7.6,4,2.7,78,26.5
New Zealand,268680,7.307,1.94,5968,258.48,104.7,14.6,6.4,4.28,78.6,27.9
Norway,324220,7.554,1.98,2914,191.85,61.9,4.6,8,2.28,78.5,26
Philippines,300000,5.631,1.829230769,70383,63.26,30.4,0.8,2.7,24.32,66.6,23.2
Puerto Rico,9104,4,3.95,0,0,74.572,4.9,6.1,5.631153846,75.8,26.19615385
Singapore,693,6.262,1.3,16310,26.81,74.572,0.004,2.9,2.85,78.8,23.7
Suriname,163270,4,1.26,1933,451.97,40.3,4.3,5.576923077,9.25,70.1,27.4
Sweden,449964,7.343,2.46,5920,635.52,74.7,1.2,7.7,2,79.2,25.8
Malta,316,6.726,0.101,283,0,75.9,4.5,5.3,0.72,77.1,27.2
Netherlands,41526,7.488,2.05,16930,92.08,78.9,5.4,5.5,7.71,78.5,25.4
Thailand,514000,6.008,0.485,213815,69.82,25.8,1.2,4.1,6.86,70.9,24.1
Trinidad and Tobago,5128,6.192,1.67,4794,186.76,37.2,4.7,3.1,6.7,66.2,28.7
United Kingdom,244820,7.054,2.07,78753,153.2130769,80.4,6.5,5.5,3.61,77.8,27.3
United States,9629091,6.892,2.81,2020000,274.04,125.3,13.7,5,3.72,75.9,28.8
,https://data.mongabay.com/igapo/world_statistics_by_area.htm,https://countryeconomy.com/demography/world-happiness-index,"https://www.nationmaster.com/country-info/stats/People/Marriage,-divorce-and-children/Total-divorces-per-thousand-people",https://www.nationmaster.com/country-info/stats/Crime/Prisoners,https://www.nationmaster.com/country-info/stats/Crime/Violent-crime/Rapes-per-million-people,http://chartsbin.com/view/12730,https://www.nationmaster.com/country-info/stats/Crime/Drugs/Annual-cannabis-use&https://dataunodc.un.org/drugs/prevalence_table-2017,https://data.worldbank.org/indicator/SE.XPD.TOTL.GD.ZS,http://weltrisikobericht.de/wp-content/uploads/2016/08/WorldRiskReport_2011.pdf,https://www.cia.gov/library/publications/the-world-factbook/rankorder/2102rank.html,https://www.who.int/nmh/publications/ncd-status-report-2014/en/
"""


# In[ ]:


#Importing custom data and making it numeric
custData = pd.DataFrame([x.split(',') for x in customDataCsv.split('\n')])
custDataHdr = custData.iloc[0]
custData = custData[1:28]
custData.columns = ['countries','land_mass','happiness_index','divorce_per_1000','total_prisoners','rapes_million','meat_kg','cannabis_use','education_gdp','ndisaster_risk','life_exp','bmindex','country_group']
custData['country_group'] = custData["countries"].apply(groupCountry)
custData['land_mass']= pd.to_numeric(custData['land_mass'])
custData['happiness_index']= pd.to_numeric(custData['happiness_index'])
custData['divorce_per_1000']= pd.to_numeric(custData['divorce_per_1000'])
custData['rapes_million']= pd.to_numeric(custData['rapes_million'])
custData['cannabis_use']= pd.to_numeric(custData['cannabis_use'])
custData['education_gdp']= pd.to_numeric(custData['education_gdp'])
custData['life_exp']= pd.to_numeric(custData['life_exp'])
custData['bmindex'] = pd.to_numeric(custData['bmindex'])
custData = custData.sort_values(by='countries')


# In[ ]:


countryXtract = combined_df.groupby(['country'], as_index=False)['suicide_100','population','gdp_capita'].mean(), #Group and extract data, average suicides per 100.
countryXtract = countryXtract[0].sort_values(by='country', ascending=True)


# In[ ]:


custData['suicide_100'] = countryXtract.suicide_100
custData['population'] = countryXtract.population
custData['gdp_capita'] = countryXtract.gdp_capita


# In[ ]:


fig = px.bar(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', color='happiness_index', barmode="group", height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country based on happiness index',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Happiness Index")

fig.show()


# * There's no indication that happiness influences suicides per 100k

# In[ ]:


fig = px.bar(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', color='rapes_million', barmode="group", height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country for rapes per million',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Rapes per Million")

fig.show()


# Lower rapes per million tend to average higher suicides per 100k pop

# In[ ]:


fig = px.scatter(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', size='divorce_per_1000', color='country_group', height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country for each Divorce per 1000 pop group',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Regions")

fig.show()

* Rich european countries seem to have proportional suicides to divorces
* There seems to be little significance of divorce rates to suicides per 100k
# In[ ]:


fig = px.scatter(custData.sort_values(by="bmindex"), x='countries', y='bmindex', size='suicide_100', color='country_group', height=500)
fig.update_layout(title_text='Chart showing comparison Body Mass Index by country by suicide per 100k pop for each group',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", legend_title="Body Mass Index")

fig.show()


# * Based on the chart a suggest that most persons across the 4 regions who commited suicide were considered overweight with a BMI between 25 and 29.9

# In[ ]:


fig = px.bar(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', color='cannabis_use', height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country vs Cannabis Use(Annual %)',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Cannabis Use(%)")

fig.show()


# * Higher cannabis use suggests lower suicidal rates.

# In[ ]:


fig = px.bar(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', color='life_exp', height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country vs Life Expectancy at Birth(Years)',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Life Expectancy(Years)")

fig.show()


# There seems to be no real correlation between life expectancy and suicides per 100k

# In[ ]:


fig = px.bar(custData.sort_values(by="suicide_100"), x='countries', y='suicide_100', color='education_gdp', height=500)
fig.update_layout(title_text='Bar chart Showing comparison suicide per 100k population by country vs % of GDP Spent on Education',xaxis_title="Countries",yaxis_title="Suicides per 100k Population", 
                  legend_title="Education GDP(%)")

fig.show()


# * * * Lower suicide rates from lower spend?

# In[ ]:


custData.corr()


# A view of correlation between the various columns.

# ## Data Transformation

# In[ ]:



LabelEncoder()


# In[ ]:


def create_label_encoder_dict(combined_df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    for column in combined_df.columns:

        if not np.issubdtype(combined_df[column].dtype, np.number) and column != 'gdp_year' and 'gdp_capita' and 'suicide_100':
            label_encoder_dict[column]= LabelEncoder().fit(combined_df[column])
    return label_encoder_dict


# In[ ]:


transform_df = combined_df['age']
transform_df_encoder = LabelEncoder()
transform_df_encoder.fit(transform_df)


# In[ ]:


age_sample = combined_df['age'][24722]
age_sample1 = combined_df['age'][4932]

print("Converting "+age_sample)
print("Converting "+age_sample1)

transform_df_encoder.transform([age_sample,])
transform_df_encoder.transform([age_sample1,])


# In[ ]:


label_encoders = create_label_encoder_dict(combined_df)
print("Encoded Values for each Label")
print("="*60)
for column in label_encoders:
    print("="*60)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_))
    print(pd.DataFrame([range(0, len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']   ).T)


# In[ ]:


age_column = 'age'
label_encoders[age_column].classes_


# In[ ]:


pd.DataFrame([range(0, len(label_encoders[age_column].classes_))], columns=label_encoders[age_column].classes_, index=['Encoded Values']).T


# In[ ]:


transform_data = combined_df.copy()
for column in transform_data.columns:
    if column in label_encoders:
        transform_data[column] = label_encoders[column].transform(transform_data[column])

print ("Transformed dataset")
print("="*60)
transform_data


# ## Data Mining

# ### Data Mining - Algorithm: Decision Tree

# In[ ]:


# separate our data into dependent (Y) and independent(X) variables
X_data = transform_data[['gdp_capita', 'sex', 'generation']]
Y_data = transform_data['age']


# In[ ]:


#70/30 split code
 
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)


# In[ ]:


# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes
# to build the tree
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=200) 


# In[ ]:


# Build the classifier  by training it on the training data
clf.fit(X_train, y_train)


# In[ ]:


record1= X_data.head(1)
record1


# In[ ]:


test_Info={
    "gdp_capita":50563,
    "sex":"female",
    "generation":"Generation Z"
}
test_Info = pd.DataFrame([test_Info])
test_Info


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
test_Info_c = test_Info.copy() # create copy of initial data set
for column in test_Info_c.columns:
    if column in label_encoders:
        test_Info_c[column] = label_encoders[column].transform(test_Info_c[column])

print("Transformed data set")
print("="*60)
test_Info_c


# In[ ]:


indxName = label_encoders[age_column].classes_


# In[ ]:


indxName[clf.predict(record1)]


# In[ ]:


indxName[clf.predict(test_Info_c)]


# In[ ]:


clf.predict(test_Info_c[['gdp_capita','sex', 'generation']])


# In[ ]:


k=(clf.predict(X_test) == y_test) # Determine how many were predicted correctly


# In[ ]:


cm=confusion_matrix(y_test, clf.predict(X_test), labels=y_test.unique())
cm


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


dot_data = tree.export_graphviz(clf,out_file=None, 
                                feature_names=X_data.columns, 
                         class_names=label_encoders[Y_data.name].classes_,  
                         filled=True, rounded=True,  proportion=True,
                                node_ids=True, impurity=False,
                         special_characters=True)


# In[ ]:


graph = graphviz.Source(dot_data) 
graph


# In[ ]:


labels = ['gdp_capita', 'sex', 'generation']
plot_confusion_matrix(cm,combined_df['age'].unique())


# # Application Interaction - Decision Tree

# In[ ]:


user_Input={
    "gdp_capita" : input("gdp_capita : "),
    "sex" : input("sex : "),
    "generation" : input("generation : ")
}
user_Input = pd.DataFrame([user_Input])
user_Input


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
user_Input_c = user_Input.copy() # create copy of initial data set
for column in user_Input_c.columns:
    if column in label_encoders:
        user_Input_c[column] = label_encoders[column].transform(user_Input_c[column])

print("Transformed data set")
print("="*60)
user_Input_c


# In[ ]:


indxName[clf.predict(user_Input_c)[0]]


# # Data Mining - Algorithm: KMeans

# In[ ]:


cluster_data = combined_df[['gdp_year', 'suicide_100']]
cluster_data.head()


# In[ ]:


cluster_data.plot(kind='scatter', x='suicide_100', y='gdp_year')


# In[ ]:





# In[ ]:


#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values


# In[ ]:


# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="random", n_init=50, max_iter=5000) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[ ]:


help(kmeans)


# In[ ]:


kmeans = KMeans(n_clusters=6, init="random", n_init=50, max_iter=5000) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data


# In[ ]:


cluster_data['cluster'].unique()


# In[ ]:


cluster_data['cluster'].value_counts()


# In[ ]:


sns.pairplot( cluster_data, hue='cluster')


# In[ ]:


cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Individuals who commit suicide across groups')


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data


# In[ ]:


grouped_cluster_data.describe()


# In[ ]:


grouped_cluster_data.plot(subplots=True,)


# In[ ]:


# separate our data into dependent (Y) and independent(X) variables
KMeansX_data = transform_data[['gdp_year', 'suicide_100']]
KMeansY_data = transform_data['age']


# In[ ]:


#70/30 split code
 
kmeans_X_train, kmeans_X_test, kmeans_y_train, kmeans_y_test = train_test_split(KMeansX_data, KMeansY_data, test_size=0.30)


# In[ ]:


# Build the classifier  by training it on the training data
kmeans.fit(kmeans_X_train, kmeans_y_train)


# In[ ]:


kmeans_record = KMeansX_data.sample()
kmeans_record


# In[ ]:


test_KMeans={
    "gdp_year":6.539299e+12,
    "suicide_100":3.6
}
test_KMeans = pd.DataFrame([test_KMeans])
test_KMeans


# In[ ]:


kmeans.predict(kmeans_record)


# In[ ]:


indxName[kmeans.predict(kmeans_record)[0]]


# In[ ]:


kmeans.predict(test_KMeans)


# In[ ]:


indxName[kmeans.predict(test_KMeans)[0]]


# In[ ]:


kmeans.score(kmeans_X_test, kmeans_y_test)


# # Application Interaction - K-Means

# In[ ]:


KMeans_user_Input={
    "gdp_year" : input("gdp_year : "),
    "suicide_100" : input("suicide_100 : ")
}
KMeans_user_Input = pd.DataFrame([KMeans_user_Input])
KMeans_user_Input


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
KMeans_user_Input_c = KMeans_user_Input.copy() # create copy of initial data set
for column in KMeans_user_Input_c.columns:
    if column in label_encoders:
        KMeans_user_Input_c[column] = label_encoders[column].transform(KMeans_user_Input_c[column])

print("Transformed data set")
print("="*60)
KMeans_user_Input_c


# In[ ]:


indxName[kmeans.predict(KMeans_user_Input_c)[0]]


# # Data Mining - Algorithm: Neural network

# In[ ]:


# Create an instance of linear regression
nn_reg = MLPRegressor()


# In[ ]:


nn_reg.fit(X_train,y_train)


# In[ ]:


nn_reg.n_layers_ # Number of layers utilized


# In[ ]:


# Make predictions using the testing set
test_predicted = nn_reg.predict(X_test)
test_predicted


# In[ ]:


nn_test = X_test.copy()
nn_test['predicted_suicide_age']=test_predicted
nn_test['suicide_age']=y_test
nn_test.head()


# In[ ]:





# In[ ]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))


# In[ ]:


print("Root Mean Squared Error: %.3f" % np.sqrt(mean_squared_error(y_test, test_predicted)))


# In[ ]:


# Explained variance score: 1 is perfect prediction
# R squared
print('Variance score: %.2f' % r2_score(y_test, test_predicted))


# In[ ]:


def create_min_max_scaler_dict(df):
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler_dict = {}
    for column in df.columns:
        # Only create encoder for categorical data types
        if np.issubdtype(df[column].dtype, np.number):
            min_max_scaler_dict[column]= MinMaxScaler().fit(pd.DataFrame(df[column]))
    return min_max_scaler_dict


# In[ ]:


min_max_scalers = create_min_max_scaler_dict(combined_df)
print("Min Max Values for each Label")
print("="*32)
min_max_scalers


# In[ ]:


#retrieving a scaler
suicide_100_scaler=min_max_scalers['suicide_100']


# In[ ]:


suicide_100_scaler


# In[ ]:


suicide_100_scaler.data_max_


# In[ ]:


suicide_100_scaler.data_min_


# In[ ]:


# Range = Max- Min
suicide_100_scaler.data_range_ 


# In[ ]:


pd.DataFrame([
    {
        'column':col,
        'min':min_max_scalers[col].data_min_[0], 
        'max':min_max_scalers[col].data_max_[0], 
        'range':min_max_scalers[col].data_range_[0] }  for col in min_max_scalers])


# In[ ]:


# Apply each scaler to the data set to obtain transformed values
scaler_data = transform_data.copy() # create copy of initial data set
for column in scaler_data.columns:
    if column in min_max_scalers:
        scaler_data[column] = min_max_scalers[column].transform(pd.DataFrame(scaler_data[column]))

print("Transformed data set")
print("="*32)
scaler_data.head(15)


# In[ ]:


# separate our data into dependent (Y) and independent(X) variables
X2_data = scaler_data[['gdp_capita','sex','generation']]
Y2_data = scaler_data['age']


# In[ ]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2_data, Y2_data, test_size=0.30)


# In[ ]:


# Create an instance of linear regression
reg2 = MLPRegressor(hidden_layer_sizes=(150,100,63,),learning_rate_init=0.001,max_iter=400)
reg2.fit(X2_train,y2_train)


# In[ ]:


reg2.n_layers_


# In[ ]:


# Make predictions using the testing set
test2_predicted = reg2.predict(X2_test)
test2_predicted


# In[ ]:


print("Mean squared error: %.5f" % mean_squared_error(y2_test, test2_predicted))


# In[ ]:


print("Root Mean Squared Error: %.3f" % np.sqrt(mean_squared_error(y2_test, test2_predicted)))


# In[ ]:


# Explained variance score: 1 is perfect prediction
# R squared
print('Variance score: %.2f' % r2_score(y2_test, test2_predicted))


# In[ ]:


sns.set(style="whitegrid")


# In[ ]:


sns.residplot(y2_test, test2_predicted,  color="g")
plt.title("Residual Plot")
plt.ylabel("Error")


# In[ ]:


nn_records = X2_test.sample()
nn_records


# In[ ]:


nn_input_Info={
    "gdp_capita":39233,
    "sex":"male",
    "generation":"Millenials"
}
nn_input_Info = pd.DataFrame([nn_input_Info])
nn_input_Info


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
nn_input_Info_c = nn_input_Info.copy() # create copy of initial data set
for column in nn_input_Info_c.columns:
    if column in label_encoders:
        nn_input_Info_c[column] = label_encoders[column].transform(nn_input_Info_c[column])

print("Transformed data set")
print("="*60)
nn_input_Info_c


# In[ ]:


# Apply each scaler to the data set to obtain transformed values
nn_input_copy = nn_input_Info_c.copy() # create copy of initial data set
for column in nn_input_copy.columns:
    if column in min_max_scalers:
        nn_input_copy[column] = min_max_scalers[column].transform(pd.DataFrame(nn_input_copy[column]))

print("Transformed data set")
print("="*60)
nn_input_copy


# In[ ]:


reg2.predict(nn_records)


# In[ ]:


reg2.predict(nn_input_copy)


# # Interactive Application - Artificial Neural Network

# In[ ]:


nn_user_Input={
    "gdp_capita":input("gdp_capita : "),
    "sex":input("sex : "),
    "generation":input("generation : ")
}
nn_user_Input = pd.DataFrame([nn_user_Input])
nn_user_Input


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
nn_user_Input_c = nn_user_Input.copy() # create copy of initial data set
for column in nn_user_Input_c.columns:
    if column in label_encoders:
        nn_user_Input_c[column] = label_encoders[column].transform(nn_user_Input_c[column])

print("Transformed data set")
print("="*60)
nn_user_Input_c


# In[ ]:


# Apply each scaler to the data set to obtain transformed values
nn_user_copy = nn_user_Input_c.copy() # create copy of initial data set
for column in nn_user_copy.columns:
    if column in min_max_scalers:
        nn_user_copy[column] = min_max_scalers[column].transform(pd.DataFrame(nn_user_copy[column]))

print("Transformed data set")
print("="*60)
nn_user_copy


# In[ ]:


reg2.predict(nn_user_copy)


# In[ ]:


np.round(reg2.predict(nn_user_copy))


# In[ ]:


indxName[np.round(reg2.predict(nn_user_copy)[0].astype('int64'))]


# **What descriptive questions would you like to answer from your data set?**
# * Which generation had the highest frequency of suicides based on the year?
# * Which gender committed a higher number of suicides based on GDP per capita?
# * What is the average age of persons who committed suicide in each country for every 5 years?
# 
# **What data mining questions would you like to answer from the data?**
# * Questions will guide your project. Try to be as specific as possible and ensure that you may be able to answer the questions from your data using a data mining algorithm of your choice. Ex. Can we determine Y from A, B and C? What groups can we identify using A, B, C and D and what do these groups tell us?
# * Can the age be determined by the gender, generation and GDP per capita?
# * Can the age be determined by the suicide rate and the GDP per year?

# In[ ]:




