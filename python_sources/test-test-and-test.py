#!/usr/bin/env python
# coding: utf-8
This is a reconstruction of the "Test, Test and Test" notebook created by Skylord.
The only addition is the construction of a ratio "control_Metric" = disease prevalence as positive tests/ million)/testing ratio
where testing ratio is total tests done/positive tests.

A high diease prevalence divided by a low testing ratio produces a exxagerated value that identifies regiions with poor control
Regions with either good control or minimal spread of the epidemic in their region will have relatively low control_Metric values.

Many factors determine the course of an epidemic spread in a region.  Aggressive testing before the virus spreads through the population
facilitates public health measure that can control the epidemic

These include:

    availability of test kits
    population density
    age distribution
    anount of travel,travel patterns and transportation conveyance type
    transportation ports, airport hubs, seaport hubs, rail hubs and highways connections
    
    public health response:
        **testing**
        contact tracing
        isolation of contacts
        quarrantine
        social distancing
        lockdown
        travel restriction
        border closing
        


From Skylord's original award winning submission "Test, Test and Test"
Following countries have a high ``Tests/Positive (Confirmed Cases)``

|Country/Region | Tests/Positive |
| --- | --- |
|Canada-NW territories | 833 |
|UAE | 385 |
|Russia | 292 |


+ All Canada territories have high ``Tests/Positive (Confirmed Cases)`` 
+ For the following countries the ratio is 

|Country/Region| Tests/Positive |
| --- | --- |
|Australia| 53|
|Singapore| 69|
|Taiwan| 98|
|Hong-Kong| 132|  

*Assumptions & Limitations*
1. Countries have different testing strategies.
2. Testing strategy is often a function of the

   a. country's preparedness for the disease,
   
   b. availability of testing kits and labs and often
   
   c. political will!
   
3. The data has been collated from multiple sources, hence it's as good as the data collected
4. Sources of the data points are cited in the [data file](https://www.kaggle.com/skylord/covid19-tests-conducted-by-country)
5. Highly connected cities/countries will have a larger incidence of the disease
# In[ ]:


# Loading the basic libraries

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Listing files in the input directory.
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load plotly related packages
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


# Read the required dataset
# I have read the 31st March version of the dataset. Kindly read the latest version 
testsC = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_23May2020.csv')
print(testsC.shape)
#testsC.head()


# In[ ]:


#list(testsC.columns.values)


# In[ ]:


#Define additional columns   
# Pre-existing columns Tests" = absolute tests, "Positive"= absol positive tests, "Tests/ million"= freq of testing
# New calculated columns:  "pop"= region population in millions, "dp"= disease prevalence = Positive/pop = positive tests/million population, 
# "test_Ratio"= Tests/Positive
testsC['pop'] = testsC['Tested']/testsC['Tested\u2009/millionpeople']
testsC['dp'] = testsC['Positive']/testsC['pop']
testsC['test_Ratio'] =testsC['Tested']/testsC['Positive']
#testsC.head()


# In[ ]:


import plotly.graph_objs as go
testsC.sort_values(by=['Tested\u2009/millionpeople'], ascending=False, inplace=True)
trace1 = go.Bar(
                x = testsC['Country'],
                y = testsC['Tested\u2009/millionpeople'],
                name = "Disease Prevalence/Testing Ratio",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = testsC['Country'])

data = [trace1]
             
layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **TESTS PER MILLION POPULATION** Just looking at the number of tests done per million population doesn't inform you of the effectiveness of the testing in controlling a pandemic.   This led Skylord to explore the parameter "Tests Done"/"Positive Tests", since this carried more information.

# In[ ]:


# Create  scatter plot for all countriesshowing  tests done per million of population versus 
# prevalence of disease as positive tests per million population

trace1 = go.Scatter(
                    y = testsC['Tested\u2009/millionpeople'],
                    x = testsC['dp'],
                    mode = "markers",
                    name = "Country",
                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),
                    text= testsC['Country'])

data = [trace1]
layout = dict(title = 'Tests Done per Million Populaton versus Disease Prevalence',
              xaxis= dict(title= 'Disease Prevalence (Positive Tests/Million Population' ,ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Tests Done per million population',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
                    


# **Frequency of Testing versus Disease Prevalence**  This chart might show a mild positive relationship between testing and disease prevalence with regions with
# greater disease prevalence testing more than regions with lower disease prevalence.  Unfortunately, if you don't test you don't find disease, so area that don't
# test aggresively may appear to have low disease prevalence but this could be the result of not testing aggressively.

# In[ ]:


# Create  scatter plot for all countriesshowing  testing ratio versus 
# prevalence of disease as positive tests per million population

trace1 = go.Scatter(
                    y = testsC['test_Ratio'],
                    x = testsC['dp'],
                    mode = "markers",
                    name = "Country",
                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),
                    text= testsC['Country'])

data = [trace1]
layout = dict(title = 'Testing Ratio versus Disease Prevalence',
              xaxis= dict(title= 'Disease Prevalence (Positive Tests/Million Population' ,ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Tests Done/ Positive Tests',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
                    


# **TESTING RATIO VERSUS DISEASE PREVALENCE**  This separates some extremes regions but does not separate them very well.  
# Those regions with Disease Prevalence above 2000 have very low testing ratio and those with testing ratio above 200 have very low Disease Prevalence.

# In[ ]:


testsC['inverse_Test_Ratio']= testsC['Positive']/testsC['Tested']
testsC.sort_values(by=['inverse_Test_Ratio'], ascending=False, inplace=True)
trace1 = go.Bar(
                x = testsC['Country'],
                y = testsC['test_Ratio'],
                name = "Disease Prevalence/Testing Ratio",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = testsC['Country'])

data = [trace1]
             
layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# Countries sorted by testing ratio defined as (tests done)/(positive test results)
# On the right side of the chart are regions with high testing ratios.  These are regionss with low infection burden.
# The left side of the chart shows regions with low testing ratios.  These regions have high disease burden.
# High testing ratios facilitate infection control.

# In[ ]:


#  Define a variable "control_Metric" = disease prevalence/testing ratio = testsC['dp']/testsC['test_Ratio']
#  If disease prevalence is high and testing ratio is low, the z variable will be very high --> poor control
#  If disease prevalence is low and testing ratio is high, the z variable will be very low --> good control
testsC['control_Metric']= testsC['dp']/testsC['test_Ratio']

testsC.sort_values(by=['control_Metric'], ascending=True, inplace=True)
trace1 = go.Bar(
                x = testsC['Country'],
                y = testsC['control_Metric'],
                name = "Control Metric = Disease Prevalence/Testing Ratio",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = testsC['Country'])

data = [trace1]
             
layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# CONTROL METRIC FOR VARIOUS REGIONS  
# Control metric is (disease prevalence/testing ratio)  Regions with high disease prevalence and low testing ratio will score high on the Control Metric.  Regions with extensive spread of disease are towards the right side of the chart.

# In[ ]:


#  Try plotting the control_Metric versus disease prevalence
# Create first scatter plot for all countries
trace1 = go.Scatter(
                    x = testsC['dp'],
                    y = testsC['control_Metric'],
                    mode = "markers",
                    name = "Country",
                    marker = dict(color = 'rgba(255, 50, 50, 0.8)'),
                    text= testsC['Country'])

data = [trace1]
layout = dict(title = ' Control Metric versus Disease Prevalence',
              xaxis= dict(title= 'Disease Prevalence as positve tests/million population',ticklen= 5,zeroline= False),
              yaxis= dict(title= ' Control Metric defined as Disease Prevalence/Testing Ratio',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
                    


# # **Control Metric Versus Disease Prevalence for all Regions**
# 
# **The Y axis represents the "Control Metric" = Disease Prevalence/Testing Ratio** 
# 
#  The left side of the chart shows regions with a low Control Metric.  This corresponds to a region with reltively aggressive testing and either control of the epidemic or a country in the very early stages of the epidemic with a low burden of infection.  
#  
# 
# The right side of the chart shows regions with a high Control Metric.  This correspondes to a region with inadequate testing relative to the current level of infection and/or they are late in the course of the epidemic and the infection is already widely spread throughout the population. 
# 
# All but one (Iran) of the regions with a Disease Prevalence < 2000/Million have a Control Metric < 200.
# A majority of regions with a Disease Prevalence > 2000 have a Control Metric > 200.

# Countries which were pro-active in prepaing for the COVID-19 pandemic,were able to control the spread & have a lower mortality rate. Their labs were preparing testing kits as far back as in early Jan [link to news article](https://www.nytimes.com/2020/03/20/world/europe/coronavirus-testing-world-countries-cities-states.html)  Countries like Taiwan had prepared for testing and contact tracing.  The national government ordered protective equipment and distributed it to hospitals. 
# 
# Please also read Nate Silver's critique on how the [number of case counts is meaningless](https://fivethirtyeight.com/features/coronavirus-case-counts-are-meaningless/) unless it's correlated with the testing strategy. 
# 
# 
# 1. Iceland is a large outlier in terms of aggressive testing, having conducted more than 45k+ tests/ million. But, they have a relatively high level of disease infection, calculated as 2,981 cases/million.  They performed 15 tests for every positive case that they find.  They are testing a lot, but they also have a fair amount of disease prevalence so the aggressive testing may be appropriate.
# 
# 2. By comparison, United Arab Emirates (UAE) performs 385 tests to find one positive case, but they have a very low level of infection, 59 positive cases/million.  They are testing very aggressively and compared to the Disease Prevalence,  UAE is over-testing.
# 
# 4. South Korea (ROK) has an intermediate level of tests done/positive tests, they perform 42 tests to find one positive  case.  ROK had an outbreak of Covid-19 similar to Wuhan, China but controlled the spread of disease by implemnting aggresive public health measures.  For ROK, Tests Done/Positive test = 410564/9786 = 41.95 =42   Similarly for Singapore, Tests Done/Positive Test = 70.  Let's call Tests Done/Positive Test the "Testing Ratio".
# 
# 5. Italy has high prevalence of disease and relatively low tests done/positive tests.  In general Italy, Spain, UK, New York, New Jersey appear to have similar high levels of disease and low Testing Ratios.  For example, New York has  a Testing Ratio of 3 and a Disease Prevalence of 3,418 positive cases/million population. New York performs only 3 tests to find one positive case.  Compare this to the UAE who performs 385 tests to find one positive case!
# 
# 6. Countries with a history of experience of surviving large outbreaks of SARS in 2003 have dealt with the pandemic successfully.  These include China, Hongkong, Taiwan, Singapore, South Korea and Canada. SARS 2003 was less contagious than Covid-19 and had higher mortaility rate.  Both SARS and Covid-19 infected healthcare workers at an increased rate.  A recent article out of Taiwan suggests the Covid 19 virus is transmissable ealier in the infected person whereas SARS spread only after the onset of symptoms.  Social Network Analysis of SARS shows that Hospitals were hubs of transmission.  Regions that had previously experienced the SARS epidemic tested aggresively, have moderately high Testing Ratios and have relatively low levle of infection in their region.  This combination of adequate or aggressive testing combined with aggresive public health measures likely allowed them to control of the advance of the pandemic.
# 
# 8. Greenland is one of the least densely populated countries on earth, similar to the Sahara in Africa.  It small population (54,000) and has a relatively low level of infection.  Greenland has 11 cases or 204 cases/million.  Iceland is very tiny island compared to Greenland and is geographically located just below Greenland.  Iceland tested vigorously and Iceland has 2,981 cases/million. (Population of Greenland is about 54,000 or about 1/7th of Iceland's 360,000).  Population density matters.
# 
