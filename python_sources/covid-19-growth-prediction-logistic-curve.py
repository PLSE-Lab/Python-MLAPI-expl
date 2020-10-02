#!/usr/bin/env python
# coding: utf-8

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


# # COVID-19 - Understand Growth Patterns for Prediction 
# 
# This notebook is intended to serve as an EDA for the current Kaggle competion on global forecasting of the spread of the coronvirus.
# 
# I think it is important for everyone to understand the nature of the growth patterns of pandemics. There is an excellent Youtube video from [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) that offers a great explanation.
# 
# ### Understanding Growth Video Link
# 
# ![image.png](attachment:image.png)
# 
# https://www.youtube.com/watch?v=Kas0tIxDvrg&t=35s

# In[ ]:


#import IPython
#IPython.display.IFrame(<iframe width="650" height="400" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" title="2019-nCoV" src="/gisanddata.maps.arcgis.com/apps/Embed/index.html?webmap=14aa9e5660cf42b5b4b546dec6ceec7c&extent=77.3846,11.535,163.5174,52.8632&zoom=true&previewImage=false&scale=true&disable_scroll=true&theme=light"></iframe>)


# In[ ]:


from IPython.display import HTML

HTML('<div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://www.youtube.com/embed/jmHbS8z57yI?ecver=2" width="640" height="360" frameborder="0" style="position:absolute;width:100%;height:100%;left:0" allowfullscreen></iframe></div>')


# # Purpose of This Document
# The purpose of this notebook is to use knowledge from tracking of the COVID-19 virus in specific countries and regions that have seen severe growth rates. 
# 
# The full report is https://www.kaggle.com/wjholst/covid-19-growth-patterns-in-critical-countries
# 
# Currently, the following countries and regions are included:
# * China
# * All China excluding the original Hubei province
# * Rest of the world
# * United States
# * Italy
# * Iran
# * Spain
# * More (see below in the Change History section)
# 
# There are two phases to the growth rate, an exponential phase, and then a flattening, downward turning of the curve. This is a sigmoid curve. The sigmoid curve is include only in the cases where the inflection point has been crossed (as of 3/15/20, all China and remainder of China excluding Hubei. 
# 
# The report now includes US states which have very active growth rates.
# 
# Overall, it is hopeful that the exponential and logistic graphs will help us define the inflection point for each separate population grouping. 
# 
# ## Predictions of the California Growth Rate
# 
# The information gathered within this report is now being used to predict the California growth rate for both confirnd cases and deaths. The approach will be simple - based on observations of other countries and current California available data, we will estimate the parameters for the logistic curve to match the current rates. We will then modify the parameters systematically to generate expected, best, and worse case scenarios.
# 
# ## Observation Log
# 
# * On 3/18, South Korea was moved to the sigmoid tracking group. The sigmoid curves now converge.
# *          Italy's exponential curve began to tilt slightly. That may signal the start of an inflection point.
# *          Italy's mortality rate is extremely high and still climbing.
# *          Washington State also seems to be flattening.
# * 3/20/20 - For the third day in a row, the confirmed rate in Iran is to the right of the curve. **Iran may be reaching an inflection** point after around 25 days. **Italy also seems close** to an inflection point.
# 
# ## Change History
# 
# * 2020-03-18 - Addressed a problem with some of the curve fitting not converging. Because some of the countries, like the US, had a long period of days with no increases of cases, the tracking start date.
# * 2020-03-18 - Added US "hot" states, NY, CA, and WA. Also added Germany, which has shown rapid recent growth.
# * 2020-03-19 - Added Colorado, per friend request. Also added France and 2 high density countries, Monaco and Singapore
# * 2020-03-20 - Removed Monaco, not enough cases
# * 2020-03-21 - Added Switzerland, New Jersey, Louisiana, and 12 'hot' European countries as a group
# * 2020-03-22 - Added United Kingdom and UK to hot European group
# * 2020-03-22 - Modified to generate predicted results for California.
# 
# 
# ## About Coronavirus
# 
# * Coronaviruses are **zoonotic** viruses (means transmitted between animals and people).  
# * Symptoms include from fever, cough, respiratory symptoms, and breathing difficulties. 
# * In severe cases, it can cause pneumonia, severe acute respiratory syndrome (SARS), kidney failure and even death.
# * Coronaviruses are also asymptomatic, means a person can be a carrier for the infection but experiences no symptoms
# 
# ## Novel coronavirus (nCoV)
# * A **novel coronavirus (nCoV)** is a new strain that has not been previously identified in humans.
# 
# ## COVID-19 (Corona Virus Disease 2019)
# * Caused by a **SARS-COV-2** corona virus.  
# * First identified in **Wuhan, Hubei, China**. Earliest reported symptoms reported in **November 2019**. 
# * First cases were linked to contact with the Huanan Seafood Wholesale Market, which sold live animals. 
# * On 30 January the WHO declared the outbreak to be a Public Health Emergency of International Concern 

# # Acknowledgements
# 
# This effort was inspired by an excellent Youtube video from [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
# 
# * Video - https://www.youtube.com/watch?v=Kas0tIxDvrg&t=35s 
# * Starting kernel - https://www.kaggle.com/imdevskp/covid-19-analysis-viz-prediction-comparisons
# * https://github.com/CSSEGISandData/COVID-19
# * https://arxiv.org/ftp/arxiv/papers/2003/2003.05681.pdf
# 
# 

# # Libraries

# ### Install

# In[ ]:


## install calmap
#! pip install calmap


# ### Import Libraries

# In[ ]:


# essential libraries
import json
import random
from urllib.request import urlopen

# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
#import calmap
import folium
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

# color pallette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# html embedding
from IPython.display import Javascript
from IPython.core.display import display
from IPython.core.display import HTML


# # Dataset

# In[ ]:


# list files
#!ls ../input/corona-virus-report
# https://www.kaggle.com/imdevskp/corona-virus-report


# In[ ]:


# importing datasets


full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
sub = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')


# In[ ]:


full_table[full_table['Province/State']=='California']


# In[ ]:


train


# In[ ]:


ca_by_state = train.copy()

train.columns
ca_by_state.columns =['Id', 'Province/State', 'Country/Region', 'Lat', 'Long', 'Date',
       'Confirmed', 'Deaths']
ca_by_state = ca_by_state[ca_by_state.Date >'2020-03-09']


# The calculation of active cases is not accurate, because it does not include recovered. However it is useful in the gaussian plot.

# In[ ]:



ca_by_state['Active'] = ca_by_state.Confirmed - ca_by_state.Deaths
ca_by_state


# ## Most Recent Update

# In[ ]:


print ('Last update of this dataset was ' + str(train.loc[len(train)-1]['Date']))
print ('Last update of the studay dataset was ' + str(full_table.loc[len(full_table)-1]['Date']))


# # Preprocessing

# ## Examining the Growth Curves
# 
# These distributions start off exponentially, but eventually become a logistic curve. We can plot them both ways, and then fit a non-linear regression to the curve to determine the rate.
# 
# First we look at mortality curves. The trend to what for is an increasing mortality curve. This means that medical treatments are not controlling the virus well. This is true in Italy, which has an older population and seemed to be slow to respond in social distancing efforts. Compare Italy to South Korea, which had an agressive testing and treatment program, we see that Italy has a severe virus growth situation.
# 
# ### What these curves show
# 
# There are several groups of curves shown. They show:
# 
# * Growth rate over time - this shows the daily growth rate for each region 
# * Exponential growth for each region - there are separate plots for confirmed cases, deaths, and recovered
# * Logistic growth curves - these are for only the countries that have reached an inflection point
# 
# The growth curves also have the coefficents and errors for each coeffients. The second coefficient is the growth rate.
# 
# You may observe, at some point, where the daily arrival rates are to the right of the predicted curve. This is a good signal that the growth rate might be reaching an inflection point. Once this point is reached, the infection point, the growth rate will slow down, and the curve will be S-shaped, a sigmoid curve. This is a very good signal!
# 
# The infection point generally indicates that 50 percent of the cummulative cases have been reached.

# In[ ]:


#rates


# In[ ]:


dict = {
        'California':ca_by_state,
        #'United States': us_by_date,
}


# Next, let's review some of the grow curves.
# 
# 

# In[ ]:


def plots_by_country (country, country_name):

    temp = country

    # adding two more columns
    temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/temp['Confirmed'], 3)*100
    # temp['No. of Recovered to 1 Death Case'] = round(temp['Recovered']/temp['Deaths'], 3)
    #print (temp)

    
    #print (temp.iloc[13]['Date'])
    last_date = temp.iloc[len(temp)-1]['Date']
    death_rate = temp[temp.Date ==last_date]['No. of Deaths to 100 Confirmed Cases']
    temp = temp.melt(id_vars='Date', value_vars=['No. of Deaths to 100 Confirmed Cases', ], 
                     var_name='Ratio', value_name='Value')

    #str(full_table.loc[len(full_table)-1]['Date'])

    fig = px.line(temp, x="Date", y="Value", color='Ratio', log_y=True, width=1000, height=700,
                  title=country_name + ' Recovery and Mortality Rate Over Time', color_discrete_sequence=[dth, rec])
    fig.show()
    return death_rate, 0
        
rates = []
for key, value in dict.items():
    death_rate, recovered_rate  = plots_by_country (value,key)
    rates.append ([key,np.float(death_rate),np.float(recovered_rate)]) 


# In[ ]:


import pylab
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

def exp (x,a,b):
    y = a* np.exp(x*b)
    return y

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def growth_rate_over_time (f, country, attribute, title):
    ydata = country[attribute]
    

    xdata = list(range(len(ydata)))

    rates = []
    for i, x in enumerate(xdata):
        if i > 2:
#            print (xdata[:x+1])
#            print (ydata[:x+1])

            popt, pcov = curve_fit(f, xdata[:x+1], ydata[:x+1])
            rates.append (popt[1])
    rates = np.array(rates) 
    pylab.style.use('dark_background')
    pylab.figure(figsize=(12,8))
    xdata = np.array(xdata)
    #pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(xdata[3:]+1, 100*rates, 'o', linestyle='solid', label=attribute)
    #if fit_good:
    #    pylab.plot(x,y, label='fit')
    #pylab.ylim(0, ymax*1.05)
    #pylab.legend(loc='best')
    pylab.xlabel('Days Since Start')
    pylab.ylabel('Growth rate percentage ' + attribute)
    pylab.title(title + attribute, size = 15)
    pylab.show()
    
        
    

def plot_curve_fit (f, country, attribute, title, normalize = False, curve = 'Exp'):
    #print (country)
    #country = country[10:]
    fit_good = True
    ydata = country[attribute]
    #ydata = np.array(ydata)
    xdata = range(len(ydata))
    mu = np.mean(ydata)
    sigma = np.std(ydata)
    ymax = np.max(ydata)    
    if normalize:
        ydata_norm = ydata/ymax
    else:
        ydata_norm = ydata
    #f = sigmoid
    try:
        if curve == 'Gauss': # pass the mean and stddev
            popt, pcov = curve_fit(f, xdata, ydata_norm, p0 = [1, mu, sigma])
        else:    
            popt, pcov = curve_fit(f, xdata, ydata_norm,)
    except RuntimeError:
        print ('Exception - RuntimeError - could not fit curve')
        fit_good = False
    else:

        fit_good = True
        
    if fit_good:
        if curve == 'Exp':   
            print (key + ' -- Coefficients for y = a * e^(x*b)  are ' + str(popt))
            print ('Growth rate is now ' + str(round(popt[1],2)))
        elif curve == 'Gauss':
            print (key + ' -- Coefficients are ' + str(popt))
        else:   # sigmoid 
            print (key + ' -- Coefficients for y 1/(1 + e^(-k*(x-x0)))  are ' + str(popt))
            
        print ('Mean error for each coefficient: ' + str(np.sqrt(np.diag(pcov))/popt))
    else:
        print (key + ' -- Could not resolve coefficients ---')
    x = np.linspace(-1, len(ydata), 100)
    if fit_good:
        y = f(x, *popt)
        if normalize:
            y = y * ymax
    plt.style.use('dark_background')
    pylab.figure(figsize=(15,12)) 
    #pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(xdata, ydata, 'o', label=attribute)
    if fit_good:
        pylab.plot(x,y, label='fit')
    pylab.ylim(0, ymax*1.05)
    pylab.legend(loc='best')
    pylab.xlabel('Days Since Start')
    pylab.ylabel('Number of ' + attribute)
    pylab.title(title + attribute, size = 15)
    pylab.show()


# ## Exponential Growth Curves
# 
# There are two sets of graphs in this section. 
# 
# ### Growth Rate Percentage Over Time
# 
# The first is a plot of growth rate percentage over time. The graph starts on the 4th day because initial rates cannot be extimated. This graph is produced by generating curve fittings iteratively for the n days. This means that the a separate is calulated for days 1-4, 1-5, 1-6,...1-n. These calculated rates are then plotted over the number of days since the localized start.
# 

# In[ ]:


for key, value in dict.items():
    if key in ["China",'Rest of China w/o Hubei']:
        pass
    else:
        #growth_rate_over_time (exp, value, 'Confirmed', "Growth Rate Percentage - ")
        growth_rate_over_time (exp, value, 'Confirmed', key + ' - Growth Rate Percentage for ',)
        #growth_rate_over_time (exp, value, 'Deaths', key + ' - Growth Curve for ',)
        #growth_rate_over_time (exp, value, 'Recovered', key + ' - Growth Curve for ',False)


# ### Growth Rates of Confirmed, Deaths, 
# 
# There are three graphs in this section which show exponential growth rate of confirmed, deaths, and recovered. The head shows the current growth rate.
# 
# You can use the rule of 72 to find the doubling rate. As of March 20th, the confirmed growth rate for the United States is around 0.35. That means that the number of confirmed cases will double in just 2 days. *( 72/35 = 2.06 )*

# In[ ]:


round (72/35,2)


# In[ ]:


for key, value in dict.items():
    if key in ["China",'Rest of China w/o Hubei']:
        pass
    else:
        plot_curve_fit (exp, value, 'Confirmed', key + ' - Growth Curve for ',False,'Exp')
        plot_curve_fit (exp, value, 'Deaths', key + ' - Growth Curve for ',False,'Exp')
        #plot_curve_fit (exp, value, 'Recovered', key + ' - Growth Curve for ',False,'Exp')


# In[ ]:


#    plot_curve_fit (sigmoid, value, 'Recovered', key + ' - Logistic Growth Curve for ',True,'Logistic')


# ## Gaussian Approximation of Active Cases
# 
# The active cases should fairly clossly resemble a Gaussian distribution. While the derivate of a sigmoid function is not the Gaussian function, a Gaussian distribution is a close approximation.

# In[ ]:


plot_curve_fit (gaussian, ca_by_state, 'Active', 'California' + ' - Curve for Cases ',False,'Gauss')


# # Logistic Prediction
# 
# Based on observations from our previous analysis, we derived logistic curve parameters of 
# * k = .25
# * X0 = 34
# 
# The first date of recorded confirmations was 3/10/2020. We will model from that point.
# 

# In[ ]:


x0 = 33
k = 0.27
kd = 0.3
x0_death = 35
results = [] 
total_estimated = 500000
total_deaths = total_estimated * 0.15
for x in range(1,44):
    conf = int(total_estimated * sigmoid(x, x0, k))
    deaths = int(total_deaths * sigmoid(x, x0_death, kd))
    print ('Confirmed estimate for day ' + str(x) + ' - ' + str(conf))
    print ('Death estimate for day ' + str(x) + ' - ' + str(deaths))
    results.append([x,conf,deaths])


# In[ ]:


ca_by_state


# In[ ]:


r = pd.DataFrame(results)
r.columns = sub.columns
sub = r.copy()
sub


# In[ ]:


sub.to_csv("submission.csv", index=False)


# https://www.kaggle.com/imdevskp/mers-outbreak-analysis  
# https://www.kaggle.com/imdevskp/sars-2003-outbreak-analysis  
# https://www.kaggle.com/imdevskp/western-africa-ebola-outbreak-analysis
# 
