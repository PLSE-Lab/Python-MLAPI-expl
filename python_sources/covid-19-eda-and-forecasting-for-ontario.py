#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/covidpic/covid2.jpg",width=800)


# <h1>Exploratory Data Analysis & Forecasting on Covid-19 global and regional data</h1> 
#     
# * [Introduction](#section-one)
# * [EDA: Global](#section-two)
# * [Forecasting: Ontario](#section-three)

# <a id="section-one"></a>
# # Introduction
# In this notebook, I am going to explore the Covid-19 dataset, with the goal of drawing insights and building some hypotheses to be tested to help answer some of the most pondered questions in the last few months: 
# 
# 1. What makes a certain region more vulnerable to Covid infections? 
# 2. What measures need to be taken in order for the infection rate to decrease?

# <a id="section-two"></a>
# # EDA: Global
# Let's start by importing the relevant libraries and loading the pre-processed data into global and country specific dataframes.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


# We'll save separate dataframes for 4 selected countries to take a closer look at: Brazil, Canada, China, Italy. The approach of these countries in dealing with the pandemic has been fairly disparate, so there is a lot of room for comparisons.

# In[ ]:


brazil_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/brazil_province_wise.csv", parse_dates=['Date'])
canadian_provinces_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/canada_province_wise.csv", parse_dates=['Date'])
china_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/china_province_wise.csv", parse_dates=['Date'])
italy_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/italy_province_wise.csv", parse_dates=['Date'])
global_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/covid_19_clean_complete.csv", parse_dates=['Date'])

brazil_df.set_index(brazil_df['Date'],drop=True,inplace=True)
canadian_provinces_df.set_index(canadian_provinces_df['Date'],drop=True,inplace=True)
china_df.set_index(china_df['Date'],drop=True,inplace=True)
italy_df.set_index(italy_df['Date'],drop=True,inplace=True)

brazil_df.drop(['Date'],axis=1, inplace=True)
canadian_provinces_df.drop(['Date'],axis=1, inplace=True)
china_df.drop(['Date'],axis=1, inplace=True)
italy_df.drop(['Date'],axis=1, inplace=True)

canada_df = canadian_provinces_df.groupby(['Date']).sum()


# **Note that some countries, such as Canada, have data split among provinces. So, if we need to analyze the country as a whole, it's important to group the provincial data, taking the sum of values within each province. That can be easily done with Pandas.**
# 
# Now, we can take a look at the loaded dataframes, and inspect their structure:

# In[ ]:


global_df.info()


# In[ ]:


canadian_provinces_df.tail()


# We're ready to make some plots. First, we can compare the number of confirmed cases, deaths, and recoveries throughout the observed months for each of the 4 selected countries:

# In[ ]:


plt.rcParams.update({'font.size': 15})
with plt.style.context('seaborn-white'):
    fig, ax = plt.subplots(2,2, figsize=(16,11))
    
brazil_df[['Confirmed','Deaths','Recovered']].plot(ax=ax[1,0],linestyle='--', linewidth=2.5)
canada_df[['Confirmed','Deaths']].plot(ax=ax[0,0], sharex=ax[0,0],linestyle='--', linewidth=2.5)
china_df.groupby(['Date']).sum()[['Confirmed','Deaths','Recovered']].plot(ax=ax[0,1],linestyle='--', linewidth=2.5)
italy_df[['Confirmed','Deaths','Recovered']].plot(ax=ax[1,1], sharex=ax[0,1],linestyle='--', linewidth=2.5)

ax[0,0].set_title('Canada')
ax[1,0].set_title('Brazil')
ax[0,1].set_title('China')
ax[1,1].set_title('Italy')

def make_yticklabel(tick_value, pos): 
    return "{}K".format(tick_value / 1000)

from matplotlib.ticker import FuncFormatter 
ax[0,0].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))
ax[1,0].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))
ax[0,1].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))
ax[1,1].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))

plt.tight_layout()


# From the curves above, it is obvious that Italy has recorded the highest amount of confirmed cases and deaths among the 4 regions. Brazil, on the other hand, displays the sharpest slope for the curve of confirmed cases. So, Brazil may outnumber Italy within a couple of months.
# 
# Differently from the 3 other countries, China has been displaying a plateau of confirmed cases since March. This could be because the pandemic started much earlier there, and the number of cases is coming to a peak.
# 
# Even though our dataset doesn't have the number of recoveries in Canada, we can infer that the recovery curve is probably not much below the confirmed curve, since Canada displays s small number of deaths.
# 

# **We can also draw some conclusions from plotting the number of confirmed COVID-19 cases for the 15 countries with the highest amount of confirmations in the dataset:**

# In[ ]:


plt.rcParams.update({'font.size': 22})
df0 = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending=False)[1:15]
df0 = df0.sort_values(ascending=True)
df1 = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending=False)[0:15]
df1 = df1.sort_values(ascending=True)
with plt.style.context('seaborn-darkgrid'):
    fig, ax = plt.subplots(1,2)
    df0.plot.barh(ax=ax[0],title="COVID-19 Confirmed Cases - US excluded",figsize=(38,20),color=['black','orange','red','black','green','navy', 'red', 'green', 'red', 'navy','black','blue','green','goldenrod'])
    df1.plot.barh(ax=ax[1],title="COVID-19 Confirmed Cases including US",figsize=(38,20),color=['black','orange','red','black','green','navy', 'red', 'green', 'red', 'navy','black','blue','green','goldenrod','blue'])
    ax[0].set_ylabel(None)
    ax[1].set_ylabel(None)


# The plot above shows that the US is easily the most contaminated region in the world. It is worth noticing, however, that the US population (328.2 million) is much larger than the population of Spain (46.94 million), which accounts for the second most contaminated region in the world. China, on the other hand, occupies the 9th position.
# 
# **Below is a bar graph of recoveries and deaths for the countries with the highest number of deaths. By comparing this chart with that of confirmed cases above, a few interesting things can be noticed:**
# <li> A large number of confirmed cases do not necessarily imply a large number of deaths. Belgium, for instance, occupies the 12th position in the number of confirmed cases, but it bumps to the 6th position when it comes to the number of deaths.
# <li> By comparing the number of recoveries with the number of deaths in each country, we can observe where the Covid combat measures have been the most effective. Germany, for instance, has shown a large number of recoveries. While Germany has a larger number of contaminations than other countries, it records fewer deaths.

# In[ ]:


df = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])[['Recovered','Deaths']].sum().sort_values(by='Deaths',ascending=False)[0:11]
df = df.sort_values(by='Deaths',ascending=True)
df.drop(['Netherlands'], axis=0,inplace=True)
with plt.style.context('seaborn-poster'):
    df.plot.barh(title="COVID-19 Deaths and Recoveries",figsize=(14,6))


# **We could, instead, compare the number of deaths to the number of cases in a more direct way. By, for instance, taking the ratio of former by the latter:**

# In[ ]:


df = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])[['Confirmed','Recovered','Deaths']].sum()
df['Deaths/Confirmed'] = df['Deaths']/df['Confirmed']
df = df.sort_values(by='Deaths/Confirmed',ascending=False)[0:45]
df = df.sort_values(by='Deaths/Confirmed',ascending=True)
with plt.style.context('seaborn-poster'):
    df['Deaths/Confirmed'].plot.barh(title="COVID-19 - Deaths/Confirmed ",figsize=(20,20))
plt.xticks(rotation=30,ha='right')
plt.show()


# The plot above isn't very helpful for drawing conclusions, as there isn't a very noticeable pattern from the regions displayed. However, some interesting points can be observed. For instance, Nicaragua has been recording more than a quarter of deaths for the contaminations in its territory. Moreover, the number of deaths/cases in Belgium endorses the previous hypothesis about the country recording many deaths for not so many confirmed cases.
# 
# **Caution must be taken here, however, when assuming that the number of confirmed cases is correct for every country. Some countries haven't been able to properly test people for contamination, so that number could, in fact, be much larger.**
# 
# We can take advantage of the Province/State data for the countries that have those records in the dataset. Below is a graph of confirmed cases in Canada for the 9 provinces with the highest number of cases:

# In[ ]:


ax = canadian_provinces_df.loc['2020-04-24'][['Province/State','Confirmed']].sort_values(by='Confirmed',ascending=False)[0:9]
with plt.style.context('seaborn-poster'):
    ax.plot.bar(x='Province/State',title="COVID-19 Confirmed Cases in Canada",figsize=(15,6),color='navy')
plt.xticks(rotation=30,ha='right')
plt.show()


# Understandably, the two provinces with the highest number of inhabitants have recorded the highest number of cases: Quebec and Ontario. Observe that the provinces in the Atlantic region and the northern territories have very few recorded cases. 
# 
# A plausible explanation for this is the fact that people live much farther apart in these places, since natural obstacles are present among cities in the Atlantic, and a low density of populous towns exist within the vast amount of land in the territories. So, social distancing can be more easily practiced.

# <a id="section-three"></a>
# # Forecasting: Ontario
# 
# Now, let's jump to analysing the regional data at a deeper level, taking Ontario as the main region to be analyzed. To do that, we can use the SIR model of infections. 
# 
# According to the definition used by [WolframMathWorld](https://mathworld.wolfram.com/SIRModel.html), "A SIR model is an epidemiological model that computes the theoretical number of people infected with a contagious illness in a closed population over time. The name of this class of models derives from the fact that they involve coupled equations relating the number of susceptible people S(t), number of people infected I(t), and number of people who have recovered  R(t)."

# In[ ]:


Image("../input/covid19sir/sir.png",width=800)


# The 3 coupled differential equations of the SIR model can be expressed as:

# $\dfrac{dS}{dt} = - \beta \frac{I}{N} S$
# 
# $\dfrac{dI}{dt} = \beta \frac{I}{N} S - \gamma I$
# 
# $\dfrac{dR}{dt} =  \gamma I S$

# Where N is the population of the system (i.e. region) considered, and $\beta, \gamma$ are free parameters to be determined. One way to implement the SIR model in Python is by using the *odeint* package from scipy, which is a ODE solving suite. Let's first import the required packages and move on to building the model:

# In[ ]:


from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# The sklearn functions will be useful for testing the accuracy of the model fit. We need to include information about the population size of Ontario

# In[ ]:


# population
N_ontario = 14446515 


# Let's take a closer look at Ontario, Canada, and see what can be concluded when the SIR model is applied to data from this region. We need to do some cleaning and pre-processing first:

# In[ ]:


# ontario recoveries table
on_recov_df = pd.read_csv('/kaggle/input/ontario-recovered/ontario_recovered.csv', parse_dates=['date_recovered'])
on_recov_df = on_recov_df[on_recov_df['province']=='Ontario']
on_recov_df.drop(['province'],axis=1,inplace=True)
on_recov_df.sort_values(by='date_recovered',inplace=True)
on_recov_df.rename(columns={'date_recovered': 'Date'},inplace=True)
on_recov_df.reset_index(drop=True,inplace=True)

# joining the ontario recoveries table to the general canadian provinces table
ontario_df = canadian_provinces_df[canadian_provinces_df['Province/State']=='Ontario'].iloc[0:,:]
ontario_df = ontario_df.merge(on_recov_df,how='inner',on='Date') 
ontario_df.drop(['Recovered'],axis=1,inplace=True)
ontario_df.rename(columns={"cumulative_recovered": "Recovered"},inplace=True)

# treating missing values on the resulting dataframe, and including day count data
ontario_df.fillna(0,inplace=True)
ontario_df['day_count'] = list(range(1,len(ontario_df)+1))


# Unfortunately, the original dataset doesn't contain information about the number of recoveries in Canada, and we need it for the SIR model. So, an additional table had to be imported and properly processed (see the steps taken on the cell right above). 
# 
# Now, we're ready to build the model. First, we define the R(t), I(t), and S(t) variables according to the SIR model theory. It's a good idea to also split the data into two sets for validation purposes, and we can use the train_test_split function for that. The dependent variable t can be taken as the number of days passed since the first recording, and we're gonna store that data into the day_count column. We have about 75 days of recordings to base our model, and we will take 120 days to be the forecasting interval.

# In[ ]:


# defining the variables for the SIR model
ontario_df['Rec_immune'] = ontario_df['Deaths'] + ontario_df['Recovered']
ontario_df['Infected'] = ontario_df['Confirmed'] - ontario_df['Rec_immune']
ontario_df['Susceptible'] = N_ontario - ontario_df['Rec_immune'] - ontario_df['Infected']

# we need arrays for odeint
sus = np.array(ontario_df['Susceptible'],dtype=float)
infec = np.array(ontario_df['Infected'],dtype=float)
rec = np.array(ontario_df['Rec_immune'],dtype=float)

# splitting the data for validation purposes
x_train, x_test, y_train, y_test = train_test_split(ontario_df['day_count'],    ontario_df[['Susceptible','Infected','Rec_immune']],test_size=0.25,shuffle=False)
xtrain = np.array(x_train.iloc[0:],dtype=float)
ytrain = np.array(y_train.iloc[0:],dtype=float)
xtest = np.array(x_test.iloc[0:],dtype=float)
ytest = np.array(y_test.iloc[0:],dtype=float)

# forecasting data
tdata = np.array(ontario_df.day_count,dtype=float)
xcast = np.linspace(0,120,121)
ycast = np.array(ontario_df[['Susceptible','Infected','Rec_immune']],dtype=float)


# The model can be created using the couple differential equations, as follows:

# In[ ]:


# create model
def sir_model(z,t,beta,gamma):  
    dSdt = -(beta*z[1]*z[0])/N_ontario
    dIdt = (beta*z[1]*z[0])/N_ontario - gamma*z[1]
    dRdt = gamma*z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt


# We'll fit the model to the train dataset using the *fit_odeint* function, and passing the beta and gamma parameters to be fit:

# In[ ]:


# fit model to train set
z0 = [ytrain[0,0],ytrain[0,1],ytrain[0,2]]

def fit_odeint(t,beta,gamma):
    return odeint(sir_model,z0,t,args=(beta,gamma,))[:,1]

popt, pcov = curve_fit(fit_odeint,xtrain,ytrain[:,1],p0=[1,1])
fitted = fit_odeint(xtrain, *popt)

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])


# Prediction is conducted by taking the optimal parameters obtained from the fit, and passing them as arguments to *odeint*. The forecast is done in a similar fashion, only by taking the longer interval xcast to be the interval of prediction:

# In[ ]:


# predict
xcast = np.linspace(0,120,121)
predicted = odeint(sir_model,ytrain[0,:],xcast,args=(popt[0],popt[1],))[:,1] 

poptcast, pcovcast = curve_fit(fit_odeint,tdata,ycast[:,1],p0=[1,1])
forecast = fit_odeint(xcast,*poptcast)


# Now, we're ready to plot. Let's make sure to include a vertical dashed line at the beginning of the forecasting prediction interval. The days are counted as of Feb-12.
# 
# The test data here is being used to verify the evolution of the infection rate: if Ontario had followed the same path as it was following on the first 50 days of the pandemic, the province would see a curve similar to the pink one:

# In[ ]:


# visualization
with plt.style.context('seaborn-white'):
    fig, ax = plt.subplots(1,1, figsize=(16,11))
plt.plot(xtrain, ytrain[:,1], 'o',label="Feb-12 to Apr-07",color='lightcoral')
plt.plot(xtest, ytest[:,1], 'o',label="Apr-07 to Apr-28",color='cornflowerblue')
plt.plot(xcast, predicted,label="Original curve",color='lightcoral')
plt.plot(xcast, forecast,label="Flattened curve",color='cornflowerblue')
plt.axvline(x=xtest[-1], ls='--',color='gray')
plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0.)
plt.title("Fit of SIR model: COVID-19 infections in Ontario")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()


# However, recent records (as of Apr-6) show that Ontario has in fact been seeing a curve similar to the blue one, which has a lower peak than the pink one. Ontario has been flattening the curve!
# 
# According to the forecasting predictions, infection rates should continue to decrease and reach a lower point near the middle-end of June. -- that is if we continue to follow the same behavior that we've been following since Apr-06, which is roughly when social distancing measures started to be implemented in Ontario.
# 
# So, the conclusion here is that social distancing and self-isolation measures are probably contributing to the decrease in the number infected people, and therefore, those measures should continue until at least mid-June in order to prevent a bigger portion of the province from getting infected.
# 
# The good news is we're on the right track for keeping the number of infections low and preventing a higher number of fatalities. Stay safe, Ontario. We are gonna make it through! 
