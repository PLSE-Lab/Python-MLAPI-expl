#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 Data Analysis for World data and India data

# Import the libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
import os
import datetime
import operator
import folium
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


today=datetime.date.today().strftime("%m-%d-%Y")
data_date=datetime.date.today()-datetime.timedelta(days=1)
print("Covid-19 Analysis on {}".format(today))
data_date=data_date.strftime("%m-%d-%Y")


# Loading of  the all neccessary data.

# In[ ]:



confirmed_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
confirmed_cases.head()


# In[ ]:


deaths_reported= pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
deaths_reported.head()


# In[ ]:


recovered_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
recovered_cases.head()


# In[ ]:


latest_data=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(data_date))
latest_data.head()


# In[ ]:


india_states= pd.read_csv('/kaggle/input/india-states-covid-data/{}data.csv'.format(str(data_date)))
#india_states= india_states.iloc[:-1,:]
#india_states.States[35]='Misc'
india_states


# ### Size of the a Datasets

# In[ ]:


print("Datasize of Confirmed_cases dataset:",confirmed_cases.shape)
print("Datasize of deaths_reported dataset:",deaths_reported.shape)
print("Datasize of recovered_cases dataset:",recovered_cases.shape)
print("Datasize of latest_data dataset:",latest_data.shape)
print("Datasize of india_states dataset:",india_states.shape)


# In[ ]:


cols=confirmed_cases.columns
confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
deaths=deaths_reported.loc[:,cols[4]:cols[-1]]
recovered=recovered_cases.loc[:,cols[4]:cols[-1]]
confirmed.head()


# In[ ]:


def country_cases(country,x):
    dates=confirmed.columns
    country_data=[]
    for i in dates:
        country_data.append(x[x['Country/Region']==country][i].sum())
    return country_data


# ##### Confirmed Cases data for some countries.

# In[ ]:


india_cases = country_cases("India",confirmed_cases)
brazil_cases= country_cases("Brazil",confirmed_cases)
us_cases= country_cases("US",confirmed_cases)
italy_cases = country_cases("Italy",confirmed_cases)
germany_cases = country_cases("Germany",confirmed_cases)
spain_cases = country_cases("Spain",confirmed_cases)
france_cases = country_cases("France",confirmed_cases)
uk_cases = country_cases("'United Kingdem'",confirmed_cases)
russia_cases = country_cases("Russia",confirmed_cases)
peru_cases = country_cases("Peru",confirmed_cases)
chile_cases = country_cases("Chile",confirmed_cases)
mexico_cases = country_cases("Mexico",confirmed_cases)


# ##### Death Cases data for some countries.

# In[ ]:


india_deaths=country_cases("India",deaths_reported)
brazil_deaths= country_cases("Brazil",deaths_reported)
us_deaths= country_cases("US",deaths_reported)
italy_deaths = country_cases("Italy",deaths_reported)
germany_deaths = country_cases("Germany",deaths_reported)
spain_deaths = country_cases("Spain",deaths_reported)
france_deaths = country_cases("France",deaths_reported)
uk_deaths = country_cases("'United Kingdem'",deaths_reported)
russia_deaths = country_cases("Russia",deaths_reported)
peru_deaths = country_cases("Peru",deaths_reported)
chile_deaths = country_cases("Chile",deaths_reported)
mexico_deaths = country_cases("Mexico",deaths_reported)


# ##### Recovered Cases data for some countries.

# In[ ]:


india_recoveries = country_cases("India",recovered_cases)
brazil_recoveries= country_cases("Brazil",recovered_cases)
us_recoveries= country_cases("US",recovered_cases)
italy_recoveries = country_cases("Italy",recovered_cases)
germany_recoveries = country_cases("Germany",recovered_cases)
spain_recoveries = country_cases("Spain",recovered_cases)
france_recoveries = country_cases("France",recovered_cases)
uk_recoveries = country_cases("'United Kingdem'",recovered_cases)
russia_recoveries = country_cases("Russia",recovered_cases)
peru_recoveries = country_cases("Peru",recovered_cases)
chile_recoveries = country_cases("Chile",recovered_cases)
mexico_recoveries = country_cases("Mexico",recovered_cases)


# In[ ]:


india_cases = country_cases("India",confirmed_cases)
india_recoveries = country_cases("India",recovered_cases)
india_deaths=country_cases("India",deaths_reported)
india_active = [x-(y+z) for x,y,z in zip(india_cases,india_recoveries,india_deaths)]


# In[ ]:


world_cases=[]
total_deaths=[]
mortatality_rate=[]
recovery_rate=[]
total_recoverd=[]
total_active=[]
mortatality_rate = []
recovery_rate = []

dates=confirmed.columns
for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recovered[i].sum()

    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recoverd.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)

    mortatality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)


# In[ ]:


# Function for daily increase cases
def daily_increase(data):
    '''function to get daily increase
    of the of the values.
    '''
    d=[]
    for i in range(len(data)):
        if i==0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d


# In[ ]:


print("Daily Increse cases in India: ",daily_increase(india_cases)[::-1])


# In[ ]:


unique_countries=list(latest_data['Country_Region'].unique())
print( "Total No of Unique countries in dataset\n",len(unique_countries))
print("\nTotal Unique countries in datasetare:-  ",unique_countries)


# In[ ]:


country_confirmed_cases=[]
country_death_cases=[]
country_recovery_cases=[]
country_active_cases=[]
country_mortality_rate=[]
country_recovery_rate=[]

no_cases=[]
for i in unique_countries:
    cases=latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_countries.remove(i)
#sort countries by the number of confirmed cases
unique_countries=[k for k ,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
    country_recovery_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
    country_active_cases.append(country_confirmed_cases[i]-country_death_cases[i]-country_recovery_cases[i])
    # moratlity_rate=[(death cases)/confirmed cases)]
    country_mortality_rate.append(country_death_cases[i]/country_confirmed_cases[i])
    #Recovery__rate=[(recovered cases)/confirmed cases)]
    country_recovery_rate.append(country_recovery_cases[i]/country_confirmed_cases[i])


# In[ ]:


#Country wise dataframe 
country_df=pd.DataFrame({'Country Name':unique_countries,'Number of Confirmed Cases':country_confirmed_cases,'Number of Deaths':country_death_cases,
                         'Number of Recoveries':country_recovery_cases,'Number of Active Cases':country_active_cases,'Mortality Rate':country_mortality_rate,"Recovery Rate":country_recovery_rate})

# number of cases per country/region
subset=country_df.columns
country_df.style.background_gradient(cmap='Blues',subset=subset[1])                        .background_gradient(cmap='Reds',subset=subset[2])                        .background_gradient(cmap='Greens',subset=subset[3])                        .background_gradient(cmap='Purples',subset=subset[4])                        .background_gradient(cmap='Oranges',subset=subset[5])                        .background_gradient(cmap='Greys',subset=subset[6])
                  


# In[ ]:


import plotly.offline as py 
import plotly.graph_objs as go
import pycountry


# In[ ]:


country_df=pd.DataFrame({'Country Name':unique_countries,'Number of Confirmed Cases':country_confirmed_cases})
country_df.head()


# In[ ]:


plt.figure(figsize=(9,9))
worldmap = [dict(type = 'choropleth', locations = country_df['Country Name'], locationmode = 'country names',
                 z = country_df['Number of Confirmed Cases'], autocolorscale = True, reversescale = False, 
                 marker = dict(line = dict(color = 'rgb(90,90,90)', width = 0.5)), 
                 colorbar = dict(autotick = False, title = 'Number of Confirmed Cases'))]

layout = dict(title = 'Number of Confirmed Cases', geo = dict(showframe = False, showcoastlines = False, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)


# In[ ]:


country_df[country_df["Country Name"]=='India']


# In[ ]:


USA_confirmed=latest_data[latest_data["Country_Region"]=="US"]['Confirmed'].sum()
outside_USA_confirmed=np.sum(country_confirmed_cases)-USA_confirmed
plt.figure(figsize=(10,5))
plt.barh("USA",USA_confirmed)
plt.barh("Outside USA",outside_USA_confirmed)
plt.title("Number of Coronavirus Confirmed Cases",size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


print("USA: {} cases".format(USA_confirmed))
print("Outside USA: {} cases".format(outside_USA_confirmed))
print("Total: {} cases in the world".format(USA_confirmed+outside_USA_confirmed))


# In[ ]:


# only show 10 countries with the most confirmed case,rest are grouped into the outher category

visual_unique_countries=[]
visaul_confirmed_cases=[]
others=np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visaul_confirmed_cases.append(country_confirmed_cases[i])

visual_unique_countries.append("Others")
visaul_confirmed_cases.append(others)


# In[ ]:


def plot_bar_graphs(x,y,title):
    plt.figure(figsize=(12,8))
    plt.barh(x,y, align = 'center', color = 'gold', edgecolor = 'k')
    for index, value in enumerate(y):
        plt.text(value+10000, index, str(value), fontsize = 18)
    plt.title(title,size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

# Pie Plot function
def plot_pie_charts(x,y,title):
    c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(x))
    plt.figure(figsize=(10,10))
    plt.title(title,size=20)
    labels=[]
    explode=[]
    for i in x:
        labels.append(str(i))
        explode.append(0.05)
    plt.pie(y,labels=labels,autopct='%1.1f%%',colors=c,explode=explode)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    #plt.legend(x,loc='best',fontsize=15)
    plt.yticks(size=20)
    plt.show()


# In[ ]:


plot_bar_graphs(visual_unique_countries,visaul_confirmed_cases,"Number of Covid-19 Confirmed cases in Countries")
plot_pie_charts(visual_unique_countries,visaul_confirmed_cases,"Number of Covid-19 Confirmed cases in Countries")


# In[ ]:


def plot_pie_country_with_regions(country_name,title):
    regions=list(latest_data[latest_data['Country_Region']==country_name]['Province_State'].unique())
    confirmed_cases=[]
    no_cases=[]
    for i in regions:
        cases=latest_data[latest_data["Province_State"]==i]["Confirmed"].sum()
        if cases>0:
            confirmed_cases.append(cases)
        else:
            no_cases.append(i)
  #remove the areas with no cases
    for i in no_cases:
        regions.remove(i)

  #show top 10 cases
    regions=[k for k,v in sorted(zip(regions,confirmed_cases),key=operator.itemgetter(1),reverse=True)]


    for i in range(len(regions)):
        confirmed_cases[i]=latest_data[latest_data["Province_State"]==regions[i]]['Confirmed'].sum()


    #others province/state will be considered others
    if(len(regions))>10:
        regions_10=regions[:10]
        regions_10.append("others")
        confirmed_cases_10=confirmed_cases[:10]
        confirmed_cases_10.append(np.sum(confirmed_cases[10:]))
        plot_pie_charts(regions_10,confirmed_cases_10,title)

    else:
        plot_pie_charts(regions,confirmed_cases,title)
    


# In[ ]:


plot_pie_country_with_regions("US","COVID-19 confirmed cases in US")


# In[ ]:


days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recoverd=np.array(total_recoverd).reshape(-1,1)

days_in_future=20
future_forcast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forcast[:-20]

start="1/22/2020"
start_date=datetime.datetime.strptime(start,"%m/%d/%Y")
future_forcast_dates=[]
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date+datetime.timedelta(days=i)).strftime("%m/%d/%Y"))


# In[ ]:


def plot_line(x,y,title):
    plt.figure(figsize=(12,6))
    plt.plot(x,y)
    plt.title(title,size=25)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Numer of cases",size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()

def plot_bar(x,y,title):
    plt.figure(figsize=(12,8))
    plt.bar(x,y,width=0.8,align='center')
    plt.title(title,size=25)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Numer of cases",size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


# In[ ]:


plot_line(adjusted_dates,world_cases,"World's Coronavirus cases  over time")
plot_line(adjusted_dates,total_deaths,"World's Coronavirus Deaths  over time")
plot_line(adjusted_dates,total_recoverd,"World's Coronavirus Recoveries over time")
plot_line(adjusted_dates,total_active,"World's Coronavirus active cases over time")


# In[ ]:


plt.figure(figsize=(15,9))
plt.plot(adjusted_dates,total_deaths,color='r')
plt.plot(adjusted_dates,total_recoverd,color="green")
plt.plot(adjusted_dates,total_active,color="b")
plt.title("Number of Coronavirus Cases in the World",size=20)
plt.legend(["Death","Recoveries","Active"],loc="best",fontsize=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("Number of cases",size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# **Comparision between Daily increase in Confirmed cases, Recoveries and Deaths**

# In[ ]:


plt.figure(figsize=(15,9))
plt.plot(adjusted_dates,daily_increase(india_cases),color='g')
plt.plot(adjusted_dates,daily_increase(india_recoveries),color="b")
plt.plot(adjusted_dates,daily_increase(india_deaths),color="r")
plt.plot(adjusted_dates,daily_increase(india_active),color="y")
plt.title("Comparision b/w Daily increase in Confirmed, recovered and Deaths in India ",size=20)
plt.legend(["Daily increse cases","Daily Recoveries","Daily Deaths","Active Cases"],loc="best",fontsize=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("Number of cases",size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
plt.plot(adjusted_dates,daily_increase(india_deaths),color='r')

plt.title("Daily increse of deaths",size=20)
plt.legend(["Daily Increse deaths"],loc="best",fontsize=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("Number of cases",size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(total_recoverd,total_deaths)
plt.title("Number of Coronavirus Deaths vs Number of Cororavirus recoveries over the world",size=20)
plt.xlabel("Number of Recoveries",size=20)
plt.ylabel("Number of Deaths",size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


def country_plot(x,y1,y2,y3,y4,country):
    plt.figure(figsize=(12,6))
    plt.plot(x,y1)
    plt.title("{} confirmed cases".format(country),size=20)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Number of Cases",size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(x,y2)
    plt.title("{} Daily increse in confirmed cases".format(country),size=20)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Number of Cases",size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(x,y3)
    plt.title("{} Daily increse in deaths".format(country),size=20)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Number of Cases",size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(x,y4)
    plt.title("{} Daily Increse in Recoveries".format(country),size=20)
    plt.xlabel("Days since 1/22/2020",size=20)
    plt.ylabel("Number of Cases",size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    
    

    
    


# In[ ]:


country_plot(adjusted_dates,world_cases,daily_increase(world_cases),daily_increase(total_deaths),daily_increase(total_recoverd),"World")


# 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w

# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(adjusted_dates,brazil_cases,'k')
plt.plot(adjusted_dates,italy_cases,'g')
plt.plot(adjusted_dates,us_cases,'r')
plt.plot(adjusted_dates,spain_cases,'c')
plt.plot(adjusted_dates,france_cases,'m')
plt.plot(adjusted_dates,peru_cases,'y')
plt.plot(adjusted_dates,india_cases,'b')
plt.title("Number of Coronavirus cases in various countries",size=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("number of cases",size=20)
plt.legend(["Brazil","Italy","US","Spain","France","Peru","India"],fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(adjusted_dates,brazil_deaths,'k')
plt.plot(adjusted_dates,italy_deaths,'g')
plt.plot(adjusted_dates,us_deaths,'r')
plt.plot(adjusted_dates,spain_deaths,'c')
plt.plot(adjusted_dates,france_deaths,'m')
plt.plot(adjusted_dates,peru_deaths,'y')
plt.plot(adjusted_dates,india_deaths,'b')
plt.title("Number of Deaths in various countries",size=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("number of cases",size=20)
plt.legend(["Brazil","Italy","US","Spain","France","Peru","India"],fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(adjusted_dates,brazil_recoveries,'k')
plt.plot(adjusted_dates,italy_recoveries,'g')
plt.plot(adjusted_dates,us_recoveries,'r')
plt.plot(adjusted_dates,spain_recoveries,'c')
plt.plot(adjusted_dates,france_recoveries,'m')
plt.plot(adjusted_dates,peru_recoveries,'y')
plt.plot(adjusted_dates,india_recoveries,'b')
plt.title("Number of Recoveries cases in various countries",size=20)
plt.xlabel("Days since 1/22/2020",size=20)
plt.ylabel("number of cases",size=20)
plt.legend(["Brazil","Italy","US","Spain","France","Peru","India"],fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# ### India Covid -19 Analysis

# In[ ]:


confirmed_india=confirmed_cases.loc[confirmed_cases["Country/Region"]=="India"]
deaths_india=deaths_reported.loc[deaths_reported["Country/Region"]=="India"]
recovered_india=recovered_cases.loc[recovered_cases["Country/Region"]=="India"]
ind_confirm=confirmed_india.iloc[0][-1]
ind_death=deaths_india.iloc[0][-1]
ind_recover=recovered_india.iloc[0][-1]
ind_active=confirmed_india.iloc[0][-1]-deaths_india.iloc[0][-1]-recovered_india.iloc[0][-1]
print("India Confirmed cases:-",ind_confirm)
print("India total deaths:-",ind_death)
print("India Recovered cases:-",ind_recover)
print("India Active cases:-",ind_active)


# In[ ]:


x=pd.DataFrame([["Confirmed","Recovered","Active","Deaths"],[ind_confirm,ind_recover,ind_active,ind_death]])
x


# In[ ]:


plot_bar_graphs(x.loc[0],x.loc[1],'India Covid Bar Graph')
plot_pie_charts((x.drop(0,axis=1)).loc[0],(x.drop(0,axis=1)).loc[1],'India Pie Graph b/w Active,Recovered & Deaths ')


# In[ ]:


var=[confirmed_india,deaths_india,recovered_india]
colors=["b","r","g"]
title=["Confirmed cases","Deaths","Recovered Cases"]
for i in range(len(var)):
    plt.figure(figsize=(15,6))
    plt.plot(var[i].iloc[:,4:].T,color=colors[i])
    plt.xticks(rotation=90)
    plt.title(title[i])
    plt.xlim(50,)
    #plt.legend([],fontsize=20
    plt.show()


# In[ ]:


var=[confirmed_india,deaths_india,recovered_india]
colors=["b","r","g"]
title=["Confirmed cases","Deaths","Recovered Cases"]
for i in range(len(var)):
    plt.figure(figsize=(15,6))
    plt.plot(var[i].iloc[:,4:].T.index,daily_increase(var[i].iloc[:,4:].T.values),color=colors[i])
    plt.xticks(rotation=90)
    plt.title("Daily increase in {}".format(title[i]))
    plt.xlim(50,)
    #plt.legend([],fontsize=20
    plt.show()


# In[ ]:


country_plot(adjusted_dates,india_cases,daily_increase(india_cases),daily_increase(india_deaths),daily_increase(india_recoveries),"India")


# ###  Predictive Analysis

# Import the libraries.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# > ****Polynomial Transformation****

# In[ ]:


X_train_confirmed,X_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,world_cases,
                                            test_size=0.35,random_state=42,shuffle=False)
#transform  data in polynomial
poly=PolynomialFeatures(degree=2)
poly_X_train_confirmed=poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed=poly.fit_transform(X_test_confirmed)
poly_future_forcast=poly.fit_transform(future_forcast)


# ## Polynomial Linear Regression

# In[ ]:


#Linear Regression

lr=LinearRegression(normalize=True,fit_intercept=False)
lr.fit(poly_X_train_confirmed,y_train_confirmed)
train_linear_pred=lr.predict(poly_X_train_confirmed)

test_linear_pred=lr.predict(poly_X_test_confirmed)
linear_pred=lr.predict(poly_future_forcast)

print("MAE:",mean_absolute_error(test_linear_pred,y_test_confirmed))
print("MSE:",mean_squared_error(test_linear_pred,y_test_confirmed))
print("Train R2 Score:",r2_score(train_linear_pred,y_train_confirmed))
print("Test R2 Score:",r2_score(test_linear_pred,y_test_confirmed))


# In[ ]:


plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(["Test data","Polynomial regression Prediction"])
plt.show()


# In[ ]:


def prediction_plot(x,y,pred,algo_name,color):
    plt.figure(figsize=(16,9))
    plt.plot(x,y)
    plt.plot(future_forcast,pred,linestyle="dashed",color=color)
    plt.title("number of coronavirus cases over time")
    plt.xlabel("Days since 1/22/2020")
    plt.ylabel("Number of cases")
    plt.legend(["confirmed cases",algo_name],prop={"size":20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


# In[ ]:


prediction_plot(adjusted_dates,world_cases,linear_pred,"Polynomial Regression Prediction",'r')


# **Prediction Model for India**

# In[ ]:


X_train_confirmed,X_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,india_cases,
                                            test_size=0.35,random_state=42,shuffle=False)
#transform  data in polynomial
poly=PolynomialFeatures(degree=4)
poly_X_train_confirmed=poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed=poly.fit_transform(X_test_confirmed)
poly_future_forcast=poly.fit_transform(future_forcast)

#Linear Regression

lr=LinearRegression(normalize=True,fit_intercept=False)
lr.fit(poly_X_train_confirmed,y_train_confirmed)
train_linear_pred=lr.predict(poly_X_train_confirmed)

test_linear_pred=lr.predict(poly_X_test_confirmed)
linear_pred=lr.predict(poly_future_forcast)

print("MAE:",mean_absolute_error(test_linear_pred,y_test_confirmed))
print("MSE:",mean_squared_error(test_linear_pred,y_test_confirmed))
print("Train R2 Score:",r2_score(train_linear_pred,y_train_confirmed))
print("Test R2 Score:",r2_score(test_linear_pred,y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(["Test data","Polynomial regression Prediction"])
plt.show()


# In[ ]:


prediction_plot(adjusted_dates,india_cases,linear_pred,"Polynomial Regression Prediction India",'r')


# In[ ]:


actual_pred=pd.DataFrame({"Dates":future_forcast_dates[:-20],"Actual Cases": india_cases,"Predicted Cases": np.round(linear_pred[:-20])})
actual_pred.iloc[70:,:]


# In[ ]:


linear_pred=linear_pred.reshape(1,-1)[0]
df_poly=pd.DataFrame({"Dates":future_forcast_dates[-20:],
                      "Predicted no of confirmed cases":np.round(linear_pred[-20:])})


# In[ ]:


# future prediction for next20 days
df_poly


# ## Forecasting Total Number of Cases in India by Prophet

# In[ ]:


from fbprophet import Prophet


# In[ ]:


confirmed_india =pd.DataFrame({"ds":future_forcast_dates[:-20],"y": india_cases})
confirmed_india.tail()
                               


# In[ ]:


m= Prophet(interval_width = 0.95)
m.fit(confirmed_india)
future_date = m.make_future_dataframe(periods=10)
future_date.tail(10)


# In[ ]:


forcast = m.predict(future_date)
forcast[['ds','yhat','yhat_lower','yhat_upper']].tail(10)


# In[ ]:


m.plot(forcast)


# In[ ]:


m.plot_components(forcast)


# **Thanks for going through in this notebook. Your suggestions are requested in comments .If you feel this notebook helpful, please upvote the notebook.**

# In[ ]:




