#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# This Notebook contains Analysis Study for World Data and Indian Covid Dataset and prediction models for Indian states.
# 
# Documentation : https://drive.google.com/file/d/1EyYGMBmiY01GZ7PDuMXWlhvdznpiAqgg/view?usp=sharing 

# In[ ]:


#Importing all necessary Libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import warnings
import math
    
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(12,6)})


# In[ ]:


###############################
#                             #
#                             #
#    ANALYSING WORLD DATA     #
#                             #
#                             #
###############################


# In[ ]:


worldConfirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv');
worldConfirmed.info();


# In[ ]:


worldConfirmed


# In[ ]:


worldConfirmed.isnull().sum()


# In[ ]:


worldSeparate = worldConfirmed.groupby('Country/Region').aggregate(sum)
worldSeparate


# In[ ]:


fig = go.Figure()
for i in worldSeparate.index:
    fig.add_trace(go.Scatter(x=worldSeparate.columns[2:], y=worldSeparate.ix[i,2:] ,
                    mode='lines+markers',
                    name=i))
else:
    fig.show()


# In[ ]:


###############################
#                             #
#                             #
#    ANALYSING INDIA DATA     #
#                             #
#                             #
###############################


# In[ ]:


data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',index_col=0)
data.head()


# In[ ]:


pd.unique(data['ConfirmedIndianNational'])


# In[ ]:


pd.unique(data['ConfirmedForeignNational'])


# In[ ]:


data = data.drop(['ConfirmedIndianNational','ConfirmedForeignNational'
                  ,'Time'],axis = 1)
data.tail()


# In[ ]:


stateWise = data.groupby(['State/UnionTerritory'])[['Confirmed','Cured','Deaths']].aggregate(max)
stateWise[['Confirmed','Cured','Deaths']].plot(kind = 'barh',figsize=(10,15))


# In[ ]:


stateList = list((pd.unique(data['State/UnionTerritory'])))
stateList.remove("Unassigned")
stateList


# In[ ]:


stateData = data[
            (data['State/UnionTerritory'] == "Maharashtra")
        ]
stateData.head()#First 5 entries


# In[ ]:


stateData.tail()#last 5 entries


# In[ ]:


stateData[['Date','Confirmed','Deaths','Cured']].set_index('Date').plot(kind = 'line',title = "Confirmed cases per day for "+"Maharashtra",
      figsize=(12,8));


# In[ ]:


perDayIncrement = []
for ind,val in enumerate(stateData['Confirmed']):
    if ind == 0 or ind == len(stateData['Confirmed'] - 1):
        perDayIncrement.append(val)
    else:
        incre = (val - stateData['Confirmed'].iloc[ind-1])
        perDayIncrement.append(incre)
else:
    stateData['perDayIncrement'] = perDayIncrement


# In[ ]:


dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]

dfPerDay[['Date','perDayIncrement']].set_index('Date').plot(kind = 'line',title = "Per Day Increment for "+
      "Maharashtra",figsize=(12,8))
plt.show()


# In[ ]:


sns.boxplot(y = dfPerDay['perDayIncrement'] ,orient='v', 
            palette="Blues").set_title('Box Plot for '+"Maharashtra")


# In[ ]:


###############################
#                             #
#                             #
#   ANALYSING HOSPITAL DATA   #
#                             #
#                             #
###############################


# In[ ]:


hospitalData = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
hospitalData = hospitalData.drop('Sno',axis = 1)
print(hospitalData.shape)
hospitalData.head()


# In[ ]:


hospitalData.describe()


# In[ ]:


hospitalData.isnull().any(1).nonzero()


# In[ ]:


hospitalData = hospitalData.fillna(-1)
hospitalData.head()


# In[ ]:


#Individual Hospital Counts in Different States

states = list(hospitalData['State/UT'])

for ind in range(0,len(states)):
    labels = hospitalData.columns.values
    values = hospitalData.iloc[ind,:].values

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,title=hospitalData.iloc[ind,0], hole=.4)])
    fig.show()


# In[ ]:


#Different Hospital Counts in India 

cols = list(hospitalData.columns)
cols.remove('State/UT')
SumData = hospitalData[cols].iloc[:-1,:].sum(axis=0,skipna=True)
bplot = sns.barplot(x = SumData.index,y = SumData,palette="twilight" );
bplot.set_xticklabels(rotation=-90,labels= SumData.index);
bplot.set_title("Count of Different Categories of Hospitals in India !");


# In[ ]:


###############################
#                             #
#                             #
#     ANALYSING LABS DATA     #
#                             #
#                             #
###############################


# In[ ]:


labData = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
labData.head()


# In[ ]:


pd.unique(labData.type)


# In[ ]:


stateLabs = labData.groupby('state')[['lab']].aggregate('count')
stateLabs.info()


# In[ ]:


labChart = sns.barplot(x = stateLabs.index,y = stateLabs.lab)
labChart.set_xticklabels(rotation = -90, labels = stateLabs.index)
labChart.set_title("Count of Number of Testing Labs Across India (State Wise)");
plt.show()


# In[ ]:


stateLabCategory = labData.groupby('type')[['lab']].aggregate('count')
stateLabCategory.info()


# In[ ]:


labCategory = sns.barplot(x = stateLabCategory.index,y = stateLabCategory.lab,palette='inferno_r')
labCategory.set_xticklabels(rotation=30,labels = stateLabCategory.index)
labCategory.set_title("Count of Type of Testing Labs in India");
plt.show()


# In[ ]:


###############################
#                             #
#                             #
#   ANALYSING LAB TEST DATA   #
#                             #
#                             #
###############################


# In[ ]:


testingDetails = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')


# In[ ]:


testingDetails.describe()


# In[ ]:


testingDetails.tail()


# In[ ]:


testingDetails.dropna(axis=0,inplace=True)
testingDetails


# In[ ]:


inc = sns.barplot(x =testingDetails['TotalIndividualsTested'],y=testingDetails['TotalPositiveCases'],palette="inferno_r")
inc.set_title("Bar Plot Showing Variation of Positive Cases with increased Testing in India");
inc.set_xticklabels(labels = testingDetails['TotalIndividualsTested'],rotation = -90);


# In[ ]:


###############################
#                             #
#                             #
#  ANALYSING STATE WISE DATA  #
#                             #
#                             #
###############################


# In[ ]:


stateTests = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
stateTests.columns


# In[ ]:


stateTests.describe()


# In[ ]:


stateTests.isnull().sum()


# In[ ]:


stateTests.dropna(axis=0,inplace=True)
stateTests.isnull().sum()


# In[ ]:


stateTests.groupby('State')[['TotalSamples','Negative','Positive']].aggregate(max).plot(kind='barh',stacked=True,title="State Wise Testing Statistics (Stacked)",figsize=(12,12))


# In[ ]:


stateTests.groupby('State')[['TotalSamples','Negative','Positive']].aggregate(max).plot(kind='barh',title="State Wise Testing Statistics",figsize=(10,15))


# In[ ]:


stateData[['Confirmed']].describe()


# In[ ]:


#Without any skew function
dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]


fig = go.Figure()
dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)


x = dfPerDay['Date'].values
x = x.reshape(-1,1)
y = dfPerDay['perDayIncrement'].values
 
rfmodel = LinearRegression(fit_intercept=True)
        
rfmodel.fit(x,y)
y_predition = rfmodel.predict(x)

fig.add_trace(go.Scatter(x = dfPerDay['Date'] , y= dfPerDay['perDayIncrement'],
                    mode='markers', name='Actual Points'))

fig.add_trace(go.Scatter(x=dfPerDay['Date'], y=y_predition,
                    mode='lines+markers',
                    name='Regression Fit'))
mse = mean_squared_error(y,y_predition)
print("Mean Squared Error : "+str(mse))
fig.show()


# In[ ]:


#CUBE ROOT FOR SKEW DATA

dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]


fig = go.Figure()
dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)
dfPerDay['perDayIncrement'] = dfPerDay['perDayIncrement'].apply(lambda x: (x**(1/3)).real )

x = dfPerDay['Date'].values
x = x.reshape(-1,1)
y = dfPerDay['perDayIncrement'].values
 
rfmodel = LinearRegression(fit_intercept=True)
        
rfmodel.fit(x,y)
y_predition = rfmodel.predict(x)

fig.add_trace(go.Scatter(x = dfPerDay['Date'] , y= dfPerDay['perDayIncrement'],
                    mode='markers', name='Actual Points'))

fig.add_trace(go.Scatter(x=dfPerDay['Date'], y=y_predition,
                    mode='lines+markers',
                    name='Regression Fit'))
mse = mean_squared_error(y,y_predition)
print("Mean Squared Error : "+str(mse))
fig.show()


# In[ ]:


#LOG FOR SKEW DATA

dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]


fig = go.Figure()
dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)
dfPerDay['perDayIncrement'] = np.log(dfPerDay['perDayIncrement'])

x = dfPerDay['Date'].values
x = x.reshape(-1,1)
y = dfPerDay['perDayIncrement'].values
 
rfmodel = LinearRegression(fit_intercept=True)
        
rfmodel.fit(x,y)
y_predition = rfmodel.predict(x)

fig.add_trace(go.Scatter(x = dfPerDay['Date'] , y= dfPerDay['perDayIncrement'],
                    mode='markers', name='Actual Points'))

fig.add_trace(go.Scatter(x=dfPerDay['Date'], y=y_predition,
                    mode='lines+markers',
                    name='Regression Fit'))
mse = mean_squared_error(y,y_predition)
print("Mean Squared Error : "+str(mse))
fig.show()


# In[ ]:


#WITHOUT ANY SKEW FUNCTION AS WELL AS PREDICTION ON CONFIRMED DATA PER DAY
dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]


fig = go.Figure()
dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)
dfPerDay['Confirmed'] = np.log(dfPerDay['Confirmed'])

x = dfPerDay['Date'].values
x = x.reshape(-1,1)
y = dfPerDay['Confirmed'].values
 
rfmodel = LinearRegression(fit_intercept=True)
        
rfmodel.fit(x,y)
y_predition = rfmodel.predict(x)

fig.add_trace(go.Scatter(x = dfPerDay['Date'] , y= dfPerDay['Confirmed'],
                    mode='markers', name='Actual Points'))

fig.add_trace(go.Scatter(x=dfPerDay['Date'], y=y_predition,
                    mode='lines+markers',
                    name='Regression Fit'))
mse = mean_squared_error(y,y_predition)
print("Mean Squared Error : "+str(mse))
fig.show()


# In[ ]:


f, axes = plt.subplots(1, 3)

sns.boxplot(y = dfPerDay['Confirmed'] ,orient='v' , ax=axes[0],  
                    palette="Blues").set_title('Box Plot for Maharashtra');

sns.boxplot(y = dfPerDay['Confirmed'].apply(lambda x: (x**(1/3)).real ) , orient='v' , ax=axes[1],  
            palette="Oranges").set_title('Box Plot '+' with Cube Root');

sns.boxplot(y = np.log(dfPerDay['Confirmed']) , orient='v' , ax=axes[2],  
            palette="Dark2").set_title('Box Plot  with Log Function');


# In[ ]:


f, axes = plt.subplots(1, 3)

sns.boxplot(y = dfPerDay['perDayIncrement'] ,orient='v' , ax=axes[0],  
                    palette="Blues").set_title('Box Plot for Maharashtra');

sns.boxplot(y = dfPerDay['perDayIncrement'].apply(lambda x: (x**(1/3)).real ) , orient='v' , ax=axes[1],  
            palette="Oranges").set_title('Box Plot '+' with Cube Root');

sns.boxplot(y = np.log(dfPerDay['perDayIncrement']) , orient='v' , ax=axes[2],  
            palette="Dark2").set_title('Box Plot  with Log Function');


# In[ ]:


#PREDICTION USING LOG AS SKEW FUNCTION

data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',index_col=0)
data = data.drop(['ConfirmedIndianNational','ConfirmedForeignNational','Time'],axis = 1)
stateWise = data.groupby(['State/UnionTerritory'])[['Confirmed','Cured','Deaths']].aggregate(max)
stateWise[['Confirmed','Cured','Deaths']].plot(kind = 'barh',figsize=(10,20))


stateList = list((pd.unique(data['State/UnionTerritory'])))
stateList.remove("Unassigned")
stateAll = go.Figure()

errorList = []
statistics = []

def printData(statList):
    print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
    print("| {:^30} | {:^20} | {:^20} |".format("State","Date","Confirmed Cases"))
    print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
    
    for val in statList:    
        print("| {:^30} | {:^20} | {:^20} |".format(val[0],val[1],val[2]))
        print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
        

for state in stateList:
    

    try:
        stateData = data[
            (data['State/UnionTerritory'] == state)
        ]

    
    
        stateData[['Date','Confirmed','Deaths','Cured']].set_index('Date').plot(kind = 'line',title = "Confirmed cases per day for "+state,figsize=(12,6))
        plt.show()
        
        perDayIncrement = []
        for ind,val in enumerate(stateData['Confirmed']):
            if ind == 0 or ind == len(stateData['Confirmed'] - 1):
                perDayIncrement.append(val)
            else:
                incre = (val - stateData['Confirmed'].iloc[ind-1])
                perDayIncrement.append(incre)
        else:
            stateData['perDayIncrement'] = perDayIncrement
    


        dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
    
        dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]

        dfPerDay[['Date','perDayIncrement']].set_index('Date').plot(kind = 'line',title = "Per Day Increment (w/o Log) for "+state,figsize=(12,6))
        plt.show()

        stateAll.add_trace(go.Scatter(x=dfPerDay['Date'], y=dfPerDay['perDayIncrement'] ,
                    mode='lines+markers',
                    name=state))
        f, axes = plt.subplots(1, 2)
        
        sns.boxplot(y = dfPerDay['perDayIncrement'] ,orient='v' , ax=axes[0],   palette="Blues").set_title('Box Plot for '+state+' without Log')
        
        
        
        dfPerDay['perDayIncrement'] = np.log(dfPerDay['perDayIncrement'])
    
        sns.boxplot(y = dfPerDay['perDayIncrement'] , orient='v' , ax=axes[1],  palette="Oranges").set_title('Box Plot for '+state+' with Log')
        plt.show()
        
        dfPerDay[['Date',('perDayIncrement')]].set_index('Date').plot(kind = 'line',title = "Per Day Increment (Log applied) for "+state,figsize=(12,6))
        plt.show()
    
    
    
    
        dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
        dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)
        dfPerDay.describe()

        x = dfPerDay['Date'].values
        x = x.reshape(-1,1)
        y = dfPerDay['perDayIncrement'].values

 
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 10)
        rfmodel = LinearRegression(fit_intercept=True)

        
        rfmodel.fit(x_train,y_train)
        y_predition = rfmodel.predict(x_test)

        mse = mean_squared_error(y_test,y_predition)
        rmse = np.sqrt(mse)

        date = dt.datetime.now()
        date = pd.to_datetime(date,dayfirst=True)
        date = dt.datetime.toordinal(date)

        x = np.array([date])
        x = x.reshape(-1,1)

        confirmedPrediction = math.ceil(float(dfPerDay['Confirmed'].iloc[-1] + np.exp(rfmodel.predict(x)) ) )
        score = rfmodel.score(x_test,y_test)
        statistics.append((state,str(dt.datetime.fromordinal(date)),confirmedPrediction))
    except ValueError:
        continue
    except TypeError:
        continue
    
    except:
        raise
else:
    stateAll.show()
    printData(statistics)


# In[ ]:


data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',index_col=0)
data = data.drop(['ConfirmedIndianNational','ConfirmedForeignNational','Time'],axis = 1)
stateWise = data.groupby(['State/UnionTerritory'])[['Confirmed']].aggregate(max)
stateWise['Confirmed'].plot(kind = 'barh',figsize=(12,10))

stateList = list((pd.unique(data['State/UnionTerritory'])))
stateList.remove("Unassigned")


errorList = []
statistics = []

def printData(statList):
    print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
    print("| {:^30} | {:^20} | {:^20} |".format("State","Date","Confirmed Cases","Score"))
    print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
    
    for val in statList:    
        print("| {:^30} | {:^20} | {:^20} |".format(val[0],val[1],val[2]))
        print("|-{:^30}-|-{:^20}-|-{:^20}-|".format("-"*30,"-"*20,"-"*20))
        

for state in stateList:
    try:
        stateData = data[
            (data['State/UnionTerritory'] == state)
        ]

        stateData[['Date','Confirmed','Deaths','Cured']].set_index('Date').plot(kind = 'line',title = "Confirmed cases per day for "+state,figsize=(10,5))
        plt.show()
        perDayIncrement = []
        for ind,val in enumerate(stateData['Confirmed']):
            if ind == 0 or ind == len(stateData['Confirmed'] - 1):
                perDayIncrement.append(val)
            else:
                incre = (val - stateData['Confirmed'].iloc[ind-1])
                perDayIncrement.append(incre)
        else:
            stateData['perDayIncrement'] = perDayIncrement
    


        dfPerDay = stateData[['Date','Confirmed',"perDayIncrement"]]
    
        dfPerDay = dfPerDay[
            (dfPerDay['perDayIncrement'] > 0)
        ]

        dfPerDay[['Date','perDayIncrement']].set_index('Date').plot(kind = 'line',title = "Per Day Increment (w/o Log) for "+state,figsize=(10,5))
        plt.show()
        dfPerDay['perDayIncrement'] = np.log(dfPerDay['perDayIncrement'])
        dfPerDay[['Date',('perDayIncrement')]].set_index('Date').plot(kind = 'line',title = "Per Day Increment (Log) for "+state,figsize=(10,5))
        plt.show()
    
        dfPerDay['Date'] = pd.to_datetime(dfPerDay['Date'],dayfirst=True)
        dfPerDay['Date'] = dfPerDay['Date'].map(dt.datetime.toordinal)
        dfPerDay.describe()
    
        sns.boxplot(y = dfPerDay['perDayIncrement'] ,    palette="Dark2")


        x = dfPerDay['Date'].values
        x = x.reshape(-1,1)
        y = dfPerDay['perDayIncrement'].values

 
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 10)
        rfmodel = LinearRegression(fit_intercept=True)

        rfmodel.fit(x_train,y_train)
        y_predition = rfmodel.predict(x_test)

        mse = mean_squared_error(y_test,y_predition)
        rmse = np.sqrt(mse)

        date = dt.datetime.now()
        date = pd.to_datetime(date,dayfirst=True)
        date = dt.datetime.toordinal(date)

        x = np.array([date])
        x = x.reshape(-1,1)

        confirmedPrediction = math.ceil(float(dfPerDay['Confirmed'].iloc[-1] + np.exp(rfmodel.predict(x))[0]))
        score = rfmodel.score(x_test,y_test)
        statistics.append((state,str(dt.datetime.fromordinal(date)),confirmedPrediction))
    except:
        print(state)
        continue
else:
    printData(statistics)


# In[ ]:




