#!/usr/bin/env python
# coding: utf-8

# please cite our work as "Alazab Moutaz, Awajan Albara, Mesleh Abed alwadood, Ajith Abraham, Jatana Venash, Alhyari Salah (2020). COVID-19 Prediction and Detection Using Artificial Intelligence. International Journal of Computer Information Systems and Industrial Management Applications (12), pp. 168- 181"

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:





# In[ ]:


data1= pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")


# In[ ]:


data1.head()


# In[ ]:


data1['Country/Region'].unique()


# In[ ]:


data = data1[data1['Country/Region'] ==  'Jordan']
aus =  data1[data1['Country/Region'] ==  'Australia']


# In[ ]:


data.head()


# In[ ]:


cnfrm = data.Confirmed.max()
cnfrm


# In[ ]:


dth =  data.Deaths.max()
dth


# In[ ]:


rcv = data.Recovered.max()
rcv


# In[ ]:


plt.figure(figsize=(50,40))
plt.bar(data.Date, data.Confirmed,label="Confirm")
plt.bar(data.Date, data.Recovered,label="Recovery")
plt.bar(data.Date, data.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(frameon=True, fontsize=42)
plt.title('Confrim vs Recovery vs Death',fontsize=30)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(30,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=data,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered", data=data,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=data,
             color="blue",label = "Death")
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)

plt.legend(frameon=True, fontsize=22)
plt.plot(data.Date,data.Confirmed,zorder=1,color="black")
plt.plot(data.Date,data.Recovered,zorder=1,color="red")
plt.plot(data.Date,data.Deaths,zorder=1,color="blue")


# In[ ]:


import folium
map = folium.Map(location=[30.5852,36.2384 ], zoom_start=6,tiles='Stamen Toner')

for lat, lon,Confirmed,Recovered,Deaths in zip(data['Latitude'], data['Longitude'],data['Confirmed'],data['Recovered'],data['Deaths']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =(
                    'Confirmed: ' + str(Confirmed) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(map)
map


# In[ ]:


data = data.tail(10)


# In[ ]:


pr_data_r = data.loc[:,['Date','Confirmed']]
pr_data_r.columns = ['ds','y']
pr_data_r.head()


# In[ ]:


m=Prophet()
m.fit(pr_data_r)
future=m.make_future_dataframe(periods=15)
forecast_r=m.predict(future)
forecast_r


# In[ ]:


cnfrm = forecast_r.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['Date','Confirmed']
cnfrm


# In[ ]:


fig_r = plot_plotly(m, forecast_r)
py.iplot(fig_r) 

fig_r = m.plot(forecast_r,xlabel='Date',ylabel='Confirmation Count')


# In[ ]:


k = data.loc[:,["Date","Confirmed"]]
k


# In[ ]:


confirm_cs = pr_data_r.cumsum()
arima_data = pd.DataFrame(confirm_cs['y'])
arima_data['date'] = pr_data_r['ds']
arima_data.columns = ['count','confirmed_date']
arima_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


aus.head(50)


# In[ ]:


aus.shape


# In[ ]:


aus =aus.tail(417)


# In[ ]:


aus1 = aus.groupby("Date")[["Confirmed","Deaths","Recovered"]].sum().reset_index()


# In[ ]:


aus1 = aus1.tail(14)


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(aus1.Date, aus1.Confirmed,label="Confirm")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation",fontsize=50)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(aus1.Date, aus1.Recovered,label="Recovery")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Recoverey",fontsize=50)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(aus1.Date, aus1.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Death",fontsize=50)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(30,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=aus1,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered", data=aus1,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=aus1,
             color="blue",label = "Death")
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)

plt.legend(frameon=True, fontsize=22)
plt.plot(aus1.Date,aus1.Confirmed,zorder=1,color="black")
plt.plot(aus1.Date,aus1.Recovered,zorder=1,color="red")
plt.plot(aus1.Date,aus1.Deaths,zorder=1,color="blue")


# In[ ]:


aus1['Confirmed_new'] = aus1['Confirmed']-aus1['Confirmed'].shift(1)
aus1['Recovered_new'] = aus1['Recovered']-aus1['Recovered'].shift(1)
aus1['Deaths_new'] = aus1['Deaths']-aus1['Deaths'].shift(1)


# In[ ]:


idata = aus1


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Confirmed_new,label="Confirm Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed Cases',fontsize = 35)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Recovered_new,label="Recovered Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Recovered Cases',fontsize = 35)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Deaths_new,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Deaths',fontsize = 35)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="Confirmed_new", data=idata,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered_new", data=idata,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths_new", data=idata,
             color="blue",label = "Death")
plt.plot(idata.Date,idata.Confirmed_new,zorder=1,color="black")
plt.plot(idata.Date,idata.Recovered_new,zorder=1,color="red")
plt.plot(idata.Date,idata.Deaths_new,zorder=1,color="blue")


# In[ ]:


import folium
map = folium.Map(location=[25.2744,133.7751 ], zoom_start=6,tiles='Stamen Toner')

for lat, lon,Confirmed,Recovered,Deaths in zip(aus['Latitude'], aus['Longitude'],aus['Confirmed'],aus['Recovered'],aus['Deaths']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =(
                    'Confirmed: ' + str(Confirmed) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(map)
map


# In[ ]:


global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


def smoother(inputdata,w,imax):
    data = 1.0*inputdata
    data = data.replace(np.nan,1)
    data = data.replace(np.inf,1)
    #print(data)
    smoothed = 1.0*data
    normalization = 1
    for i in range(-imax,imax+1):
        if i==0:
            continue
        smoothed += (w**abs(i))*data.shift(i,axis=0)
        normalization += w**abs(i)
    smoothed /= normalization
    return smoothed

def growth_factor(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    confirmed_iminus2 = confirmed.shift(2, axis=0)
    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)

def growth_ratio(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    return (confirmed/confirmed_iminus1)

# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.
def plot_country_active_confirmed_recovered(country):
    
    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.
    country_data = global_data[global_data['Country/Region']==country]
    table = country_data.drop(['SNo','Province/State', 'Last Update'], axis=1)
    table['ActiveCases'] = table['Confirmed'] - table['Recovered'] - table['Deaths']
    table2 = pd.pivot_table(table, values=['ActiveCases','Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)
    table3 = table2.drop(['Deaths'], axis=1)
   
    # Growth Factor
    w = 0.5
    table2['GrowthFactor'] = growth_factor(table2['Confirmed'])
    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)

    # 2nd Derivative
    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative
    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)


    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio
    table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])
    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)
    
    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.
    table2['GrowthRate']=np.gradient(np.log(table2['Confirmed']))
    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)
    
    # horizontal line at growth rate 1.0 for reference
    x_coordinates = [1, 100]
    y_coordinates = [1, 1]
    #plots
    table2['Deaths'].plot(title='Deaths')
    plt.show()
    table3.plot() 
    plt.show()
    table2['GrowthFactor'].plot(title='Growth Factor')
    plt.plot(x_coordinates, y_coordinates) 
    plt.show()
    table2['2nd_Derivative'].plot(title='2nd_Derivative')
    plt.show()
    table2['GrowthRatio'].plot(title='Growth Ratio')
    plt.plot(x_coordinates, y_coordinates)
    plt.show()
    table2['GrowthRate'].plot(title='Growth Rate')
    plt.show()

    return 


# In[ ]:


plot_country_active_confirmed_recovered('Australia')

