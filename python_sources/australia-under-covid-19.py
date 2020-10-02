#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/dVuyBgq2z5gVBkFtDc/giphy.gif)

# **Coronaviruses are a large family of viruses which may cause illness in animals or humans. In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The most recently discovered coronavirus causes coronavirus disease COVID-19.COVID-19 is the infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019.**
# * [Source](https://www.who.int/news-room/q-a-detail/q-a-coronaviruses)

# # **Coronavirus in the World**

# In[ ]:


import pandas as pd 
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
import plotly.offline as py
import plotly.express as px


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Active", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="natural earth",
                     animation_frame="Date",width=800, height=500,
                     color_continuous_scale='Reds',
                     range_color=[1000,100000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Importing Libraries**

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


# # **Symptoms of Coronavirus**

# In[ ]:


symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms


# **Bar Plot**

# In[ ]:


fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
fig.show()


# **Pie Plot**

# In[ ]:


fig = px.pie(symptoms,
             values="percentage",
             names="symptom",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# **Tree Plot**

# In[ ]:


fig = px.treemap(symptoms, path=['symptom'], values='percentage',
                  color='percentage', hover_data=['symptom'],
                  color_continuous_scale='burgyl')
fig.show()


# **Fever,Dry cough,Fatigue,Sputum production are the major symtoms of Coronavirus, so if anyone have symptoms like this they should connect to medical authorities**

# **Exploring Data**

# In[ ]:


data1= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data2 = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")


# In[ ]:


aus =  data1[data1['Country/Region'] ==  'Australia']
aus1 = data2[data2['Country/Region'] ==  'Australia']


# In[ ]:


aus.head()


# In[ ]:


aus = aus.dropna()
aus = aus.drop("Last Update",axis=1)


# In[ ]:


from pandas_profiling import ProfileReport 
report = ProfileReport(aus)
report


# # **Grouping Data by Region and Date**

# In[ ]:


aus_grp = aus.groupby(["ObservationDate","Province/State"])[["Confirmed","Recovered","Deaths"]].sum().reset_index()


# In[ ]:


aus_grp = aus_grp.rename(columns={"Province/State":"state","ObservationDate":"date"})


# In[ ]:


aus_grp.head()


# # **Map with Latest Data**

# In[ ]:


import folium
map = folium.Map(location=[-25.2744,133.7751 ], zoom_start=4,tiles='cartodbpositron')

for lat, lon,state,Confirmed,Recovered,Deaths in zip(aus1['Latitude'], aus1['Longitude'],aus['Province/State'],aus1['Confirmed'],aus1['Recovered'],aus['Deaths']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =(
                    'State: ' + str(state) + '<br>'
                    'Confirmed: ' + str(Confirmed) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(map)
map


# # **Daily Changes in Map**

# In[ ]:


aus1 = aus1.dropna()
aus_map = aus1.tail(80)
aus_map1  = aus_map.groupby(['Date', 'Country/Region','Latitude','Longitude'])['Confirmed', 'Deaths', 'Recovered'].max()


aus_map1 = aus_map1.reset_index()
aus_map1.head()
aus_map1['size'] = aus_map1['Confirmed']*90000000
aus_map1
fig = px.scatter_mapbox(aus_map1, lat="Latitude", lon="Longitude",
                     color="Confirmed", size='size',hover_data=['Confirmed','Recovered','Deaths'],
                     color_continuous_scale='burgyl',
                     animation_frame="Date", 
                     title='Spread total cases over time in Australia')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=3, mapbox_center = {"lat": -25.2744, "lon": 133.7751})
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# **If we analyse the world map, we can observe that number of cases are in coastal area and they are keep increasing humidity can be one of the reason as we know virus can stay for longer period of time in humid places, as prevention measure they should be locked down and no on should allowed to go away as they can be carriers of Coronavirus**

# In[ ]:


aus_grp_plot = aus_grp.tail(80)


# # **Confirmed vs Region**

# In[ ]:


fig=px.bar(aus_grp_plot,x='state', y="Confirmed", animation_frame="date", 
           animation_group="state", color="state", hover_name="state")
fig.update_yaxes(range=[0, 3500])
fig.update_layout(title='Confirmed vs Region')


# In[ ]:


get_ipython().system('pip install chart_studio')


# In[ ]:


pip install bubbly


# **New South Waled and Victoria are most affected states in Australia whose confirmation count is keep increasing where  South Australia and Capital Territory control the cases **

# # **Recovery vs Region**

# In[ ]:


fig=px.bar(aus_grp_plot,x='state', y="Recovered", animation_frame="date", 
           animation_group="state", color="state", hover_name="state")
fig.update_yaxes(range=[0, 1200])
fig.update_layout(title='Recovery vs Region')


# **Victoria show extremely good in recovery followed by Western Australia**

# # **Deaths vs Region**

# In[ ]:


fig=px.bar(aus_grp_plot,x='state', y="Deaths", animation_frame="date", 
           animation_group="state", color="state", hover_name="state")
fig.update_yaxes(range=[0, 30])
fig.update_layout(title='Deaths vs Region')


# **New SOuth Wales has more death followed by Victoria, as the no of cases were also high in these states so that is much expected**

# # **Grouping Data by Region**

# In[ ]:


aus_grp_r = aus_grp.groupby("state")[["Confirmed","Recovered","Deaths"]].max().reset_index()


# In[ ]:


aus_grp_r.head()


# In[ ]:


aus_grp_rl20 = aus_grp_r.tail(20)


# # **Confirmed Cases vs Region**

# **Bar Plot**

# In[ ]:


fig = px.bar(aus_grp_rl20[['state', 'Confirmed']].sort_values('Confirmed', ascending=False), 
             y="Confirmed", x="state", color='state', 
             log_y=True, template='ggplot2', title='Confirmed Cases vs Region')
fig.show()


# **Tree Plot**

# In[ ]:


fig = px.treemap(aus_grp_rl20, path=['state'], values='Confirmed',
                  color='Confirmed', hover_data=['state'],
                  color_continuous_scale='burgyl')
fig.show()


# # **Recovered Cases vs Region**

# **Bar Plot**

# In[ ]:


fig = px.bar(aus_grp_rl20[['state', 'Recovered']].sort_values('Recovered', ascending=False), 
             y="Recovered", x="state", color='state', 
             log_y=True, template='ggplot2', title='Recovered Cases vs Region')
fig.show()


# **Tree Plot**

# In[ ]:


fig = px.treemap(aus_grp_rl20, path=['state'], values='Recovered',
                  color='Recovered', hover_data=['state'],
                  color_continuous_scale='burgyl')
fig.show()


# # **Deaths Cases vs Region**

# **Bar Plot**

# In[ ]:


fig = px.bar(aus_grp_rl20[['state', 'Deaths']].sort_values('Deaths', ascending=False), 
             y="Deaths", x="state", color='state', 
             log_y=True, template='ggplot2', title='Deaths Cases vs Region')
fig.show()


# **Tree Plot**

# In[ ]:


fig = px.treemap(aus_grp_rl20, path=['state'], values='Deaths',
                  color='Deaths', hover_data=['state'],
                  color_continuous_scale='burgyl')
fig.show()


# # Recovery Rate

# In[ ]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=aus_grp_rl20, x_column='Confirmed', y_column='Recovered', 
    bubble_column='state',size_column='Confirmed', color_column='state', 
    x_title="Confirm", y_title="Recovery", title='Recovery Rate',
     scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})


# # Death Rate

# In[ ]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=aus_grp_rl20, x_column='Confirmed', y_column='Deaths', 
    bubble_column='state',size_column='Confirmed', color_column='state', 
    x_title="Confirm", y_title="Deaths", title='Death Rate',
     scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})


# In[ ]:


aus_grp_rl20 = aus_grp_rl20.sort_values(by=['Confirmed'],ascending = False)


# # **Confirm, Revovery and Death**

# In[ ]:


plt.figure(figsize=(40,15))
plt.bar(aus_grp_rl20.state, aus_grp_rl20.Confirmed,label="Confirmed")
plt.bar(aus_grp_rl20.state, aus_grp_rl20.Recovered,label="Recovery")
plt.bar(aus_grp_rl20.state, aus_grp_rl20.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=13)
plt.yticks(fontsize=15)

plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed vs Recovery vs Death',fontsize=30)
plt.show()

f, ax = plt.subplots(figsize=(40,15))
ax=sns.scatterplot(x="state", y="Confirmed", data=aus_grp_rl20,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="state", y="Recovered", data=aus_grp_rl20,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="state", y="Deaths", data=aus_grp_rl20,
             color="blue",label = "Death")
plt.plot(aus_grp_rl20.state,aus_grp_rl20.Confirmed,zorder=1,color="black")
plt.plot(aus_grp_rl20.state,aus_grp_rl20.Recovered,zorder=1,color="red")
plt.plot(aus_grp_rl20.state,aus_grp_rl20.Deaths,zorder=1,color="blue")
plt.xticks(fontsize=13)
plt.yticks(fontsize=15)
plt.legend(frameon=True, fontsize=12)


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(15,2))
h=pd.pivot_table(aus_grp_r,columns='state',values=["Confirmed","Recovered","Deaths"])
sns.heatmap(h,cmap=['skyblue','green','red','black'],linewidths=0.5)


# **Group by Date**

# In[ ]:



aus_grp_d = aus_grp.groupby("date")[["Confirmed","Recovered","Deaths"]].sum().reset_index()


# In[ ]:


aus_grp_d.head()


# In[ ]:


aus_grp_dl20 = aus_grp_d.tail(20)


# # ****Confirm, Revovery and Death****

# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Confirmed,label="Confirm Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed Cases',fontsize = 35)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Recovered,label="Recovered Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Recovered Cases',fontsize = 35)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Deaths,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Deaths',fontsize = 35)
plt.show()


# # **Confirmed vs Recovery vs Deaths**

# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Confirmed,label="Confirm Cases")
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Recovered,label="Recovered Cases")
plt.bar(aus_grp_dl20.date, aus_grp_dl20.Deaths,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed vs Recovery vs Deaths',fontsize = 35)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="date", y="Confirmed", data=aus_grp_dl20,
             color="black",label = "Confirmed Patients")
ax=sns.scatterplot(x="date", y="Recovered", data=aus_grp_dl20,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="date", y="Deaths", data=aus_grp_dl20,
             color="blue",label = "Death")
plt.plot(aus_grp_dl20.date,aus_grp_dl20.Confirmed,zorder=1,color="black")
plt.plot(aus_grp_dl20.date,aus_grp_dl20.Recovered,zorder=1,color="red")
plt.plot(aus_grp_dl20.date,aus_grp_dl20.Deaths,zorder=1,color="blue")


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(15,2))
h=pd.pivot_table(aus_grp_dl20,columns='date',values=["Confirmed","Recovered","Deaths"])
sns.heatmap(h,cmap=['skyblue','yellow','green','red','black'],linewidths=0.5)


# # **Recovery Rate**

# In[ ]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=aus_grp_dl20, x_column='Confirmed', y_column='Recovered', 
    bubble_column='date',size_column='Confirmed', color_column='date', 
    x_title="Confirm", y_title="Recovery", title='Recovery Rate',
     scale_bubble=2, height=650)

iplot(figure, config={'scrollzoom': True})


# # Death Rate

# In[ ]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=aus_grp_dl20, x_column='Confirmed', y_column='Deaths', 
    bubble_column='date',size_column='Confirmed', color_column='date', 
    x_title="Confirm", y_title="Deaths", title='Recovery Rate',
     scale_bubble=2, height=650)

iplot(figure, config={'scrollzoom': True})


# # **Status of Patient**

# In[ ]:



Total_confirmed = aus_grp_d['Confirmed'].sum()
Total_recovered = aus_grp_d['Recovered'].sum()
Total_death = aus_grp_d['Deaths'].sum()
data = [['Confirmed', Total_confirmed], ['Recovered', Total_recovered], ['Death', Total_death]] 
df = pd.DataFrame(data, columns = ['state', 'count']) 
fig = px.pie(df,
             values="count",
             names="state",
             title="State of Patient",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# **New Cases**

# In[ ]:


aus_grp_d['Confirmed_new'] = aus_grp_d['Confirmed']-aus_grp_d['Confirmed'].shift(1)
aus_grp_d['Recovered_new'] = aus_grp_d['Recovered']-aus_grp_d['Recovered'].shift(1)
aus_grp_d['Deaths_new'] = aus_grp_d['Deaths']-aus_grp_d['Deaths'].shift(1)


# In[ ]:


aus_grp_d.head()


# In[ ]:


new = aus_grp_d
new = new.tail(14)


# # **Daily Confirm , Recovery and Death**

# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(new.date, new.Confirmed_new,label="Confirm Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed Cases',fontsize = 35)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(new.date, new.Recovered_new,label="Recovered Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Recovered Cases',fontsize = 35)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(new.date, new.Deaths_new,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Deaths',fontsize = 35)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="date", y="Confirmed_new", data=new,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="date", y="Recovered_new", data=new,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="date", y="Deaths_new", data=new,
             color="blue",label = "Death")
plt.plot(new.date,new.Confirmed_new,zorder=1,color="black")
plt.plot(new.date,new.Recovered_new,zorder=1,color="red")
plt.plot(new.date,new.Deaths_new,zorder=1,color="blue")


# # **GrowthFactor,GrowthRatio,GrowthRate**

# In[ ]:


global_data = aus
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
    f, ax = plt.subplots(figsize=(15,5))

    country_data = global_data[global_data['Country/Region']==country]
    table = country_data.drop(['SNo','Province/State'], axis=1)
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
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthFactor'].plot(title='Growth Factor')
    plt.plot(x_coordinates, y_coordinates) 
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['2nd_Derivative'].plot(title='2nd_Derivative')
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthRatio'].plot(title='Growth Ratio')
    plt.plot(x_coordinates, y_coordinates)
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthRate'].plot(title='Growth Rate')
    plt.show()

    return


# In[ ]:


plot_country_active_confirmed_recovered('Australia')


# # **Predictions of Confirmation, Recovery and Deaths**
# ** Algorithms Used**
#  1. Prophet
#  2. Arima
#  3. LSTM

# # **Confirm Prediction**

# In[ ]:


pred_cnfrm = aus_grp_d.loc[:,["date","Confirmed"]]


# **Prophet**

# **Model**

# In[ ]:


pr_data = pred_cnfrm.tail(10)
pr_data.columns = ['ds','y']
m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)
forecast


# **Prediction**

# In[ ]:


cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['Date','Confirm']
cnfrm.head(10)


# **Plotting Prediction**

# In[ ]:


import plotly.offline as py
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')


# **Arima**

# In[ ]:


confirm_cs = pred_cnfrm.cumsum()
confirm_cs['date1'] = pred_cnfrm['date']
confirm_cs = confirm_cs.drop('date',axis=1)
arima_data = confirm_cs
arima_data.columns = ['count','confirmed_date']
arima_data = arima_data.head(63)
arima_data


# **Model**

# In[ ]:


model = ARIMA(arima_data['count'].values, order=(1, 2, 1))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()


# **Prediction**

# In[ ]:


forcast = fit_model.forecast(steps=7)
pred_y = forcast[0].tolist()
pred = pd.DataFrame(pred_y)
pred['pred']=  pred - pred.shift(1) 
pred


# In[ ]:


dataset_c = pd.DataFrame(pred_cnfrm['Confirmed'])


# In[ ]:


data = np.array(dataset_c).reshape(-1, 1)
train_data = dataset_c[:len(dataset_c)-8]
test_data = dataset_c[len(dataset_c)-8:]


# **LSTM**

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input =5
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 8)


# **Epochs vs Loss**

# In[ ]:


losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# **Prediction**

# In[ ]:


lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
prediction


# # **Recovery**

# **Prophet**

# **Model**

# In[ ]:


pred_rec = aus_grp_d.loc[:,["date","Recovered"]]
pr_data_r = pred_rec.tail(10)
pr_data_r.columns = ['ds','y']
m=Prophet()
m.fit(pr_data_r)
future_r=m.make_future_dataframe(periods=15)
forecast_r=m.predict(future)
forecast_r


# **Prediction**

# In[ ]:


rec = forecast_r.loc[:,['ds','trend']]
rec = rec[rec['trend']>0]
rec.columns = ['Date','Recovered']
rec.head(10)


# **Plotting Prediction**

# In[ ]:


fig_r = plot_plotly(m, forecast_r)
py.iplot(fig_r) 

fig_r = m.plot(forecast_r,xlabel='Date',ylabel='Recovered Count')


# **Arima**

# In[ ]:


rec_cs = pred_rec.cumsum()
rec_cs['date1'] = pred_cnfrm['date']
rec_cs = rec_cs.drop('date',axis=1)
arima_data_r = rec_cs
arima_data_r.columns = ['count','confirmed_date']
arima_data_r = arima_data_r.head(63)
arima_data_r


# **Model**

# In[ ]:


model = ARIMA(arima_data_r['count'].values, order=(1, 2, 1))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()


# **Prediction**

# In[ ]:


forcast_r = fit_model.forecast(steps=7)
pred_y_r = forcast_r[0].tolist()
pred = pd.DataFrame(pred_y_r)
pred['pred']=  pred - pred.shift(1) 
pred


# In[ ]:


dataset = pd.DataFrame(pred_rec['Recovered'])
data = np.array(dataset).reshape(-1, 1)
train_data = dataset[:len(dataset)-8]
test_data = dataset[len(dataset)-8:]


# **LSTM**

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input =5
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 7)


# **Epochs vs Loss**

# In[ ]:


losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# In[ ]:



lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# **Prediction**

# In[ ]:


prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
prediction


# # **Deaths**

# In[ ]:


pred_dth = aus_grp_d.loc[:,["date","Deaths"]]


# **Prophet**

# **Model**

# In[ ]:


pr_data_d = pred_dth.tail(10)
pr_data_d.columns = ['ds','y']
m=Prophet()
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=15)
forecast_d=m.predict(future)
forecast_d


# **Prediction**

# In[ ]:


dth = forecast_d.loc[:,['ds','trend']]
dth = dth[dth['trend']>0]
dth.columns = ['Date','Death']
dth.head(10)


# **Plotting Prediction**

# In[ ]:


fig = plot_plotly(m, forecast_d)
py.iplot(fig) 

fig = m.plot(forecast_d,xlabel='Date',ylabel='Confirmed Count')


# **Arima**

# In[ ]:


dth_cs = pred_dth.cumsum()
dth_cs['date1'] = pred_cnfrm['date']
dth_cs = dth_cs.drop('date',axis=1)
arima_data_d = dth_cs
arima_data_d.columns = ['count','confirmed_date']
arima_data_d = arima_data_d.head(66)
arima_data_d


# **Model**

# In[ ]:


model = ARIMA(arima_data_d['count'].values, order=(1, 2, 1))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()


# **Prediction**

# In[ ]:


forcast_d = fit_model.forecast(steps=6)
pred_y_d = forcast_d[0].tolist()
pred = pd.DataFrame(pred_y_d)
pred['pred'] = pred - pred.shift(1)
pred


# In[ ]:


dataset = pd.DataFrame(pred_dth['Deaths'])
data = np.array(dataset).reshape(-1, 1)
train_data = dataset[:len(dataset)-8]
test_data = dataset[len(dataset)-8:]


# **LSTM**

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input =5
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 10)


# **Epochs vs loss**

# In[ ]:


losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# In[ ]:


lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# **Prediction**

# In[ ]:


prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
prediction


# # Prevention
# *  To avoid the critical situation people are suggested to do following things  
# * Avoid contact with people who are sick.
# * Avoid touching your eyes, nose, and mouth.
# * Stay home when you are sick.
# * Cover your cough or sneeze with a tissue, then throw the tissue in the trash.
# * Clean and disinfect frequently touched objects and surfaces using a regular household
# * Wash your hands often with soap and water, especially after going to the bathroom; before eating; and after blowing your nose, coughing,   or sneezing. If soap and water are not readily available, use an alcohol-based hand sanitizer.
