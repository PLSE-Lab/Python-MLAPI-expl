#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/dVuyBgq2z5gVBkFtDc/giphy.gif)

# **Coronaviruses are a large family of viruses which may cause illness in animals or humans. In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The most recently discovered coronavirus causes coronavirus disease COVID-19.COVID-19 is the infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019.**
# * [Source](https://www.who.int/news-room/q-a-detail/q-a-coronaviruses)

# **Coronavirus Spread IN World**

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
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="natural earth",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[1000,100000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Realated Work**
# * For Analysis on Coronavirus(Canada), Click [here](https://www.kaggle.com/vanshjatana/coronavirus-in-canada)
# * For Analysis and Prediction on Coronavirus(Italy), Click [here](https://www.kaggle.com/vanshjatana/analysis-and-prediction-on-coronavirus-italy)
# *  For Analysis and Prediction on Coronavirus(South-Korea), Click [here](https://www.kaggle.com/vanshjatana/analysis-on-coronavirus)
# *  For Machine Learning on Cornovirus, Click [here](https://www.kaggle.com/vanshjatana/machine-learning-on-coronavirus)
# *  For report on Coronavirus, Click [here](https://www.researchgate.net/publication/339738108_Analysis_On_Coronavirus)
# 

# **Libraries**

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
from sklearn.cluster import KMeans
from fbprophet import Prophet


# **Symptoms of Coronavirus**
# * [Source](http://en.wikipedia.org/wiki/Coronavirus_disease_2019)

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


# **Bar Chart**

# In[ ]:


fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
fig.show()


# **Pie Chart**

# In[ ]:


plt.figure(figsize=(15,15))
plt.title('Symptoms of Coronavirus',fontsize=20)    
plt.pie(symptoms['percentage'],autopct='%1.1f%%')
plt.legend(symptoms['symptom'],loc='best')
plt.show() 


# In[ ]:


fig = px.treemap(symptoms, path=['symptom'], values='percentage',
                  color='percentage', hover_data=['symptom'],
                  color_continuous_scale='Rainbow')
fig.show()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in symptoms.symptom)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# **Reading File**

# In[ ]:


data1=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
age = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")


# **Looking into Data**

# In[ ]:


data1.head()


# **Shape of Data**

# In[ ]:


data1.shape


# **Checking for Null values**

# In[ ]:


data1.isna().sum()


# **Descripton of data**

# In[ ]:


data1.describe().T


# In[ ]:


daily = data1.sort_values(['Date','Country/Region','Province/State'])
latest = data1[data1.Date == daily.Date.max()]
latest.head()


# In[ ]:


data=latest.rename(columns={ "Country/Region": "country", "Province/State": "state","Confirmed":"confirm","Deaths": "death","Recovered":"recover"})
data.head()


# In[ ]:


dgc=data.groupby("country")[['confirm', 'death', 'recover']].sum().reset_index()

dgc.head()


# **Description of data group by religion**

# In[ ]:


dgc.describe().T


# In[ ]:


data.head()


# **World Map of Coronavirus**

# In[ ]:


import folium
worldmap = folium.Map(location=[32.4279,53.6880 ], zoom_start=4,tiles='Stamen Toner')

for Lat, Long, confirm,death,recover in zip(data['Lat'], data['Long'],data['confirm'],data['death'],data['recover']):
   folium.CircleMarker([Lat, Long],
                       radius=5,
                       color='red',
                     popup =('Confirm: ' + str(confirm) + '<br>'
                             'Recover: ' + str(recover) + '<br>'
                             'Death: ' + str(death) + '<br>'
                            ),
                       fill_color='red',
                       fill_opacity=0.7 ).add_to(worldmap)
worldmap


# **Daily Count In Coronavirus**

# **Based on Confirmation**

# In[ ]:


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[1000,50000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Based on Deaths**

# In[ ]:


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Recovered", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[100,10000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Based on Recovery**

# In[ ]:


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Deaths", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[100,10000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Based on Death**

# In[ ]:


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Active", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[100,10000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# **Confirm Cases**

# In[ ]:


fig = px.bar(dgc[['country', 'confirm']].sort_values('confirm', ascending=False), 
             y="confirm", x="country", color='country', 
             log_y=True, template='ggplot2', title='Confirmed Cases')
fig.show()


# In[ ]:


fig = px.treemap(dgc, path=['country'], values='confirm',
                  color='confirm', hover_data=['country'],
                  color_continuous_scale='burgyl')
fig.show()


# **Recovered Cases**

# In[ ]:


fig = px.bar(dgc[['country', 'recover']].sort_values('recover', ascending=False), 
             y="recover", x="country", color='country', 
             log_y=True, template='ggplot2', title='Recovered Cases')
fig.show()


# In[ ]:


fig = px.treemap(dgc, path=['country'], values='recover',
                  color='recover', hover_data=['country'],
                  color_continuous_scale='burgyl')
fig.show()


# **Deaths**

# In[ ]:


fig = px.bar(dgc[['country', 'death']].sort_values('death', ascending=False), 
             y="death", x="country", color='country', 
             log_y=True, template='ggplot2', title='Death')
fig.show()


# In[ ]:


fig = px.treemap(dgc, path=['country'], values='death',
                  color='death', hover_data=['country'],
                  color_continuous_scale='burgyl')
fig.show()


# In[ ]:


data1.head()


# **For Iran**

# In[ ]:


iran_data = data1[data1['Country/Region']=='Iran']
idata = iran_data.tail(22)
idata.head()


# In[ ]:


iran_data.describe().T


# In[ ]:


cmap1 = iran_data
cmap1  = cmap1.groupby(['Date', 'Country/Region','Lat','Long'])['Confirmed', 'Deaths', 'Recovered'].max()


cmap1 = cmap1.reset_index()
cmap1.head()
cmap1['size'] = cmap1['Confirmed']*90000000
cmap1
fig = px.scatter_mapbox(cmap1, lat="Lat", lon="Long",
                     color="Confirmed", size='size',hover_data=['Confirmed','Recovered','Deaths'],
                     color_continuous_scale='burgyl',
                     animation_frame="Date", 
                     title='Spread total cases over time')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=3)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# **Confirmation vs Recoverey vs Death**

# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Confirmed,label="Confirm")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation",fontsize=50)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Recovered,label="Recovery")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Recoverey",fontsize=50)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Death",fontsize=50)
plt.show()


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(idata.Date, idata.Confirmed,label="Confirm")
plt.bar(idata.Date, idata.Recovered,label="Recovery")
plt.bar(idata.Date, idata.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death",fontsize=50)
plt.show()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=idata,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered", data=idata,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=idata,
             color="blue",label = "Death")
plt.plot(idata.Date,idata.Confirmed,zorder=1,color="black")
plt.plot(idata.Date,idata.Recovered,zorder=1,color="red")
plt.plot(idata.Date,idata.Deaths,zorder=1,color="blue")


# **Daily Change**

# In[ ]:


idata['Confirmed_new'] = idata['Confirmed']-idata['Confirmed'].shift(1)
idata['Recovered_new'] = idata['Recovered']-idata['Recovered'].shift(1)
idata['Deaths_new'] = idata['Deaths']-idata['Deaths'].shift(1)


# In[ ]:


idata = idata.fillna(0)


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


idata.head()


# In[ ]:


idata.describe().T


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


# **Groping By Date**

# In[ ]:


dgd=data.groupby("Date")[['confirm', 'death', 'recover']].sum().reset_index()

dgd.head()


# **Ratio and Percentage of Recovery and Death after Confirmation**

# In[ ]:


r_cm = float(dgd.recover/dgd.confirm)
d_cm = float(dgd.death/dgd.confirm)


# In[ ]:


print("The percentage of recovery after confirmation is "+ str(r_cm*100) )
print("The percentage of death after confirmation is "+ str(d_cm*100) )


# **Growth Factor and Ratio**

# This snippet of code is taken by [covid-19-predictions-growth-factor-and-calculus](https://www.kaggle.com/dferhadi/covid-19-predictions-growth-factor-and-calculus) , do check this kernel by [Daner Ferhadi](https://www.kaggle.com/dferhadi)

# In[ ]:


global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


# This functions smooths data, thanks to Dan Pearson. We will use it to smooth the data for growth factor.
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
    f, ax = plt.subplots(figsize=(15,5))
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


plot_country_active_confirmed_recovered('Iran')


#  > **For Confirmed Cases**

# **Prophet**

# In[ ]:


prophet=iran_data.iloc[: , [4,5 ]].copy() 
prophet.head()
prophet.columns = ['ds','y']
prophet.head()


# In[ ]:


m=Prophet()
m.fit(prophet)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)
forecast


# In[ ]:


cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm']
cnfrm.head()


# **Graphical Representation of Predicted Death**

# In[ ]:


figure = plot_plotly(m, forecast)
py.iplot(figure) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')


# In[ ]:


figure=m.plot_components(forecast)


# > **For Recover Cases**

# This prediction is truly based on the dataset depend on the current situation. In future if we able to get vaccine there will be gradual changes in recovery

# **Prophet**

# In[ ]:


prophet_rec=iran_data.iloc[: , [4,7 ]].copy() 
prophet_rec.head()
prophet_rec.columns = ['ds','y']
prophet_rec.head()


# In[ ]:


m1=Prophet()
m1.fit(prophet_rec)
future_rec=m1.make_future_dataframe(periods=15)
forecast_rec=m1.predict(future_rec)
forecast_rec


# In[ ]:


rec = forecast_rec.loc[:,['ds','trend']]
rec = rec[rec['trend']>0]
rec=rec.tail(15)
rec.columns = ['Date','Recovery']
rec.head()


# **Graphical Representation of Predicted Recovery**

# In[ ]:


figure_rec = plot_plotly(m1, forecast_rec)
py.iplot(figure_rec) 

figure_rec = m1.plot(forecast_rec,xlabel='Date',ylabel='Recovery Count')


# In[ ]:


figure_rec=m1.plot_components(forecast_rec)


# **For Death**

# **Prophet**

# In[ ]:


prophet_dth=iran_data.iloc[: , [4,6 ]].copy() 
prophet_dth.head()
prophet_dth.columns = ['ds','y']
prophet_dth.head()


# In[ ]:


m2=Prophet()
m2.fit(prophet_dth)
future_dth=m2.make_future_dataframe(periods=15)
forecast_dth=m2.predict(future_dth)
forecast_dth


# In[ ]:


dth = forecast_dth.loc[:,['ds','trend']]
dth = dth[dth['trend']>0]
dth=dth.tail(15)
dth.columns = ['Date','Death']
dth.head()


# **Graphical Representation of Predicted Death**

# In[ ]:


figure_dth = plot_plotly(m2, forecast_dth)
py.iplot(figure_dth) 

figure_dth = m2.plot(forecast_dth,xlabel='Date',ylabel='Death Count')


# In[ ]:


figure_dth=m2.plot_components(forecast_dth)


# **How future looks like!!**

# In[ ]:


prediction = cnfrm
prediction['Recover'] = rec.Recovery
prediction['Death'] = dth.Death
prediction.head()


# **Future Raio and percentages**

# In[ ]:


pr_pps = float(prediction.Recover.sum()/prediction.Confirm.sum())
pd_pps = float(prediction.Death.sum()/prediction.Confirm.sum())


# 

# In[ ]:


print("The percentage of Predicted recovery after confirmation is "+ str(pr_pps*100) )
print("The percentage of Predicted Death after confirmation is "+ str(pd_pps*100) )


# **Comparision with other country**

# In[ ]:


comp = pd.read_excel('/kaggle/input/covid19327/COVID-19-3.27-top30-500.xlsx')


# In[ ]:


comp.head()


# In[ ]:


comp_table = pd.DataFrame(comp.describe().T)
comp_table


# In[ ]:


comp = comp.loc[:,["Canada","US","Iran","China"]]


# In[ ]:


comp.plot()


# **Prevention**
# To avoid the critical situation people are suggested to do following things
# 
# * Avoid contact with people who are sick.
# * Avoid touching your eyes, nose, and mouth.
# * Stay home when you are sick.
# * Cover your cough or sneeze with a tissue, then throw the tissue in the trash.
# * Clean and disinfect frequently touched objects and surfaces using a regular household
# * Wash your hands often with soap and water, especially after going to the bathroom; before eating; and after blowing your nose, coughing, or sneezing. If soap and water are not readily available, use an alcohol-based hand sanitizer.
