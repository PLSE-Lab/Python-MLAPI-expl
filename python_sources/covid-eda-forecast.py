#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/kcUYFhoCZwWF3fivnI/giphy.gif)

# * <font size="5" color="blue">Contents</font>
# 
# * [Loading Libraries & EDA](#1)
#     
# 
# 
#     

# **Please upvote in case you find Notebook helpful**

# 
# ## [Loading Libraries & EDA]() <a id="1" ></a>

# In[ ]:


get_ipython().system('pip install pycountry_convert')
get_ipython().system('pip install folium')
get_ipython().system('pip install plotly')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker 
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests


get_ipython().run_line_magic('matplotlib', 'inline')


# ###  <font color='red' size='3'>Applying Scrapping</font>
# 

# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup

req = requests.get('https://www.worldometers.info/coronavirus/')
soup = BeautifulSoup(req.text, "lxml")

df_country = soup.find('div',attrs={"id" : "nav-tabContent"}).find('table',attrs={"id" : "main_table_countries_today"}).find_all('tr')
arrCountry = []
for i in range(8,len(df_country)-1):
    tmp = df_country[i].find_all('td')
    if (tmp[0].string.find('<a') == -1):
        country = [tmp[0].string]
    else:
        country = [tmp[0].a.string] # Country
    for j in range(1,12):
        if (str(tmp[j].string) == 'None' or str(tmp[j].string) == ' '):
            country = country + [0]
        else:
            country = country + [float(tmp[j].string.replace(',','').replace('+',''))]
    arrCountry.append(country)
df_worlddata = pd.DataFrame(arrCountry)
df_worlddata.columns = ['Country','Total Cases','Cases','Total Deaths','Deaths','Total Recovers','Active','Serious Critical',
                         'Total Cases/1M pop','Deaths/1M pop','Total Test','Tests/1M pop']
for i in range(0,len(df_worlddata)):
    df_worlddata['Country'].iloc[i] = df_worlddata['Country'].iloc[i].strip()


# In[ ]:


df_worlddata.head()


# In[ ]:


df_worlddata.style.background_gradient(cmap='Wistia')


# In[ ]:


df_worlddata = df_worlddata[df_worlddata.Country != 'World']
df_worlddata.index = df_worlddata["Country"]
df_worlddata = df_worlddata.drop(['Country'],axis=1)
df_worlddata.head()


# <hr>
# 1. Validating Testing Data around the world
# 2. How different counties are performing test wise around the world
# 3. Let us visulize as per total number of test and test among Million people
# <hr>

# In[ ]:


df_test=df_worlddata.drop(['Total Cases','Cases','Total Deaths','Deaths','Total Recovers','Active','Serious Critical',
                           'Total Cases/1M pop','Deaths/1M pop'],axis=1)


# In[ ]:


df_test.head()


# <hr>
# **Identifying Top 20 Countries**
# 1. No of total test done
# 2. Number of test as per total  1 million of population

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_test.sort_values('Tests/1M pop')["Tests/1M pop"].index[-50:],df_test.sort_values('Tests/1M pop')["Tests/1M pop"].values[-50:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Tests/1M pop ",fontsize=18)
plt.title("Top Countries (Tests/1M pop )",fontsize=20)
plt.grid(alpha=0.3)


# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_test.sort_values('Total Test')["Total Test"].index[-50:],df_test.sort_values('Total Test')["Total Test"].values[-50:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Total Test",fontsize=18)
plt.title("Top Countries (Total Test )",fontsize=20)
plt.grid(alpha=0.3)


# <hr>
# ** As per figure above you can notice following points **
# 1. USA is doing maximum number of test nowdays and that is the reason they are having so much count nowdays
# 2. India have also increase number of test at daily basis now
# 3. South Korea, despite being less number of cases have done more number of test, and that is the reason they are able to make the curve flat after their count of 9 k cases
# 

# <hr>
# ### Let us now analyze top 20 countries with number of Total cases
# **Will analyze Top counties as per confirmed, Most number of deaths, recovered and Critical case(Requiring ICU)**
# 
# 

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Total Cases')["Total Cases"].index[-20:],df_worlddata.sort_values('Total Cases')["Total Cases"].values[-20:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Total Cases",fontsize=18)
plt.title("Top Countries (Total #)",fontsize=20)
plt.grid(alpha=0.3)


# ### Top Countries as per Active cases ###

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Active')["Active"].index[-20:],df_worlddata.sort_values('Active')["Active"].values[-20:],color="darkcyan")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Active",fontsize=18)
plt.title("Top Countries (Active #)",fontsize=20)
plt.grid(alpha=0.3)


# <hr>
# **Let us now identify critical cases**
# 1. This will be most important as this will identify the number of deaths in coming days
# 2. USA will have more number of death in coming day as they are having more number of serious cases
# 3. France and Spain followed by USA

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Serious Critical')["Serious Critical"].index[-20:],df_worlddata.sort_values('Serious Critical')["Serious Critical"].values[-20:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Serious Critical",fontsize=18)
plt.title("Top Countries (Critical #)",fontsize=20)
plt.grid(alpha=0.3)


# In[ ]:


df_worlddata.head()


# In[ ]:


# Retriving Dataset
confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# Depricated
# df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")


# In[ ]:


covid_country = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")


# In[ ]:


confirmed = confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})
deaths = deaths.rename(columns={"Province/State":"state","Country/Region": "country"})
recovered = recovered.rename(columns={"Province/State":"state","Country/Region": "country"})
covid_country = covid_country.rename(columns={"Country_Region": "country"})
covid_country["Active"] = covid_country["Confirmed"]-covid_country["Recovered"]-covid_country["Deaths"]


# In[ ]:


confirmed.head(),deaths.head(),covid_country.head(),recovered.head()


# In[ ]:


def plot_params(ax,axis_label= None, plt_title = None,label_size=15, axis_fsize = 15, title_fsize = 20, scale = 'linear' ):
    # Tick-Parameters
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', width=1,labelsize=label_size)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3, color='0.8')
    
    # Grid
    plt.grid(lw = 1, ls = '-', c = "0.7", which = 'major')
    plt.grid(lw = 1, ls = '-', c = "0.9", which = 'minor')

    # Plot Title
    plt.title( plt_title,{'fontsize':title_fsize})
    
    # Yaxis sacle
    plt.yscale(scale)
    plt.minorticks_on()
    # Plot Axes Labels
    xl = plt.xlabel(axis_label[0],fontsize = axis_fsize)
    yl = plt.ylabel(axis_label[1],fontsize = axis_fsize)
    
def visualize_covid_cases(confirmed, deaths, continent=None , country = None , state = None, period = None, figure = None, scale = "linear"):
    x = 0
    if figure == None:
        f = plt.figure(figsize=(10,10))
        # Sub plot
        ax = f.add_subplot(111)
    else :
        f = figure[0]
        # Sub plot
        ax = f.add_subplot(figure[1],figure[2],figure[3])
    
    plt.tight_layout(pad=10, w_pad=5, h_pad=5)
    
    stats = [confirmed, deaths]
    label = ["Confirmed", "Deaths"]
    
    if continent != None:
        params = ["continent",continent]
    elif country != None:
        params = ["country",country]
    else: 
        params = ["All", "All"]
    color = ["darkcyan","crimson"]
    marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=4, markerfacecolor='#ffffff')
    for i,stat in enumerate(stats):
        if params[1] == "All" :
            cases = np.sum(np.asarray(stat.iloc[:,5:]),axis = 0)[x:]
        else :
            cases = np.sum(np.asarray(stat[stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        date = np.arange(1,cases.shape[0]+1)[x:]
        plt.plot(date,cases,label = label[i]+" (Total : "+str(cases[-1])+")",color=color[i],**marker_style)

    if params[1] == "All" :
        Total_confirmed = np.sum(np.asarray(stats[0].iloc[:,5:]),axis = 0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1].iloc[:,5:]),axis = 0)[x:]
    else :
        Total_confirmed =  np.sum(np.asarray(stats[0][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        
    text = "From "+stats[0].columns[5]+" to "+stats[0].columns[-1]+"\n"
    text += "Mortality rate : "+ str(int(Total_deaths[-1]/(Total_confirmed[-1])*10000)/100)+"\n"
    text += "Last 5 Days:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-6])+"\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-6])+"\n"
    text += "Last 24 Hours:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-2])+"\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-2])+"\n"
    
    plt.text(0.02, 0.78, text, fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.4))
    
    # Plot Axes Labels
    axis_label = ["Days ("+confirmed.columns[5]+" - "+confirmed.columns[-1]+")","No of Cases"]
    
    # Plot Parameters
    plot_params(ax,axis_label,scale = scale)
    
    # Plot Title
    if params[1] == "All" :
        plt.title("COVID-19 Cases World",{'fontsize':25})
    else:   
        plt.title("COVID-19 Cases for "+params[1] ,{'fontsize':25})
        
    # Legend Location
    l = plt.legend(loc= "best",fontsize = 15)
    
    if figure == None:
        plt.show()
        
def get_total_cases(cases, country = "All"):
    if(country == "All") :
        return np.sum(np.asarray(cases.iloc[:,5:]),axis = 0)[-1]
    else :
        return np.sum(np.asarray(cases[cases["country"] == country].iloc[:,5:]),axis = 0)[-1]
    
def get_mortality_rate(confirmed,deaths, continent = None, country = None):
    if continent != None:
        params = ["continent",continent]
    elif country != None:
        params = ["country",country]
    else :
        params = ["All", "All"]
    
    if params[1] == "All" :
        Total_confirmed = np.sum(np.asarray(confirmed.iloc[:,5:]),axis = 0)
        Total_deaths = np.sum(np.asarray(deaths.iloc[:,5:]),axis = 0)
        mortality_rate = np.round((Total_deaths/Total_confirmed)*100,2)
    else :
        Total_confirmed =  np.sum(np.asarray(confirmed[confirmed[params[0]] == params[1]].iloc[:,5:]),axis = 0)
        Total_deaths = np.sum(np.asarray(deaths[deaths[params[0]] == params[1]].iloc[:,5:]),axis = 0)
        mortality_rate = np.round((Total_deaths/Total_confirmed)*100,2)
    
    return np.nan_to_num(mortality_rate)
def dd(date1,date2):
    return (datetime.strptime(date1,'%m/%d/%y') - datetime.strptime(date2,'%m/%d/%y')).days


#out = "output/"


# In[ ]:


country_df = covid_country.copy().drop(['Lat','Long_','Last_Update'],axis =1)
country_df.index = country_df["country"]
country_df = country_df.drop(['country'],axis=1)


# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(country_df.sort_values('Confirmed')["Confirmed"].index[-20:],country_df.sort_values('Confirmed')["Confirmed"].values[-20:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top Countries (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)
#plt.savefig(out+'Top Countries (Confirmed Cases).png')


# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(country_df.sort_values('Deaths')["Deaths"].index[-20:],country_df.sort_values('Deaths')["Deaths"].values[-20:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top Countries (Death #)",fontsize=20)
plt.grid(alpha=0.3)
#plt.savefig(out+'Top Countries (Death #).png')


# In[ ]:


pd.DataFrame(country_df.sum()).transpose().style.background_gradient(cmap='Wistia',axis=1)


# In[ ]:


country_df.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='PuBu')


# In[ ]:


world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)
for i in range(0,len(confirmed)):
    folium.Circle(
        location=[confirmed.iloc[i]['Lat'], confirmed.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+confirmed.iloc[i]['country']+"</h5>"+
                    "<div style='text-align:center;'>"+str(np.nan_to_num(confirmed.iloc[i]['state']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(confirmed.iloc[i,-1])+"</li>"+
        "<li>Deaths:   "+str(deaths.iloc[i,-1])+"</li>"+
        "<li>Mortality Rate:   "+str(np.round(deaths.iloc[i,-1]/(confirmed.iloc[i,-1]+1.00001)*100,2))+"</li>"+
        "</ul>"
        ,
        radius=(int((np.log(confirmed.iloc[i,-1]+1.00001)))+0.2)*50000,
        color='#ff6600',
        fill_color='#ff8533',
        fill=True).add_to(world_map)

world_map


# In[ ]:


temp_df = pd.DataFrame(country_df['Confirmed'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="country",
                    color=np.log10(temp_df.iloc[:,-1]), # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    hover_data=["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Confirmed Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# **Forecasting Model**

# In[ ]:


temp_df = pd.DataFrame(country_df['Deaths'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="country",
                    color=np.log10(temp_df.iloc[:,-1]+1), # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    hover_data=["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Deaths Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Deaths (Log Scale)",colorscale="Reds")
# fig.to_image("Global Heat Map deaths.png")
fig.show()


# In[ ]:


confirmed = confirmed.replace(np.nan, '', regex=True)
deaths = deaths.replace(np.nan, '', regex=True)


# In[ ]:


df_countries = confirmed.groupby(["country"]).sum()
df_countries = df_countries.sort_values(df_countries.columns[-1],ascending = False)
countries = df_countries[df_countries[df_countries.columns[-1]] >= 3500].index

cols =2
rows = int(np.ceil(countries.shape[0]/cols))
f = plt.figure(figsize=(20,8*rows))
for i,country in enumerate(countries):
    visualize_covid_cases(confirmed, deaths,country = country,figure = [f,rows,cols, i+1])

plt.show()


# In[ ]:


temp = confirmed.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(confirmed.columns[-1], ascending= False)

threshold = 50
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i,country in enumerate(temp.index):
    if i >= 9:
        if country != "India" and country != "Japan" :
            continue
    x = 91
    t = temp.loc[temp.index== country].values[0]
    t = t[t>threshold][:x]
     
    date = np.arange(0,len(t[:x]))
    xnew = np.linspace(date.min(), date.max(), 91)
    spl = make_interp_spline(date, t, k=1)  # type: BSpline
    power_smooth = spl(xnew)
    if country != "India":
        plt.plot(xnew,power_smooth,'-o',label = country,linewidth =3, markevery=[-1])
    else:
        marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='#ffffff')
        plt.plot(date,t,"-.",label = country,**marker_style)

plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0,91,7),[ "D "+str(i) for i in range(91)][::7])     

# Reference lines 
x = np.arange(0,18)
y = 2**(x+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/2+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/7+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/30+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)


# India is following trend similar to doulbe the cases in 4 days but it may increase the rate 
x = np.arange(0,26)
y = 2**(x/4+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "Red")
plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)

# plot Params
plt.xlabel("Day",fontsize=17)
plt.ylabel("Number of Confirmed Cases",fontsize=17)
plt.title("Trend Comparison of Different Countries\n and India (confirmed) ",fontsize=22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which="both")
#plt.savefig(out+'Trend Comparison with India (confirmed).png')
plt.show()


# In[ ]:


temp = deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(deaths.columns[-1], ascending= False)

threshold = 10
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i,country in enumerate(temp.index):
    if i > 10:
        break
    x = 91
    t = temp.loc[temp.index== country].values[0]
    t = t[t>threshold][:x]
     
    date = np.arange(0,len(t[:x]))
    xnew = np.linspace(date.min(), date.max(), 91)
    spl = make_interp_spline(date, t, k=1)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew,power_smooth,'-o',label = country,linewidth =3, markevery=[-1])


plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0,91,7),[ "D "+str(i) for i in range(91)][::7])     

# Reference lines 
x = np.arange(0,18)
y = 2**(x+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/2+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/7+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/30+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

# plot Params
plt.xlabel("Days",fontsize=17)
plt.ylabel("Number of Deaths",fontsize=17)
plt.title("Trend Comparison of Different Countries \n(Deaths)",fontsize=22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which="both")
#plt.savefig(out+'Trend Comparison countries deaths.png')
plt.show()


# **Let us explore data for India**

# In[ ]:


india_data = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()
india_covid = pd.io.json.json_normalize(india_data['data']['statewise'])
india_covid = india_covid.set_index("state")


# In[ ]:


total = india_covid.sum()
total.name = "Total"
pd.DataFrame(total).transpose().style.background_gradient(cmap='Wistia',axis=1)


# **State Wise Comparison**

# In[ ]:


india_covid.style.background_gradient(cmap='Wistia')


# **State with more then 5 deaths**

# In[ ]:


india_covid[india_covid['deaths'] > 5].style.background_gradient(cmap='PuBu')


# In[ ]:



locations = {
    "Kerala" : [10.8505,76.2711],
    "Maharashtra" : [19.7515,75.7139],
    "Karnataka": [15.3173,75.7139],
    "Telangana": [18.1124,79.0193],
    "Uttar Pradesh": [26.8467,80.9462],
    "Rajasthan": [27.0238,74.2179],
    "Gujarat":[22.2587,71.1924],
    "Delhi" : [28.7041,77.1025],
    "Punjab":[31.1471,75.3412],
    "Tamil Nadu": [11.1271,78.6569],
    "Haryana": [29.0588,76.0856],
    "Madhya Pradesh":[22.9734,78.6569],
    "Jammu and Kashmir":[33.7782,76.5762],
    "Ladakh": [34.1526,77.5770],
    "Andhra Pradesh":[15.9129,79.7400],
    "West Bengal": [22.9868,87.8550],
    "Bihar": [25.0961,85.3131],
    "Chhattisgarh":[21.2787,81.8661],
    "Chandigarh":[30.7333,76.7794],
    "Uttarakhand":[30.0668,79.0193],
    "Himachal Pradesh":[31.1048,77.1734],
    "Goa": [15.2993,74.1240],
    "Odisha":[20.9517,85.0985],
    "Andaman and Nicobar Islands": [11.7401,92.6586],
    "Puducherry":[11.9416,79.8083],
    "Manipur":[24.6637,93.9063],
    "Mizoram":[23.1645,92.9376],
    "Assam":[26.2006,92.9376],
    "Meghalaya":[25.4670,91.3662],
    "Tripura":[23.9408,91.9882],
    "Arunachal Pradesh":[28.2180,94.7278],
    "Jharkhand" : [23.6102,85.2799],
    "Nagaland": [26.1584,94.5624],
    "Sikkim": [27.5330,88.5122],
    "Dadra and Nagar Haveli":[20.1809,73.0169],
    "Lakshadweep":[10.5667,72.6417],
    "Daman and Diu":[20.4283,72.8397]    
}
india_covid["Lat"] = ""
india_covid["Long"] = ""
for index in india_covid.index :
    india_covid.loc[india_covid.index == index,"Lat"] = locations[index][0]
    india_covid.loc[india_covid.index == index,"Long"] = locations[index][1]


# In[ ]:


# url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
# state_json = requests.get(url).json()
india = folium.Map(location=[23,80], zoom_start=4,max_zoom=6,min_zoom=4,height=500,width="80%")
for i in range(0,len(india_covid[india_covid['confirmed']>0].index)):
    folium.Circle(
        location=[india_covid.iloc[i]['Lat'], india_covid.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+india_covid.iloc[i].name+"</h5>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(india_covid.iloc[i]['confirmed'])+"</li>"+
        "<li>Active:   "+str(india_covid.iloc[i]['active'])+"</li>"+
        "<li>Recovered:   "+str(india_covid.iloc[i]['recovered'])+"</li>"+
        "<li>Deaths:   "+str(india_covid.iloc[i]['deaths'])+"</li>"+
        
        "<li>Mortality Rate:   "+str(np.round(india_covid.iloc[i]['deaths']/(india_covid.iloc[i]['confirmed']+1)*100,2))+"</li>"+
        "</ul>"
        ,
        radius=(int(np.log2(india_covid.iloc[i]['confirmed']+1)))*15000,
        color='#ff6600',
        fill_color='#ff8533',
        fill=True).add_to(india)

india


# **USA**

# ### Forecasting Model

# In[ ]:


data= pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")  


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from random import random
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm

def RMSLE(pred,actual):
    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))


# In[ ]:


pd.set_option('mode.chained_assignment', None)
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
train['Province_State'].fillna('', inplace=True)
test['Province_State'].fillna('', inplace=True)
train['Date'] =  pd.to_datetime(train['Date'])
test['Date'] =  pd.to_datetime(test['Date'])
train = train.sort_values(['Country_Region','Province_State','Date'])
test = test.sort_values(['Country_Region','Province_State','Date'])


# In[ ]:


train.shape,train.head(),train.info()


# **Fixing Errors**

# In[ ]:


train[['ConfirmedCases', 'Fatalities']] = train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].transform('cummax') 


# **Forecast with BayesianRidge**

# In[ ]:


from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

feature_day = [1,20,50,100,200,500,1000]
def CreateInput(data):
    feature = []
    for day in feature_day:
        #Get information in train data
        data.loc[:,'Number day from ' + str(day) + ' case'] = 0
        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        
        else:
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       
        for i in range(0, len(data)):
            if (data['Date'].iloc[i] > fromday):
                day_denta = data['Date'].iloc[i] - fromday
                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 
        feature = feature + ['Number day from ' + str(day) + ' case']
    
    return data[feature]
pred_data_all = pd.DataFrame()
with tqdm(total=len(train['Country_Region'].unique())) as pbar:
    for country in train['Country_Region'].unique():
        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():
            df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
            df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
            X_train = CreateInput(df_train)
            y_train_confirmed = df_train['ConfirmedCases'].ravel()
            y_train_fatalities = df_train['Fatalities'].ravel()
            X_pred = CreateInput(df_test)

            # Define feature to use by X_pred
            feature_use = X_pred.columns[0]
            for i in range(X_pred.shape[1] - 1,0,-1):
                if (X_pred.iloc[0,i] > 0):
                    feature_use = X_pred.columns[i]
                    break
            idx = X_train[X_train[feature_use] == 0].shape[0]          
            adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)
            adjusted_y_train_confirmed = y_train_confirmed[idx:]
            adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)
              
            adjusted_X_pred = X_pred[feature_use].values.reshape(-1, 1)

            model = make_pipeline(PolynomialFeatures(2), BayesianRidge())
            model.fit(adjusted_X_train,adjusted_y_train_confirmed)                
            y_hat_confirmed = model.predict(adjusted_X_pred)

            model.fit(adjusted_X_train,adjusted_y_train_fatalities)                
            y_hat_fatalities = model.predict(adjusted_X_pred)

            pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
            pred_data['ConfirmedCases_hat'] = y_hat_confirmed
            pred_data['Fatalities_hat'] = y_hat_fatalities
            pred_data_all = pred_data_all.append(pred_data)
        pbar.update(1)
    
df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')
df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0
df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

df_val_1 = df_val.copy()


# In[ ]:


RMSLE(df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases'].values,df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases_hat'].values)


# In[ ]:


RMSLE(df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities'].values,df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities_hat'].values)


# In[ ]:


val_score = []
for country in df_val['Country_Region'].unique():
    df_val_country = df_val[(df_val['Country_Region'] == country) & (df_val['Fatalities'].isnull() == False)]
    val_score.append([country, RMSLE(df_val_country['ConfirmedCases'].values,df_val_country['ConfirmedCases_hat'].values),RMSLE(df_val_country['Fatalities'].values,df_val_country['Fatalities_hat'].values)])
    
df_val_score = pd.DataFrame(val_score) 
df_val_score.columns = ['Country','ConfirmedCases_Scored','Fatalities_Scored']
df_val_score.sort_values('ConfirmedCases_Scored', ascending = False)


# In[ ]:


country = "India"
df_val = df_val_1
df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()
df_train = train[(train['Country_Region'].isin(df_country['Country_Region'].unique())) & (train['ConfirmedCases'] > 0)].groupby(['Date']).sum().reset_index()

idx = df_country[((df_country['ConfirmedCases'].isnull() == False) & (df_country['ConfirmedCases'] > 0))].shape[0]
fig = px.line(df_country, x="Date", y="ConfirmedCases_hat", title='Forecast Total Cases of ' + df_country['Country_Region'].values[0])
fig.add_scatter(x=df_train['Date'], y=df_train['ConfirmedCases'], mode='lines', name="Actual train", showlegend=True)
fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['ConfirmedCases'][0:idx], mode='lines', name="Actual test", showlegend=True)
fig.show()

fig = px.line(df_country, x="Date", y="Fatalities_hat", title='Forecast Total Fatalities of ' + df_country['Country_Region'].values[0])
fig.add_scatter(x=df_train['Date'], y=df_train['Fatalities'], mode='lines', name="Actual train", showlegend=True)
fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['Fatalities'][0:idx], mode='lines', name="Actual test", showlegend=True)

fig.show()


# In[ ]:


df_val = df_val_1
submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('submission.csv', index=False)
submission


# In[ ]:




