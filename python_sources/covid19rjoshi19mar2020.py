#!/usr/bin/env python
# coding: utf-8

# <a id="top"></a>
# <h4> Hello,this is my effort at Data Visualisation using the Covid-19 dataset maintained by the John Hopkins Medical College</h4>
# 
# #### Table of Contents
# 
# 1. [Basic Numbers](#BasicNumbers)
# 2. [Top Twenty Countries with Loss of Lives](#TableTopTwenty)
# 3. [Global increase in Cases,Deaths and Recoveries](#GlobalGraph)
# 4. [Loss of life Top 20](#barTop20)
# 5. [Confirmed Cases: Reported and Trends](#ConfirmedProjections)
# 6. [Deaths: Reported and Trends](#DeathsProjections)

# In[ ]:


#Import the necessary stuff
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns

from IPython.core.display import display, HTML #for embedding html tags for navigation etc.

#below are required for using plotly
import plotly as py
import plotly.graph_objects as go
# import plotly.offline as pyo
from plotly.offline import init_notebook_mode,iplot,plot
# init_notebook_mode(connected = True)
import plotly.express as px
# import plotly.io as pio

get_ipython().run_line_magic('matplotlib', 'inline')

#Read the files directly from source. Uncomment the below cell if working offline on your system
#url1 ="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv";
#url2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
#url3 ="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

#new urls since 24th March:

url1 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
url2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
url3 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

#Create dataframes from urls
dfRawConfirmed = pd.read_csv(url1, index_col = None);
dfRawDeaths = pd.read_csv(url2,index_col=None);

dfRawRecovered = pd.read_csv(url3, index_col = None);

#Replace "Province/State" column with "State" and "Country/Region" with Country

dfRawDeaths.rename(columns = {"Country/Region":"Country","Province/State":"State"},inplace= True)
dfRawConfirmed.rename(columns ={"Country/Region":"Country","Province/State":"State"},inplace= True)
dfRawRecovered.rename(columns = {"Country/Region":"Country","Province/State":"State"},inplace= True)

#Convert from Wide to long format
dfCovDeaths= dfRawDeaths.melt(id_vars=dfRawDeaths.columns[:4],value_vars=dfRawDeaths.columns[4:],
                              value_name= "Count", var_name="Date")
dfCovConfirmed = dfRawConfirmed.melt(id_vars=dfRawConfirmed.columns[:4],value_vars=dfRawConfirmed.columns[4:],
                              value_name= "Count", var_name="Date")
dfCovRecovered = dfRawRecovered.melt(id_vars=dfRawRecovered.columns[:4],value_vars=dfRawRecovered.columns[4:],
                              value_name= "Count", var_name="Date")

#Add a Category Column
dfCovDeaths["Category"] = "Deaths"
dfCovConfirmed["Category"]="Confirmed_Cases"
dfCovRecovered["Category"] = "Recovered"

#Convert Date field to date type
dfCovDeaths.Date = pd.to_datetime(dfCovDeaths["Date"])
dfCovConfirmed.Date =pd.to_datetime(dfCovConfirmed["Date"])
dfCovRecovered.Date = pd.to_datetime(dfCovRecovered["Date"])

#Save to .csv file for offline use
dfCovDeaths.to_csv("DFCovidDeaths.csv",index = False,index_label=False)
dfCovConfirmed.to_csv("DFCovidConfirmedCases.csv",index_label=False,index = False)
dfCovRecovered.to_csv("DFCovidRecoveredCases.csv",index_label=False,index = False)

#create a dataframe for list of countries and state
dfStateCountryList =dfCovDeaths.drop_duplicates(["State","Country"])[["State","Country"]]
dfStateCountryList =dfStateCountryList.astype(str)

#Join three dataframes to get one combined dataframe
df_temp = pd.merge(dfCovDeaths,dfCovConfirmed,on=['State','Country','Date'])
dfMain = pd.merge(df_temp,dfCovRecovered, on =['State','Country','Date'])
#Rename Columns
dfMain.rename(columns={"Count_x":"Deaths","Count_y":"Confirmed","Count":"Recovered"},inplace = True)
dfMain = dfMain[['State','Country','Date','Deaths','Confirmed','Recovered']]


# In[ ]:


#Process the dataframe to add deltas for deaths, confirmed cases and recoveries. delta is increase over previous value
temp=dfMain.copy()
temp['DeltaDeaths']=0
temp['DeltaConfirmed']=0
temp['DeltaRecovered']=0
temp['DeathsDeltaPercent']=0
temp['MortalityPercent']=0
temp['RecoveryPercent']=0
# temp['ConfirmedDeltaPercent']=0
# temp['RecoveredDeltaPercent']=0
varState = ""
varCountry=""

#for j in [16,155]:

for j in range(len(dfStateCountryList)):
    varState = dfStateCountryList.loc[j,'State']
    varCountry = dfStateCountryList.loc[j,'Country']
# Handle cases like Cote d'Ivore   
    if "'" in varState:
        varState.replace("'","\'")
    if "'" in varCountry:
        varCountry.replace("'","\'")
        
#If the data row does not have a province/state
    if (varState =='nan'):
        varStr = "Country == \""+ varCountry+"\""
    else:
        varStr = "(Country == \"" +  varCountry+"\")" + " & " +" (State =="+ "\""+ varState +"\")"
#     print(str(j)+'/'+str(len(dfStateCountryList)))
    y1=temp.query(varStr).copy()
    y2 = y1.shift(1)
    for i in y2.index:
        temp.loc[i,'DeltaDeaths']=temp.loc[i,'Deaths'] - y2.loc[i,'Deaths']
        temp.loc[i,'DeltaConfirmed']=temp.loc[i,'Confirmed'] - y2.loc[i,'Confirmed']
        temp.loc[i,'DeltaRecovered']=temp.loc[i,'Recovered'] - y2.loc[i,'Recovered']
        if (temp.loc[i,'Deaths']!=0):
            temp.loc[i,'DeathsDeltaPercent'] =round(temp.loc[i,'DeltaDeaths']/temp.loc[i,'Deaths']*100,1)
        if (temp.loc[i,'Confirmed']!=0):
            temp.loc[i,'MortalityPercent'] =round(temp.loc[i,'Deaths']/temp.loc[i,'Confirmed']*100,1)
            temp.loc[i,'RecoveryPercent'] =round(temp.loc[i,'Recovered']/temp.loc[i,'Confirmed']*100,1)
            
        
dfMain = temp.copy()    
#print('Done')  


#Countrywise consolidated figures
#dfMain[dfMain['Date']==dfMain['Date'].max()].groupby('Country').sum().sort_values('Deaths',ascending=False)[['Deaths','Confirmed','Recovered']]

#Separate dataframes for China and others(excluding China)
dfNChina=dfMain[dfMain['Country']!='China']#Excluding China
dfChina = dfMain[dfMain['Country']=='China']#China only


# In[ ]:


#Put the various styles used here:

styles = [
    dict(selector="caption",props=[('font-size','10pt'),('text-align','left'),('color','#0000ff'),('font-weight','bold')]),
    dict(selector="th",props=[('color','teal'),('border','1px solid #000000')]),
    dict(selector="td",props=[('font-size','10pt'),('border','1px solid #000000')])
]

#Style for small tables
styles2 = [
    dict(selector="caption",props=[('font-size','9pt'),('text-align','left'),('color','#0000ff'),('font-weight','bold')]),
    dict(selector="th",props=[('color','teal'),('border','1px solid #000000'),('width','100px')]),
    dict(selector="td",props=[('font-size','10pt'),('border','1px solid #000000')]),
    dict(selector="col",props=[('width','100px')]),
#   dict(selector="table",props=[('align','center'),('margin-left','auto'),('margin-right','auto')])
]


# In[ ]:


strDate = dfMain.Date.max().strftime("%d-%b-%Y")#Get the last date in dataset
strToday = pd.datetime.now().strftime("%d-%b-%Y %H:%M")
display(HTML("<h5> Updated at " + strToday + " UTC </h5>"))


# In[ ]:


t1 = pd.DataFrame(dfMain[dfMain.Date==dfMain.Date.max()].sum()[['Deaths','Confirmed','Recovered']]).transpose()
t1.Confirmed = t1.Confirmed.astype(int)
t1.style.background_gradient(cmap='RdPu').hide_index().set_caption("<a id='BasicNumbers'></a>Overall Numbers as on :" + strDate).set_table_styles(styles2)


# [Back to Top](#top)

# In[ ]:


dfPlot = dfMain[dfMain.Date==dfMain.Date.max()].groupby('Country').sum().reset_index()
countries = dfPlot.Country.drop_duplicates()

data = dict(type='choropleth', locationmode ='country names',
           locations=countries,
            #np.array(dfPlot.Country.drop_duplicates()),
           z=np.arange(0,len(countries)),
           autocolorscale = True,
           #colorscale ='',
           text ="Country: " + dfPlot.Country+'<br> Confirmed: ' + dfPlot.Confirmed.astype(str),
           colorbar ={'title':'Country Colors','len':200,'lenmode':'pixels'})
layout=dict(geo ={'scope':'world'})


col_map = go.Figure(dict(data=data,layout=layout))
col_map.update_layout(title={
    'text':"Global Spread of Covid19",
    'x':0.5,'y':0.85
})
iplot(col_map)


# In[ ]:


dfTop20 = dfMain[dfMain.Date==dfMain.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending=False).head(20).reset_index()
country = dfTop20.Country
if ("India" in country) ==False :
    country=np.append(country,'India')
dfPlot = dfMain[(dfMain.Country.isin(country))&(dfMain.Date==dfMain.Date.max())].groupby('Country').sum().sort_values('Deaths', ascending =False)
dfPlot = dfPlot.reset_index()

data = dict(type='choropleth', locationmode ='country names',
           locations=country,
            #np.array(dfPlot.Country.drop_duplicates()),
           z=dfPlot.Deaths.astype(int),
           autocolorscale = False,
           colorscale ='Reds',
           text ="Country: " + dfPlot.Country+'<br> Confirmed: ' + dfPlot.Confirmed.astype(str),
           colorbar ={'title':'Country Colors','len':200,'lenmode':'pixels'})
layout=dict(geo ={'scope':'world'})


col_map = go.Figure(dict(data=data,layout=layout))
col_map.update_layout(title={
    'text':"Reported Deaths: Top 20 countries and India",
    'x':0.5,'y':0.85
})
iplot(col_map)


# In[ ]:


#Top 20 countries in terms of loss of lives
dfMain[dfMain['Date']==dfMain['Date'].max()].groupby('Country').sum().sort_values('Deaths',ascending=False)[['Deaths','Confirmed','Recovered','MortalityPercent']].head(20).style.set_caption("<a id='TableTopTwenty'></a>20 Countries with maximum deaths as on:" + strDate).set_table_styles(styles).background_gradient(cmap="Reds").set_properties(**{'font-size': '9pt', 'font-family': 'Calibri'})


# [Back to Top](#top)

# In[ ]:


dfTop20 = dfMain[dfMain.Date==dfMain.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending=False).head(20).reset_index()
country = dfTop20.Country
if ("India" in country) ==False :
    country=np.append(country,'India')
#for i in range(0,len(country)):
fig =go.Figure()
for i in range(0,len(country)):
    dfPlot = dfMain[(dfMain.Country == country[i])&(dfMain.Deaths<5000)&(dfMain.Confirmed <50000)].groupby(['Country','Date']).sum().reset_index()
# plot_data = go.Scatter(x=dfTop20.Confirmed,y=dfTop20.Deaths,mode = 'lines+markers')
# go.Figure(data=plot_data,layout=dict(title="test"))
    fig.add_trace(go.Scatter(x=dfPlot.Confirmed,y=dfPlot.Deaths,text=country[i],mode='markers+lines', name=country[i]))
    fig.update_layout(title=dict(text ="Comparision of Confirmed cases vs deaths",xref="paper",font=dict(size=12)),
                      xaxis_title="Confirmed Cases upto 50000",yaxis_title='No of Deaths upto 5000')
fig.show()


# In[ ]:


plt.plot('Deaths',data=dfMain.groupby(['Date']).sum().reset_index(),label='Deaths');
plt.plot('Confirmed',data=dfMain.groupby(['Date']).sum().reset_index(),label='Confirmed Cases');
plt.plot('Recovered',data=dfMain.groupby(['Date']).sum().reset_index(),label='Recovered Cases');
plt.legend();
plt.title("Global change of deaths, confirmed and recovered cases");
plt.xlabel("Days since 22-Jan-2020");
plt.ylabel("No. of Cases");
plt.grid(True)
display(HTML("<a id='GlobalGraph'></a>"));
display(HTML("<a href=#top>Back to Top</a>"))


# [Back to Top](#top)

# In[ ]:


colors = sns.color_palette('colorblind')
plt.figure(figsize=(15,9))
LatestByCountry =dfMain[dfMain['Date']==dfMain.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending = False).reset_index()
temp = LatestByCountry.sort_values('Deaths',ascending=False).head(20)
sns.set_style('whitegrid')
ax=sns.barplot(x='Deaths',y='Country',data=temp);
for p in ax.patches:
    ax.text(p.get_x()+p.get_width(),p.get_y()+0.6,int(p.get_width()),fontsize=12,color='black')
    #print(p.get_width())
plt.title("Top 20 Countries in loss of lives updated on: " + strDate,{'color':'blue'});
plt.xticks(ticks= np.arange(0,temp.Deaths.max(),500),fontsize=8);
#ax=plt.gca()

#plt.gca().invert_yaxis()
plt.show()
display(HTML('<a id="barTop20"></a>'))


# [Back to Top](#top)

# In[ ]:


pd.plotting.register_matplotlib_converters()
countries = dfMain[dfMain.Date == dfMain.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending=False).head(5).reset_index() #top 5 countries
countries_arr = np.array(countries.Country)#Get the list of countries as array
dfSelectedCountries = pd.DataFrame(np.append(countries_arr,'India'))#Add India and convert to Dataframe
dfSelectedCountries.rename(columns={0:'Country'},inplace = True)#rename column
dfMerged = pd.merge(dfSelectedCountries,dfMain,on=('Country'), how= 'left')#merge to get other data from dfMain
plt.figure(figsize = (15,9))
varCutOffDate = (dfMerged.Date.max() -pd.to_timedelta(30,'d'))
sns.lineplot(x ='Date', y='Confirmed',
             data=dfMerged[dfMerged.Date >= varCutOffDate],
             hue ='Country',marker=True,legend='brief');
plt.title('Confirmed Cases last one month',fontsize=16);
plt.figure(figsize = (15,9))
sns.lineplot(x ='Date', y='Deaths',
             data=dfMerged[dfMerged.Date >= varCutOffDate],
             hue ='Country',marker=True,legend='brief');
plt.title('Deaths last one month',fontsize=16);


# No of Deaths in China - All provinces

# In[ ]:


LatestChina = dfChina[dfChina['Date']==dfChina.Date.max()]
plt.figure(figsize=(9,12))
ax=LatestChina.groupby('State')['Deaths'].sum().plot(kind='barh')
a = ax.set_title('All China', fontsize=20)


# In[ ]:


dfNHubei = LatestChina[LatestChina['State']!='Hubei']#Excluding Hubei which has very large number of deaths
plt.figure(figsize=(12,9))
dfNHubei.groupby('State').sum()['Deaths'].plot(kind='barh')
a =plt.title("No of deaths in China excluding Hubei")


# In[ ]:


plt.figure(figsize=(15,10));
ax=dfNChina.groupby("Date").sum()['Deaths'].plot(marker='x',markersize=5,label="World excluding China");
ax=dfChina.groupby("Date").sum()['Deaths'].plot(marker="o",markersize=5,label="China")
a = ax.set_title("No of Deaths China and World",fontsize=25)
plt.legend();


# In[ ]:


plt.figure(figsize=(20,10))
plt.subplots_adjust(hspace = 1.0)
plt.subplot(3,2,1)
df = dfMain[dfMain['Country']=='Spain']
x_val = np.linspace(0,len(df),len(df),True)
plt.plot(x_val,'DeathsDeltaPercent', data= df,marker='o',linewidth=3,label="% increase in mortality")
y_mean = dfMain[dfMain['Country']=='Spain']['DeathsDeltaPercent'].mean()
plt.axhline(y_mean)
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
plt.title("Spain")
plt.subplot(3,2,2)
#plt.figure(figsize=(20,6))
plt.plot(x_val,'DeathsDeltaPercent', data= dfMain[dfMain['Country']=='Italy'],marker='o',label = "% increase in mortality",color='red')
y_mean = dfMain[dfMain['Country']=='Italy']['DeathsDeltaPercent'].mean()
tempvar = (dfMain[dfMain['Country']=='Italy']['Deaths']/dfMain[dfMain['Country']=='Italy']['Confirmed'])*100
#plt.plot('Date',dfMain[dfMain['Country']=='Italy']['Deaths']/dfMain[dfMain['Country']=='Italy']['Confirmed']),data= dfMain[dfMain['Country']=='Italy'])

plt.axhline(y_mean,color='red')
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
plt.title("Italy")
plt.subplot(3,2,3)
#plt.figure(figsize=(20,6))
plt.plot(x_val,'DeathsDeltaPercent', data= dfMain[dfMain['Country']=='Germany'],marker='o',label="% increase in mortality",color='green')
y_mean = dfMain[dfMain['Country']=='Germany']['DeathsDeltaPercent'].mean()

plt.axhline(y_mean,color='black',label="mean")
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
plt.title("Germany")
plt.subplot(3,2,4)
plt.plot(x_val,'DeathsDeltaPercent', data= dfMain[dfMain['Country']=='Iran'],marker='o',label="% increase in mortality",color='brown')

y_mean = dfMain[dfMain['Country']=='Iran']['DeathsDeltaPercent'].mean()

plt.axhline(y_mean,color='teal')
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
plt.title("Iran")

plt.subplot(3,2,5)
plt.plot(x_val,'DeathsDeltaPercent', data= dfMain[dfMain['Country']=='India'],marker='o',label="% increase in mortality",
         color='purple')

y_mean = dfMain[dfMain['Country']=='India']['DeathsDeltaPercent'].mean()

plt.axhline(y_mean,color='purple')
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
plt.title("India")

plt.subplot(3,2,6)

#dfGrouped = dfMain[dfMain.Country == 'USA']
plt.plot(x_val,'DeathsDeltaPercent', data= dfMain[dfMain['Country']=='US'].groupby('Date').sum(),
         marker='o',label="% increase in mortality",color='orange')

y_mean = dfMain[dfMain['Country']=='US']['DeathsDeltaPercent'].mean()

plt.axhline(y_mean,color='teal')
plt.legend()
plt.xlabel("No of days from 22nd Jan 2020")
a=plt.title("US combined")
plt.suptitle("Daily change in percent mortality", fontsize = 20,color='#068b81');


# plt.plot('Date','DeathsDeltaPercent', data= dfMain[dfMain['Country']=='United Kingdom'],marker='o')
#plt.plot('Date','DeathsDeltaPercent', data= dfMain[dfMain['Country']=='US'],marker='o')
#plt.plot('Date','DeathsDeltaPercent', data= dfMain[dfMain['Country']=='France'],marker='o')


# In[ ]:


from scipy.optimize import curve_fit
def linfit(x,a,b):
    return a*x+b

def deg2fit(x,a,b,c):
    return (a*x**2)+(b*x)+c

def deg3fit (x,a,b,c,d):
    return(a*x**3)+(b*x**2) +(c*x)+ d

def expfit(x,a,b):
    return np.exp(a*x)+b


# #### The graphs below plot the reported cases and projections based on linear, polynomial and exponential curve fit.The curves are drawn using the acutal data. Polynomial curves are two types - deg2 which is of the form of at^2+bt+c and deg3 which is of the form at^3+bt^2+ct+d

# In[ ]:


display(HTML("<a id='ConfirmedProjections'></a>"))
dfcountries = dfNChina[dfNChina.Date == dfNChina.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending=False).head(9)
countries = np.array(dfcountries.index)
if not('India'in countries):
    countries = np.append(countries,'India')
plt.figure(figsize = (10,100));
plt.subplots_adjust(hspace = 1.0)
for i in range(1,11):
    ax = plt.subplot(10,1,i)
    
    #myfig.set_facecolor('#ffffff')
    df = dfMain[(dfMain.Country == countries[i-1])& (dfMain.Confirmed >100)].groupby(['Country','Date']).sum().reset_index()
    #plt.xticks(ticks=np.arange(0,len(df),1))
    ax.plot(np.arange(0,len(df)),df.Confirmed,marker='o', label = "Actual",color='black')
    no_of_days = len(df)
    
    x= np.arange(0,len(df))
    y = df.Confirmed
    varDate = df.Date.min().strftime("%d-%m-%Y")
    
    #linear fit coeff:
    popt1,pcov1 = curve_fit(linfit,x,y,[1,1])
    popt2,pcov2 =curve_fit(deg2fit,x,y,[1,1,1])
    popt3,pcov3 =curve_fit(deg3fit,x,y,[1,1,1,1])
    popte,pcove = curve_fit(expfit,x,y)
    poly_coeff = np.polyfit(x,y,2)
    popt7,pcov7 = curve_fit(linfit,x[-7:],y[-7:],[1,1])#7 days
    
#     y2 = np.poly1d(poly_coeff)
    #ax.plot(x,y2(x),label="Projected",linestyle= ':',color='red')

    #projected for 15 days since 100 confirmations
    x= np.arange(0,len(df)+15,1)
    #for exponential take only 5 days:
    xe =np.arange(0,len(df)+5,1)
    #for 7 days  moving consider from the last point for 7 days
    x7 = np.arange(len(df)-6,len(df)+15,1)
    y1 = linfit(x,*popt1) # projected y for x
    y2 = deg2fit(x,*popt2)
    y3 =deg3fit(x,*popt3)
    ye = expfit(xe,*popte)
    y7 = linfit(x7,*popt7)
    
    lim = [y1.max(),y2.max(),y3.max(),ye.max()]# get the limits of the graph
#   
    ax.plot(x,y1,linestyle ='-.',label='linear')
    ax.plot(x,y2,linestyle = '--',label = 'deg2')
    ax.plot(x,y3,label ='deg3')
    ax.plot(xe,ye,label ='exp')
    ax.plot(x7,y7,linestyle="-",label='7days')
    
    lim = plt.axis()# get the limits of the graph
    d7 = no_of_days+7 #get the day 7 days from the last update
    date7 = df.Date.max()+pd.DateOffset(days = 7)
    strDate7 = date7.strftime("%d-%m-%Y")
    lastUpdate = df.Date.max().strftime("%d-%m-%Y")

    
    #plot a vertical line 7 days from the date of last update
    ax.axvline(x=d7,color='red',linestyle='dotted')
    
    #plot a marker at intersection of the 7 day line
    y7lin = linfit(d7,*popt1)
    y7deg2 = deg2fit(d7,*popt2)
    y7deg3 = deg3fit(d7,*popt3)
    y7exp = expfit(d7,*popte)
    y7mov = linfit(d7,*popt7)
    ax.plot(d7,y7lin,marker='o',color='red')
    ax.plot(d7,y7deg2,marker='o',color='red')
    ax.plot(d7,y7deg3,marker='o',color='red')
    ax.plot(d7,y7mov,marker = 'o', color= 'red')
    
       
    ax.set_title(countries[i-1]+ ": Reported and Projected cases since first 100", fontsize =18)
    
    strPlotComments = 'Last updated on: '+ lastUpdate
    strPlotComments += '\n'+ 'Projected 7 days \n  linear: '+ str(int(y7lin))+'\n  deg2: '+ str(int(y7deg2))+           "\n  deg3: " + str(int(y7deg3))+ "\n  7 days linear:  " + str(int(y7mov))
    ax.text(0,lim[3]*0.6, strPlotComments,fontsize = 10)
    #ax.text(0,lim[3]*0.8,'First 100 Cases:' + varDate, fontsize = 10)
    
    
#     ax.text(no_of_days+8,plt.axis()[3]*0.50,str(round(y7lin,0)))
    #ax.text(0,y.max()*.55, "y = " + str(round(poly_coeff[0],2))+"*t^2 + " + str(round(poly_coeff[1],2))+"*t + " +  str(round(poly_coeff[2])))
    bb_box = dict(boxstyle='square',fc='w',alpha=1.0)
    
    myArrowProp1 = dict(arrowstyle='->',color='black') #define arrow style and colour = black
    myArrowProp2 = dict(arrowstyle='->',color='blue')#define arrow style and colour = blue
    myArrowProp3 = dict(arrowstyle='->',color='red')#define arrow style and colour = blue
    ax.annotate(str(df.loc[len(df)-1,'Confirmed']),  
                xy=(len(df)*.98,df.loc[len(df)-1,'Confirmed']), 
                xytext =(len(df)-10,df.loc[len(df)-1,'Confirmed']*1.5),  
                arrowprops= myArrowProp1)
    
    #check if exponential curve intersects within the limit of y
    if (int(y7exp) > int(lim[3])):
        projected_values = np.array([int(y7lin),int(y7deg2),int(y7deg3)])
    else:
        projected_values = [int(y7lin),int(y7deg2),int(y7deg3),int(y7exp)]
    projected_max = projected_values.max()
    projected_min = projected_values.min()
    
    ax.annotate(strDate7,xy=(d7,0),xytext = (d7-8,lim[3]*0.02),arrowprops = myArrowProp3)
    ax.annotate(projected_min,xy=(d7,projected_min),xytext=(d7-7,projected_min*0.5),arrowprops= myArrowProp1)
    ax.annotate(projected_max,xy=(d7,projected_max),xytext=(d7+4,projected_max*1.5),arrowprops= myArrowProp2)

    
    #txtbox = 
    ax.legend(loc='upper center', ncol = 5)
    ax.grid(True)
    ax.set_xlabel("No of Days")
    ax.set_ylabel("Cases")
    

# plt.title("Actual and Projected Confirmed Cases", fontsize = 20, color='#068b81');


# In[ ]:


display(HTML("<a href=#top>Top</a>"))


# #### The figures below show the reported deaths over time and projections based on a linear, polynomial and exponential growth. Polynomial growth rates are of second deg (marked as deg2) and third degree (marked as deg3) while the linear and exponential growths are marked as 'linear' and 'exp' respectively.

# In[ ]:


display(HTML("<a id='DeathsProjections'></a>"))
dfcountries = dfNChina[dfNChina.Date == dfNChina.Date.max()].groupby('Country').sum().sort_values('Deaths',ascending=False).head(9)
countries = np.array(dfcountries.index)
if not('India'in countries):
    countries = np.append(countries,'India')
fig = plt.figure(figsize=(10,100));
#plt.subplots_adjust(hspace = 1.0)
for i in range(1,11):
    ax = plt.subplot(10,1,i)
    df = dfMain[(dfMain.Country == countries[i-1])& (dfMain.Deaths >=1)].groupby(['Country','Date']).sum().reset_index()
    #plt.xticks(ticks=np.arange(0,len(df),1))
    ax.plot(np.arange(0,len(df)),df.Deaths,marker='o', label = "Actual",color='k')
    no_of_days = len(df)
    x= np.arange(0,len(df))
    y = df.Deaths
    
    

    varDate = df.Date.min().strftime("%d-%m-%Y")
    
    
    #linear fit coeff:
    popt1,pcov1 = curve_fit(linfit,x,y,[1,1])
    popt2,pcov2 =curve_fit(deg2fit,x,y,[1,1,1])
    popt3,pcov3 =curve_fit(deg3fit,x,y,[1,1,1,1])
    popte,pcove = curve_fit(expfit,x,y)
    popt7,pcov7 = curve_fit(linfit,x[-7:],y[-7:],[1,1])#7 days
    #print(popt2[0])
#     poly_coeff = np.polyfit(x,y,2)
#     y2 = np.poly1d(poly_coeff)
    #ax.plot(x,y2(x),label="Projected",linestyle= ':',color='red')

    #projected for 15 days since 100 confirmations
    x= np.arange(0,len(df)+15,1)
    #for exponential take only5 days:
    xe =np.arange(0,len(df)+5,1)
    
    x7 = np.arange(len(df)-6,len(df)+15,1)

    y1 = linfit(x,*popt1) # projected y for x
    y2 = deg2fit(x,*popt2)
    y3 =deg3fit(x,*popt3)
    ye = expfit(xe,*popte)
    y7 = linfit(x7,*popt7)
    

    lim = [y1.max(),y2.max(),y3.max(),ye.max()]# get the limits of the graph
    #print(lim)
    d7 = no_of_days+7 #get the day 7 days from the last update
    date7 = df.Date.max()+pd.DateOffset(days = 7)
    strDate7 = date7.strftime("%d-%m-%Y")
    lastUpdate = df.Date.max().strftime("%d-%m-%Y")

    
    #plot a vertical line 7 days from the date of last update
    ax.axvline(x=d7,color='red',linestyle='dotted')
    
    #plot a parallel line from last data point to d7 vertical line
    #ax.axhline(y=df.Deaths.max())
    
    #plot a marker at intersection of the 7 day line
    y7lin = linfit(d7,*popt1)
    y7deg2 = deg2fit(d7,*popt2)
    y7deg3 = deg3fit(d7,*popt3)
    y7exp = expfit(d7,*popte)
    y7mov = linfit(d7,*popt7)
    
    #check if exponential curve intersects within the limit of y
    if (int(y7exp) > int(lim[3])):
        projected_values = np.array([int(y7lin),int(y7mov),int(y7deg2),int(y7deg3)])
    else:
        projected_values = ndarray.sort([int(y7lin),int(y7mov),int(y7deg2),int(y7deg3),int(y7exp)])
    projected_max = projected_values.max()
    projected_min = projected_values.min()
    if (projected_min <=y.max()):
        projected_min=projected_values[projected_values>y.max()][0]
        print(countries[i-1],y.max(),projected_min)
    
    ax.plot(d7,projected_max,marker='o',color='red')
    ax.plot(d7,projected_min,marker='o',color='red')
    #ax.plot(d7,y7deg3,marker='o',color='red')
#   
    ax.plot(x,y1,linestyle ='-.',label='linear')
    ax.plot(x,y2,linestyle = '--',label = 'deg2')
    ax.plot(x,y3,label ='deg3')
    ax.plot(xe,ye,label ='exp')
    ax.plot(x7,y7,linestyle="-",label='7days')

    y = (poly_coeff[0]*x**2) +(poly_coeff[1]*x)+poly_coeff[2]
#     ax.plot(x,y,linestyle= ':',label ="Projected_deg2",color='purple')
    ax.set_title(countries[i-1]+ ": Deaths-Reported and Trends")
    strPlotComments= 'Last updated on: '+ lastUpdate
    strPlotComments += '\n'+ 'Projected 7 days: '+ str(int(projected_min))+' to '+ str(int(projected_max))
    ax.text(0,lim[3]*0.9,strPlotComments,fontsize = 10)
    

    bb_box = dict(boxstyle='square',fc='w',alpha=1.0)
    
    myArrowProp1 = dict(arrowstyle='->',color='black') #define arrow style and colour = black
    myArrowProp2 = dict(arrowstyle='->',color='blue')#define arrow style and colour = blue
    myArrowProp3 = dict(arrowstyle='->',color='red')#define arrow style and colour = blue    
    
    ax.annotate(str(df.loc[len(df)-1,'Deaths']),  
                xy=(len(df)*.98,df.loc[len(df)-1,'Deaths']), 
                xytext =(len(df)-10,df.loc[len(df)-1,'Deaths']*1.5),  
                arrowprops=dict(arrowstyle='->',color='b'))
    
    
    ax.annotate(strDate7,xy=(d7,0),xytext = (d7-8,lim[3]*0.05),arrowprops = myArrowProp3)
    ax.annotate(projected_min,xy=(d7,projected_min),xytext=(d7+4,projected_min*0.75),arrowprops= myArrowProp1)
    ax.annotate(projected_max,xy=(d7,projected_max),xytext=(d7+4,projected_max*1.5),arrowprops= myArrowProp2)

    
    ax.legend(loc='upper center',ncol=5)
    plt.xticks(ticks=np.arange(0,len(x),2))
    ax.grid(True)
    ax.ticklabel_format(useOffset = False)
#plt.title("Actual and Projected Loss of lives", fontsize=20, color='#068b81');
display(HTML("<a href=#top>Top</a>"))


# [Top](#top)

# In[ ]:




