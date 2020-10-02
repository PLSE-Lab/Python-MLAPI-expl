#!/usr/bin/env python
# coding: utf-8

# This notebook has below mentioned Dashboard features:
# 
# 1. New definition for calling D0 for any Nation > atleast 5 cases has been confirmed (The reason: observing it from the very 1st case of infection, does give true picture of spread, at some country the spread was kind of increasing day by day, but in some countries the value of infected people didn't blow up instatly, but took its time where the spread was unstoppable over the next 2 weeks, so idea is to start with when the spread started growing and through that not infected countries can take appropriate decisions)
# 2. Dashboard which compares any 2 combination of Nation's COVID infections confirmed cases in Log values.
# 3. Overall clustering of Nations at current states (X: Slope of the Infected (Non cumulated) values, Y: Standard Deviation of the data Infected (Non cumulated) values)
# 4. Same analysis over a period of time, same calculations on cumulated steps of values and cluster
# 5. Looking India's number makes a alarming suggestion, if all the things are as it shuold be then we are in controlled state,
#     but if it errupts or we get leakage in our behaviour, we might have severe infections in coming days (If you are looking at this notebook, try to play with the 1st dashboard 
#     and let me know your thoughts.)
# 6. I will add my observations on this slide in some time, But I think it's a good dashboard to play and observer your country and take action
# 
# Feel free to suggest me anything which I have done wrong or any thing by which I can improve my analysis

# # Part 1

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_decomposition,model_selection,metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')
import dateutil
from tqdm import tqdm
from sklearn import linear_model
import datetime
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels import tsa
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets
# Any results you write to the current directory are saved as output.


# In[ ]:


confirmedGlobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',encoding='utf-8',na_values=None)
deathGLobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoverGlobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
confirmedGlobal['NewContryCode']=confirmedGlobal['Country/Region']+confirmedGlobal['Province/State'].fillna('')
deathGLobal['NewContryCode']=deathGLobal['Country/Region']+deathGLobal['Province/State'].fillna('')
recoverGlobal['NewContryCode']=recoverGlobal['Country/Region']+recoverGlobal['Province/State'].fillna('')


# In[ ]:


confirmedGlobal.head()


# In[ ]:


#Considering day0 when atleast 5 cases has been declared
def getStartPoint(country):
    for num,i in enumerate(confirmedGlobal[confirmedGlobal['NewContryCode']==country].values[0][4:]):
        if i >= 5:
#             print (num,i)
            break
    return 4+num

def getSlope(dataSet):
    df=pd.DataFrame(dataSet)
    df=df.reset_index()
    df.columns=['x','y']
    model=linear_model.LinearRegression(fit_intercept=False).fit(df[['x']],df[['y']])
    return model.coef_[0][0]

'''This code returns:
1. The original data based on the new D0 day.
2. absolute values of any country from given cumulated data, 
3. scaled absolute values  with MinMax method, 
4. standard deviation of the absolute values so that we can compare between countries, 
5. the function also provides the slope of the absolute data to find the rate of change over the given period of time, slope I am calculating using Linear regresion by not
considering the intercept which will give me y=mx+c and we needed m from this equation, some data might not have a linear relation but at the end of the day this
value would point the direction and a magintude of increment over a period of time.
6. Logarithimic values of the cumulated values''' 
def getDataArranged(country):
    temp={}
    startPoint=getStartPoint(country)
    temp['logValues']=[np.log(i) if i !=0  else 0 for i in confirmedGlobal[confirmedGlobal['NewContryCode']==country].values[0][startPoint:-1]]
    temp['actualValues']=[i for i in confirmedGlobal[confirmedGlobal['NewContryCode']==country].values[0][startPoint:-1]]
    countryData=confirmedGlobal[confirmedGlobal['NewContryCode']==country].values[0][startPoint:-1]
    datesOfInfections=[str(dateutil.parser.parse(i))[:10] for i in confirmedGlobal[confirmedGlobal['NewContryCode']==country].columns[startPoint:-1]]
    dayIndex=['D_'+str(i).zfill(3) for i in range(len(countryData))]
    temp['countryData']=countryData
    pp=list(countryData[:-1])
    pp.insert(0,0)
    absInfected=countryData-pp
    temp['absInfected']=absInfected
    lbl=preprocessing.MinMaxScaler()
    temp['minMaxScaledval']=np.ravel(lbl.fit_transform(absInfected.reshape(len(absInfected),1)))
    temp['stdVal']=np.std(temp['absInfected'])
    temp['slopeData']=getSlope(absInfected)
    temp['datesOfInfections']=datesOfInfections
    temp['dayIndex']=dayIndex
    temp['lastValue']=countryData[-1]
    return temp


# In[ ]:


countryDataDict={}
for con in tqdm(confirmedGlobal['NewContryCode']):
    try:
        countryDataDict[con]=getDataArranged(con)    
    except:
        pass


# In[ ]:


dfList=[]
for i in countryDataDict:
    tmp=pd.DataFrame({'logVals':countryDataDict[i]['logValues'],'infected':countryDataDict[i]['countryData']}).reset_index()
    tmp.columns=['dayInfo','logVals','infected']
    tmp['Country']=i
    dfList.append(tmp)


# In[ ]:


dfList=[]
for i in countryDataDict:
    tmp=pd.DataFrame(countryDataDict[i]['logValues'])
    tmp.columns=[i]
    dfList.append(tmp)
    
dfLinePlotD0=pd.concat(dfList,axis=1)
dfLinePlotD0[list(dfLinePlotD0.columns.values[np.where(dfLinePlotD0.max() >9)]) +['India']].plot(figsize=(20,10))
filteredDFForLinePlot=dfLinePlotD0#[list(dfLinePlotD0.columns.values[np.where(dfLinePlotD0.max() >9)]) +['India','Japan']]
dfList=[]
for i in countryDataDict:
    tmp=pd.DataFrame(data={'logValues':countryDataDict[i]['logValues'],'countryData':countryDataDict[i]['countryData']})
    tmp['CountryName']=i
    dfList.append(tmp)
    
allDaatForLineG=pd.concat(dfList)
allDaatForLineG=allDaatForLineG.reset_index()
allDaatForLineG.columns=['DayIndex', 'logValues', 'countryData', 'CountryName']


# In[ ]:


filtCountry=list(dfLinePlotD0.columns.values[np.where(dfLinePlotD0.max() >9)]) +['India','Japan']


# In[ ]:


textbox = widgets.Dropdown(
    description='Country:   ',
    value='India',
    options=filtCountry
)

textbox2 = widgets.Dropdown(
    description='Country2:   ',
    value='Japan',
    options=filtCountry
)

container = widgets.HBox(children=[textbox,textbox2])
# container = widgets.HBox(children=[daySLider])
# from plotly.subplots import make_subplots
tempDFFOrSlider=allDaatForLineG[allDaatForLineG['CountryName']=='India']
trace1 = go.Scatter(
    x=tempDFFOrSlider['DayIndex'],
    y=tempDFFOrSlider['logValues'],
#     mode='markers',
     mode="markers+text+lines",
#     marker=dict(size=list(tempDFFOrSlider['countryData'].values),sizemode='area',
#         sizeref=2.*max(list(tempDFFOrSlider['countryData'].values))/(40.**2),
# #                 color=tempDFFOrSlider['clusterInfo'],
#         sizemin=4),
    showlegend=False,
    fill='tozeroy',
    text=tempDFFOrSlider['countryData'])

tempDFFOrSlider2=allDaatForLineG[allDaatForLineG['CountryName']=='Germany']
trace2 = go.Scatter(
    x=tempDFFOrSlider2['DayIndex'],
    y=tempDFFOrSlider2['logValues'],
#     mode='markers',
     mode="markers+text+lines",
#     marker=dict(size=list(tempDFFOrSlider2['countryData'].values),sizemode='area',
#         sizeref=2.*max(list(tempDFFOrSlider2['countryData'].values))/(40.**2),
# #                 color=tempDFFOrSlider['clusterInfo'],
#         sizemin=4),
    showlegend=False,
    fill='tozeroy',
    text=tempDFFOrSlider2['countryData'])

g = go.FigureWidget(data=[trace1,trace2],layout=go.Layout(title=dict(text='Covid19'),barmode='overlay'))

def response(change):
    vaL=textbox.value
    vaL2=textbox2.value
    tempDFFOrSlider=allDaatForLineG[allDaatForLineG['CountryName']==vaL]
    tempDFFOrSlider2=allDaatForLineG[allDaatForLineG['CountryName']==vaL2]
    x1 = tempDFFOrSlider['DayIndex'].values
    x2 = tempDFFOrSlider['logValues'].values
    x3 = tempDFFOrSlider2['DayIndex'].values
    x4 = tempDFFOrSlider2['logValues'].values

    with g.batch_update():
        g.data[0].x = x1
        g.data[0].y = x2
#         g.data[0].marker['size']= list(tempDFFOrSlider['countryData'].values)
#         g.data[0].marker['color']= tempDFFOrSlider['clusterInfo'].values
        g.data[0].text = tempDFFOrSlider['countryData']
    
        g.data[1].x = x3
        g.data[1].y = x4
#         g.data[1].marker['size']= list(tempDFFOrSlider2['countryData'].values)
#         g.data[0].marker['color']= tempDFFOrSlider['clusterInfo'].values
        g.data[1].text = tempDFFOrSlider2['countryData']
    
        g.layout.barmode = 'overlay'
# daySLider.observe(response, names="value")
textbox.observe(response, names="value")
textbox2.observe(response, names="value")
widgets.VBox([container,g])


# # Part 2

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

''' Clustering the data using the scaled values of stdDev and Slope values calculated earlier'''

def kmeanCluster(df):
    tempClusterOutput=[]
    metricOutput=[]
    for j in range(3,8):
        kmeans_model = KMeans(n_clusters=j, random_state=1)#.fit(overallSpacingDF[['varianceVal','scaledSlope']])
        tempClusterOutput.append(kmeans_model.fit_predict(df[['stdVal','scaledSlope']]))
        metricOutput.append(metrics.silhouette_score(df[['stdVal','scaledSlope']], kmeans_model.labels_, metric='euclidean'))
    
    return tempClusterOutput[np.argmax(np.array(metricOutput))]


# In[ ]:


countryOverallSpcaing=[]
for i in countryDataDict:
    temp={}
    temp['stdVal']=countryDataDict[i]['stdVal']
    temp['slopeData']=countryDataDict[i]['slopeData']
    temp['lastValue']=countryDataDict[i]['lastValue']
    temp['countryName']=i
    countryOverallSpcaing.append(temp)


# In[ ]:


overallSpacingDF=pd.DataFrame(countryOverallSpcaing)
overallSpacingDF['scaledSlope']=preprocessing.MinMaxScaler().fit_transform(overallSpacingDF[['slopeData']])
overallSpacingDF['stdVal']=preprocessing.MinMaxScaler().fit_transform(overallSpacingDF[['stdVal']])
overallSpacingDF.head()
overallSpacingDF['clusterInfo']=kmeanCluster(overallSpacingDF)


# In[ ]:


import plotly.express as px
fig = px.scatter(overallSpacingDF, x="scaledSlope", y="stdVal",
	         size="lastValue", color="clusterInfo",
                 hover_name="countryName", log_x=True, size_max=60,text='countryName')
fig.update_layout(showlegend=False)
fig.update_layout(
    title="ALl Countries Cluster Variance vs Slope",
  
)
fig.show()


# ## Adding filters > 800  and Removing China

# In[ ]:


filoverallSpacingDF=overallSpacingDF[overallSpacingDF['lastValue']>800]
filoverallSpacingDF=filoverallSpacingDF[~filoverallSpacingDF['countryName'].str.contains('ina')]
filoverallSpacingDF.shape
import plotly.express as px
fig = px.scatter(filoverallSpacingDF, x="scaledSlope", y="stdVal",
	         size="lastValue", color="clusterInfo",
                 hover_name="countryName", log_x=True, size_max=60,text='countryName')
fig.update_layout(showlegend=False)
fig.update_layout(
    title="Filtered List of Countries Cluster Variance vs Slope",
  
)
fig.show()


# # Creating a slider for observing the change over time for All countries behavior on Infection over the the day from which it started

# In[ ]:


'''This function calculates the Std Deviation and Slope of the actual data over the nth day of the infection not using cumulated data '''

def getCountryDayWiseDetails(country):
    indiCountyDaywise=[]
    tempX=countryDataDict[country]
    for i in range(0,len(tempX['countryData'])):
        try:
            tempDict={'stdVal':np.std(tempX['absInfected'][:i]),
                  'dayInfo':i,#'D_'+str(i).zfill(3),
                  'slopeVal':getSlope(tempX['absInfected'][:i]),
                'infected':tempX['countryData'][i]}
        except:
            tempDict={'stdVal':None,
                  'dayInfo':i,#'D_'+str(i).zfill(3),
                  'slopeVal':None,
                     'infected':tempX['countryData'][i]}

        indiCountyDaywise.append(tempDict)
    tempDWiseData=pd.DataFrame(indiCountyDaywise)
    tempDWiseData['CountryName']=country
    return tempDWiseData


# In[ ]:


countryDataDictDayWise=[]
for con in tqdm(countryDataDict):
    countryDataDictDayWise.append(getCountryDayWiseDetails(con))
    
allCOuntryDayWiseDataDF=pd.concat(countryDataDictDayWise)
allCOuntryDayWiseDataDF.head()


# In[ ]:


'''Creating cluster for each Nth day to observe how the pattern is among the countries on the respective Nth day '''
moreDataFrameForSlider=[]
for indDate in tqdm(pd.unique(allCOuntryDayWiseDataDF['dayInfo'])):
    try:
        tempDF=allCOuntryDayWiseDataDF[allCOuntryDayWiseDataDF['dayInfo']==indDate]
    #     tempDF=allCOuntryDayWiseDataDF[allCOuntryDayWiseDataDF['dayInfo']==21]
        tempDF['scaledSlope']=preprocessing.MinMaxScaler().fit_transform(tempDF[['slopeVal']])
        tempDF['scaledstdVal']=preprocessing.MinMaxScaler().fit_transform(tempDF[['stdVal']])
        tempDF['clusterInfo']=kmeanCluster(tempDF)
        moreDataFrameForSlider.append(tempDF)
    except:
        pass
    
allDFWithCluster=pd.concat(moreDataFrameForSlider)
allDFWithCluster.head()


# # Dashboard 2 cluster

# In[ ]:


daySLider = widgets.IntSlider(
    value=1,
    min=1.0,#min(allDFWithCluster['dayInfo']),
    max=max(allDFWithCluster['dayInfo']),
    step=1.0,
    description='Day:',
    continuous_update=False
)

# textbox = widgets.Dropdown(
#     description='Country:   ',
#     value='India',
#     options=allCOuntryData.columns.tolist()
# )
container = widgets.HBox(children=[daySLider])
# from plotly.subplots import make_subplots

tempDFFOrSlider=allDFWithCluster[allDFWithCluster['dayInfo']==59]
trace1 = go.Scatter(
    x=tempDFFOrSlider['slopeVal'],
    y=tempDFFOrSlider['stdVal'],
#     mode='markers',
     mode="markers+text",
    marker=dict(size=tempDFFOrSlider['infected'],sizemode='area',
        sizeref=3.*max(tempDFFOrSlider['infected'])/(40.**2),
                color=tempDFFOrSlider['clusterInfo'],
        sizemin=4),
    showlegend=False,
    text=tempDFFOrSlider['CountryName'])
g = go.FigureWidget(data=[trace1],layout=go.Layout(title=dict(text='Covid19'),barmode='overlay'))

def response(change):
    vaL=daySLider.value
    tempDFFOrSlider=allDFWithCluster[allDFWithCluster['dayInfo']==vaL]
    x1 = tempDFFOrSlider['slopeVal'].values
    x2 = tempDFFOrSlider['stdVal'].values
    with g.batch_update():
        g.data[0].x = x1
        g.data[0].y = x2
        g.data[0].marker['size']= tempDFFOrSlider['infected'].values
        g.data[0].marker['color']= tempDFFOrSlider['clusterInfo'].values
        g.layout.barmode = 'overlay'
        g.data[0].text = tempDFFOrSlider['CountryName']
daySLider.observe(response, names="value")
# textbox.observe(response, names="value")

widgets.VBox([container,g])


# ## Dashboard 2 with filter countries

# In[ ]:


filtCountry=list(allDFWithCluster.groupby(['CountryName']).agg({'infected':'max'})     .reset_index().sort_values(by='infected',ascending=False).head(20)['CountryName'].values)    + ['India','Japan']

# filallCOuntryDayWiseDataDF=allCOuntryDayWiseDataDF[]
filterNextForIndia=allDFWithCluster[allDFWithCluster['CountryName'].isin(filtCountry)]
# filterNextForIndia=allDFWithCluster[~allDFWithCluster['CountryName'].str.contains('ina')]


# In[ ]:


daySLider1 = widgets.IntSlider(
    value=59,
    min=1,#min(filterNextForIndia['dayInfo']),
    max=max(filterNextForIndia['dayInfo']),
    step=1.0,
    description='Day:',
    continuous_update=False
)

container1 = widgets.HBox(children=[daySLider1])
# from plotly.subplots import make_subplots

tempDFFOrSlider=filterNextForIndia[filterNextForIndia['dayInfo']==59]
trace11 = go.Scatter(
    x=tempDFFOrSlider['slopeVal'],
    y=tempDFFOrSlider['stdVal'],
#     mode='markers',
     mode="markers+text",
    marker=dict(size=tempDFFOrSlider['infected'],sizemode='area',
        sizeref=3.*max(tempDFFOrSlider['infected'])/(40.**2),
                color=tempDFFOrSlider['clusterInfo'],
        sizemin=4),
    showlegend=False,
    text=tempDFFOrSlider['CountryName'])
g2 = go.FigureWidget(data=[trace11],layout=go.Layout(title=dict(text='Covid19'),barmode='overlay'))

def response(change):
    vaL=daySLider1.value
    tempDFFOrSlider=filterNextForIndia[filterNextForIndia['dayInfo']==vaL]
    x1 = tempDFFOrSlider['slopeVal'].values
    x2 = tempDFFOrSlider['stdVal'].values
    with g.batch_update():
        g2.data[0].x = x1
        g2.data[0].y = x2
        g2.data[0].marker['size']= tempDFFOrSlider['infected'].values
        g2.data[0].marker['color']= tempDFFOrSlider['clusterInfo'].values
        g2.data[0].text = tempDFFOrSlider['CountryName']
        g2.layout.barmode = 'overlay'
daySLider1.observe(response, names="value")
# textbox.observe(response, names="value")

widgets.VBox([container1,g2])


# In[ ]:




