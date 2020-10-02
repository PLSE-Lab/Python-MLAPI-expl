#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.json import json_normalize
import json
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
json_conv = {col: json.loads for col in (json_cols)}
train = pd.read_csv("../input/train.csv",
                    #nrows = 10000,
                    dtype={'fullVisitorId': str},
                    converters={'device': json.loads,
                               'geoNetwork': json.loads,
                               'totals': json.loads,
                               'trafficSource': json.loads,
                              })


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


def extractJsonColumns(df):
    for col in json_cols:
        print('Working on :' + col)
        jsonCol = json_normalize(df[col].tolist())
        jsonCol.columns = [col+'_'+jcol for jcol in jsonCol.columns]
        df = df.merge(jsonCol,left_index=True,right_index=True)
        df.drop(col,axis=1,inplace=True)
    return(df)


# In[ ]:


train = extractJsonColumns(train)
train.columns


# In[ ]:


len(train)


# In[ ]:


def generateColumnInfo(df):
    cls = []
    nullCount = []
    nonNullCount = []
    nullsPct = []
    uniqCount = []
    dataType = []
    for i,col in enumerate(df.columns):
        cls.append(col)
        nullCount.append(df[col].isnull().sum())
        nonNullCount.append(len(df)-df[col].isnull().sum())
        nullsPct.append((df[col].isnull().sum())*(100)/len(df))
        uniqCount.append(df[col].nunique())
        dataType.append(df[col].dtype)
        
    column_info = pd.DataFrame(
        {'ColumnName': cls,
         'NullCount': nullCount,
         'NonNullCount': nonNullCount,
         'NullPercent': nullsPct,
         'UniqueValueCount': uniqCount,
         'DataType':dataType
        })
    return(column_info)


# In[ ]:


generateColumnInfo(train)


# ## Drop Columns

# trafficSource_campaignCode is not present in test set

# In[ ]:


train.drop('trafficSource_campaignCode',axis=1,inplace=True)


# In[ ]:


trn_colInfo = generateColumnInfo(train)
trn_colInfo[(trn_colInfo['NullCount'] == 0) & (trn_colInfo['UniqueValueCount'] == 1)]


# These columns have a single unique value. They can be dropped.

# In[ ]:


train.drop(['socialEngagementType',
'device_browserSize',
'device_browserVersion',
'device_flashVersion',
'device_language',
'device_mobileDeviceBranding',
'device_mobileDeviceInfo',
'device_mobileDeviceMarketingName',
'device_mobileDeviceModel',
'device_mobileInputSelector',
'device_operatingSystemVersion',
'device_screenColors',
'device_screenResolution',
'geoNetwork_cityId',
'geoNetwork_latitude',
'geoNetwork_longitude',
'geoNetwork_networkLocation',
'totals_visits',
'trafficSource_adwordsClickInfo.criteriaParameters'],axis=1,inplace=True)


# ## Missing Values and EDA

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


def plot_colCount(df,col,xtick=0,w=12,h=7):
    plt.figure(figsize=(w,h))
    p = sns.countplot(data =df,x=col)
    plt.xticks(rotation=xtick)
    plt.title('Count of ' + col)
    plt.show()
    
def plot_totalRevenue(df,col,xtick=0,w=12,h=7):
    groupedDf = df.groupby(col,as_index=False)['totals_transactionRevenue'].sum()
    groupedDf = groupedDf[groupedDf['totals_transactionRevenue']>0]
    plt.figure(figsize=(w,h))
    p = sns.barplot(data=groupedDf,x=col,y='totals_transactionRevenue')
    plt.xticks(rotation=xtick)
    plt.title('Total revenue by ' + col)
    plt.show()
    
def plot_revenuePerUnitCol(df,col,xtick=0,w=12,h=7):
    plt.figure(figsize=(w,h))
    plt.ylim()
    p = sns.barplot(data =df,x=col,y='totals_transactionRevenue',ci=False)
    plt.xticks(rotation=xtick)
    plt.title('Revenue per visit')
    plt.show()


# ### Target Column

# In[ ]:


print(train['totals_transactionRevenue'].isnull().sum())


# In[ ]:


train['totals_transactionRevenue'].fillna(0,inplace=True)
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].astype('int64')


# In[ ]:


plt.figure(figsize=[10,6])
sns.distplot(train['totals_transactionRevenue'])


# ### Channel Grouping

# In[ ]:


plot_colCount(train,'channelGrouping',30,10,6)
plot_totalRevenue(train,'channelGrouping',30,10,6)
plot_revenuePerUnitCol(train,'channelGrouping',30,10,6)


# - Organic search generates the most number of visits
# - Referral generates the 4th most number of visits but generates the most revenue
# - Display generates the most revenue per visit

# ### Date

# In[ ]:


train['date'] = pd.to_datetime(train['date'],format='%Y%m%d')


# In[ ]:


import math
byDate = train.groupby('date',as_index=False).agg({'visitId':'count','totals_transactionRevenue':'sum'}).rename(columns={'visitId':'visits','totals_transactionRevenue':'totalRevenue'})
byDate['totalRevenue'] = byDate['totalRevenue']/1000000
byDateFlat = byDate.melt('date',var_name ='Numbers',value_name='values')


# In[ ]:


plt.figure(figsize=(16,8))
new_labels = ['label 1', 'label 2']
sns.lineplot(data=byDateFlat,x='date',y='values',hue='Numbers')
plt.title('Visit Count and Total Revenue (in 1000000) by date')
plt.ylabel('')
plt.show()


# - There is an increase in visits during the holiday period
# - There is an increase in the revenue during the same period

# In[ ]:


train['date_year'],train['date_month'],train['date_weekday'] = train['date'].dt.year,train['date'].dt.month,train['date'].dt.weekday
train.drop('date',axis=1,inplace=True)


# In[ ]:


plot_colCount(train,'date_weekday',0,10,6)
#Monday=0, Sunday=6


# In[ ]:


plot_totalRevenue(train,'date_weekday',0,10,6)


# -  Tuesdays, Wednesdays and Thursdays generate the most visits and revenue

# In[ ]:


plot_colCount(train,'date_month',0,10,6)


# In[ ]:


plot_totalRevenue(train,'date_month',0,10,6)


# - October and November have the highest traffic
# - April, Agust and December generate the most revenue

# ### fullVisitorId

# In[ ]:


train['fullVisitorId'].value_counts().head(10)


# In[ ]:


train.groupby('fullVisitorId').sum()['totals_transactionRevenue'].sort_values(ascending=False).head(10)


# - Visitor ID 1957458976293878100 has 278 visits and generates the most revenue

# ### visitNumber

# In[ ]:


train['visitNumber'].value_counts().head()


# ### device_browser

# In[ ]:


train['device_browser'].value_counts().head(10)


# In[ ]:


plot_colCount(train,'device_browser',80)
plot_totalRevenue(train,'device_browser',30,10,6)


# In[ ]:


plot_revenuePerUnitCol(train,'device_browser',80)


# - Chrome generates a significant majority of the visits and revenue 

# ### device_deviceCategory

# In[ ]:


f = sns.FacetGrid(train,hue='device_deviceCategory',size=5,aspect=4)
#plt.xlim(0, 300)
plt.figure(figsize=(15,10))
f.map(sns.kdeplot,'totals_transactionRevenue',shade= True)
f.add_legend()


# In[ ]:


f = sns.FacetGrid(train,hue='device_deviceCategory',size=5,aspect=4)
plt.xlim(0, 500000000)
plt.figure(figsize=(15,10))
f.map(sns.kdeplot,'totals_transactionRevenue',shade= True)
f.add_legend()


# In[ ]:


plot_colCount(train,'device_deviceCategory',60)
plot_totalRevenue(train,'device_deviceCategory',30,10,6)
plot_revenuePerUnitCol(train,'device_deviceCategory',60)


# - Desktops generate the highest visits and revenue 
# - Desktops generate the most high revenue transactions
# - Desktops generate almost 8 times the revenue per visit compared to tablets and mobile
# - Tablets generate the least total revenue 
# - Tablets generate a high number of low revenue transactions

# ### device_isMobile

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(data =train,x='device_isMobile',y='totals_transactionRevenue')


# ### device_operatingSystem

# In[ ]:


plot_colCount(train,'device_operatingSystem',60)
plot_totalRevenue(train,'device_operatingSystem',30,10,6)
plot_revenuePerUnitCol(train,'device_operatingSystem',60)


# - Windows is the popular operating system but Mac generates more revenue. 
# - Chrome generates more revenue per visit but is 3rd behind Windows and Mac in popularity

# ### geoNetwork_city

# In[ ]:


topCities = train['geoNetwork_city'].value_counts().head(50).reset_index()
topCities.columns = ['city','count']
topCities = topCities[topCities.city !='not available in demo dataset']
topCities = topCities[topCities.city !='(not set)']
topCitiesTrain = train[train['geoNetwork_city'].isin(topCities['city'])]


# In[ ]:


plot_colCount(topCitiesTrain,'geoNetwork_city',60)
plot_totalRevenue(topCitiesTrain,'geoNetwork_city',60)
plot_revenuePerUnitCol(topCitiesTrain,'geoNetwork_city',60)


# ### geoNetwork_continent

# In[ ]:


plot_colCount(train,'geoNetwork_continent',0,10,6)
plot_totalRevenue(train,'geoNetwork_continent',0,10,6)
plot_revenuePerUnitCol(train,'geoNetwork_continent',0,10,6)


# ### geoNetwork_country

# In[ ]:


import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
import plotly.graph_objs as go
import cufflinks as cf
byCountry = train.groupby('geoNetwork_country',as_index=False).agg({'visitId':'count','totals_transactionRevenue':'sum'}).rename(columns={'visitId':'visits','totals_transactionRevenue':'totalRevenue'})
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


data=dict(type='choropleth',
         locations = byCountry['geoNetwork_country'],
         locationmode = 'country names',
         colorscale = 'Blues',
         reversescale=True,
         text = ['text 1','text 2','text 3'],
         z=byCountry['visits'],
         colorbar={'title':'Total visits'})

layout = dict(title='Visit count by Country')

choromap = go.Figure(data=[data])
iplot(choromap)


# In[ ]:


data=dict(type='choropleth',
         locations = byCountry['geoNetwork_country'],
         locationmode = 'country names',
         colorscale = 'Blues',
         reversescale=True,
         text = ['text 1','text 2','text 3'],
         z=byCountry['totalRevenue'],
         colorbar={'title':'Total revenue'})

layout = dict(title='Total revenue by Country')

choromap = go.Figure(data=[data])
iplot(choromap)


# In[ ]:


topCountries = train['geoNetwork_country'].value_counts().head(80).reset_index()
topCountries.columns = ['country','count']
topCountriesTrain = train[train['geoNetwork_country'].isin(topCountries['country'])]


# In[ ]:


plot_colCount(topCountriesTrain,'geoNetwork_country',80,16)
plot_totalRevenue(topCountriesTrain,'geoNetwork_country',80,16)


# In[ ]:


plot_revenuePerUnitCol(topCountriesTrain,'geoNetwork_country',80,16)


# - United States generates the highest number of visits and revenue
# - Surprisingly, Venezuela and Kenya generate the highest revenue per visit

# In[ ]:


topCountries = train['geoNetwork_country'].value_counts().head(8).index


# In[ ]:


def plotByCountry(plotCol,n_labels = 0, xtick=0,plotType = 'line',order=0):
    groupByCountry = train.groupby(['geoNetwork_country',plotCol],as_index=False).count()[['geoNetwork_country',plotCol,'visitId']]
    groupByCountry = groupByCountry[groupByCountry['geoNetwork_country'].isin(topCountries)]
    if n_labels != 0:
        topLabels = train[plotCol].value_counts().head(n_labels).index
        groupByCountry = groupByCountry[groupByCountry[plotCol].isin(topLabels)]
    groupByCountry.columns = ['geoNetwork_country', plotCol, 'visits']
    plt.figure(figsize=[14,10])
    plt.xticks(rotation=xtick)
    if plotType == 'line':
        sns.lineplot(data=groupByCountry,x=plotCol,y='visits',hue='geoNetwork_country')
    if plotType == 'bar':
        if order == 0:
            sns.barplot(data=groupByCountry,x=plotCol,y='visits',hue='geoNetwork_country')
        if order == 1:
            sns.barplot(data=groupByCountry,x='geoNetwork_country',y='visits',hue=plotCol)


# In[ ]:


plotByCountry(plotCol='date_month',n_labels=12)


# - There is a significant spike in traffic from Vietnam between October and December
# - Thailand also shows a similar pattern
# - Most countries show a spike in summer and holiday season
# - Germany and Canada show very little seasonal fluctuation
# - United Kingdom is the only country where the most visits occur outside the November to January period 

# In[ ]:


plotByCountry('date_weekday')


# - There is a significant difference in traffic from United States between weekdays and weekends. Other countries do not show this pattern

# In[ ]:


plotByCountry(plotCol='device_deviceCategory',plotType='bar',order=1)


# - India and Thailand have very low desktop to tablet ratio. Tablets don't seem to be popular in these countries
# - Vietnam has the least desktop to mobile ratio

# In[ ]:


plotByCountry('channelGrouping',plotType='bar')


# In[ ]:


plotByCountry('device_operatingSystem',n_labels=5,plotType='bar',order=1)


# - Mac is the leader in United States. It has a significant presence in Thailand and Vietnam but lags in other countries
# - Windows is the leader in all countries except United States
# - Countries with more iOS traffic than Android: United States, Canada, United Kingdom

# In[ ]:


plotByCountry('device_browser',plotType='bar',n_labels=5,order=1)


# - As discussed above, Vietnam and Thailand have high Mac adoption and this could be the reason why they have almost equal traffic from Chrome and Safari

# ### geoNetwork_metro

# In[ ]:


topMetrosTrain = train[~train['geoNetwork_metro'].isin(['not available in demo dataset','(not set)'])]


# In[ ]:


plot_colCount(topMetrosTrain,'geoNetwork_metro',90,16)
plot_totalRevenue(topMetrosTrain,'geoNetwork_metro',90,16)


# In[ ]:


plot_revenuePerUnitCol(topMetrosTrain,'geoNetwork_metro',90,16)


# ### geoNetwork_region

# In[ ]:


topRegions = train['geoNetwork_region'].value_counts().head(80).reset_index()
topRegions.columns = ['region','count']
topRegions = topRegions[(topRegions.region !='not available in demo dataset') &(topRegions.region !='(not set)')]
topRegionsTrain = train[train['geoNetwork_region'].isin(topRegions['region'])]


# In[ ]:


plot_colCount(topRegionsTrain,'geoNetwork_region',80,16)
plot_totalRevenue(topRegionsTrain,'geoNetwork_region',80,16)


# In[ ]:


plot_revenuePerUnitCol(topRegionsTrain,'geoNetwork_region',80,16)


# ### geoNetwork_subContinent

# In[ ]:


plot_colCount(train,'geoNetwork_subContinent',30,15,6)
plot_totalRevenue(train,'geoNetwork_subContinent',30,15,6)
plot_revenuePerUnitCol(train,'geoNetwork_subContinent',30,15,6)


# ### totals_bounces

# In[ ]:


train['totals_bounces'].fillna(0,inplace=True)
train['totals_bounces'] = train['totals_bounces'].astype('int64')


# ### totals_newVisits

# In[ ]:


train['totals_newVisits'].fillna(0,inplace=True)
train['totals_newVisits'] = train['totals_newVisits'].astype('int64')


# ### totals_hits

# In[ ]:


train['totals_hits'] = train['totals_hits'].astype('int64')


# ### totals_pageviews

# In[ ]:


#totals_pageviews
train['totals_pageviews'] = train['totals_pageviews'].astype(float)
print(train['totals_pageviews'].min())
print(train['totals_pageviews'].max())
train['totals_pageviews'].fillna(0,inplace=True)


# In[ ]:


sns.distplot(train['totals_pageviews'])


# In[ ]:


sns.lmplot(data=train,x='totals_pageviews',y='totals_transactionRevenue',
           hue='geoNetwork_continent',col='geoNetwork_continent',col_wrap=2,fit_reg=False)


# ### Traffic Source Columns

# There are 3 Traffic Source related columns that do not have any null values.
# trafficSource_campaign
# trafficSource_medium
# trafficSource_source

# In[ ]:


train['trafficSource_campaign'].value_counts()


# Though trafficsource_campaign does not contain any null values, there are many unknowns.

# In[ ]:


train['trafficSource_medium'].value_counts()


# In[ ]:


train['trafficSource_medium'].replace('(not set)','none',inplace=True)
train['trafficSource_medium'].replace('(none)','none',inplace=True)


# In[ ]:


plot_colCount(train,'trafficSource_medium',30,10,6)
plot_totalRevenue(train,'trafficSource_medium',30,10,6)
plot_revenuePerUnitCol(train,'trafficSource_medium',30,10,6)


# In[ ]:


train['trafficSource_source'].value_counts().head()


# In[ ]:


#trafficSource_adwordsClickInfo.isVideoAd
train['trafficSource_adwordsClickInfo.isVideoAd'].unique()


# Not enough information. All non-values are 'False'. Dropping column.

# In[ ]:


train.drop(['trafficSource_adwordsClickInfo.isVideoAd'],axis=1,inplace=True)


# In[ ]:


#trafficSource_isTrueDirect
train['trafficSource_isTrueDirect'].fillna(0,inplace=True)
train['trafficSource_isTrueDirect'].replace(True,1,inplace=True)
train['trafficSource_isTrueDirect']=train['trafficSource_isTrueDirect'].astype(bool)


# In[ ]:


#trafficSource_adContent
train['trafficSource_adContent'].fillna('Unknown',inplace=True)


# In[ ]:


#trafficSource_adwordsClickInfo.adNetworkType
train['trafficSource_adwordsClickInfo.adNetworkType'].value_counts()
train['trafficSource_adwordsClickInfo.adNetworkType'].fillna('Unknown',inplace=True)


# In[ ]:


#trafficSource_adwordsClickInfo.gclId
train['trafficSource_adwordsClickInfo.gclId'].fillna('Unknown',inplace=True)


# In[ ]:


#trafficSource_adwordsClickInfo.page
train['trafficSource_adwordsClickInfo.page'].fillna(0,inplace=True)
train['trafficSource_adwordsClickInfo.page'] = train['trafficSource_adwordsClickInfo.page'].astype('int64')


# In[ ]:


#trafficSource_referralPath
train['trafficSource_referralPath'].fillna(0,inplace=True)


# In[ ]:


#trafficSource_adwordsClickInfo.slot
train['trafficSource_adwordsClickInfo.slot'].value_counts()


# In[ ]:


train.drop(['trafficSource_adwordsClickInfo.slot'],axis=1,inplace=True)


# In[ ]:


#trafficSource_keyword
train['trafficSource_keyword'].fillna(0,inplace=True)


# In[ ]:


train.drop(['sessionId',
            'visitId','visitStartTime',
            'geoNetwork_region'],axis=1,inplace=True)


# ## Categorical Variables

# In[ ]:


from sklearn import preprocessing
encoder = preprocessing.OneHotEncoder()


# In[ ]:


train.columns


# In[ ]:


unknownLabel = 'zzzUnknown'
leColumns = ['device_deviceCategory','geoNetwork_continent','trafficSource_adwordsClickInfo.adNetworkType',
                'channelGrouping','date_month','date_weekday','geoNetwork_subContinent','trafficSource_medium',
                'geoNetwork_city','geoNetwork_networkDomain','trafficSource_adContent','trafficSource_campaign',
                'trafficSource_keyword','trafficSource_source','device_operatingSystem','device_browser', 
             'geoNetwork_metro','geoNetwork_country','trafficSource_referralPath' ,
             'trafficSource_adwordsClickInfo.gclId']


# In[ ]:


for col in leColumns:
    print('Processing column ' + col)
    le = preprocessing.LabelEncoder()
    le.fit(train[col].astype(str))
    if unknownLabel not in le.classes_:
        le.classes_ = np.append(le.classes_,unknownLabel)
        #adding unknownLabel to handle test set labels not present in train set
    train[col] = le.transform(train[col].astype(str))
    np.save(col +'.npy',le.classes_)


# In[ ]:


train.columns


# ## Correlation

# In[ ]:


plt.figure(figsize=(26,18))
sns.heatmap(train.corr(),annot=True)


# In[ ]:


pd.DataFrame(train.corr()['totals_transactionRevenue']).abs().sort_values('totals_transactionRevenue',ascending=False).head(30)


# ## Extract X and y

# In[ ]:


import math
from sklearn.model_selection import train_test_split


# In[ ]:


X = train.drop(['totals_transactionRevenue','fullVisitorId'],axis=1)
y = train['totals_transactionRevenue'].apply(lambda x:0 if x==0 else math.log(x))    


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


print(len(X_train))
print(len(X_val))


# ### LGBM

# In[ ]:


import lightgbm as lgb
from math import sqrt
from sklearn.metrics import mean_squared_error

params = {'objective' : 'regression','metric' :'rmse','bagging_fraction' :0.5, 'bagging_frequency':8 ,'feature_fraction':0.7, 'learning_rate':0.01, 'max_bin' :100, 
           'max_depth' :7, 'num_leaves':30}

lgbmReg = lgb.LGBMRegressor(**params,n_estimators=1000) 
lgbmReg.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=30,eval_metric='rmse')


# In[ ]:


imp = pd.DataFrame({'Feature':X_val.columns,'Importance':lgbmReg.booster_.feature_importance()})
imp.sort_values(by='Importance',ascending=False)


# In[ ]:


plt.figure(figsize=(14,20))
sns.barplot(data=imp.sort_values(by='Importance',ascending=False),x='Importance',y='Feature')

