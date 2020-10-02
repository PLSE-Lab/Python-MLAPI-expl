#!/usr/bin/env python
# coding: utf-8

# **Objective:**
# We are challenged to analyze a Google Merchandise Store  (where Google swag is sold) customer dataset to predict revenue per customer. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model ,neighbors,preprocessing,svm,tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
from IPython.display import display
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import linear_model , neighbors,preprocessing,svm,tree
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,accuracy_score,make_scorer
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,ExtraTreesClassifier,BaggingClassifier
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
import sys
import os 
from xgboost import XGBRegressor
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
plt.style.use('dark_background')
import vecstack
from vecstack import stacking
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
from scipy.stats import probplot
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score,accuracy_score,make_scorer,log_loss,precision_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC,SVC
from scipy import std ,mean
from scipy.stats import norm
from scipy import stats
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb



import json 
from pandas.io.json import json_normalize


# **About the data:**
# Each row in the dataset is one visit to the store. We are predicting the natural log of the sum of all transactions per user. 
# The data fields in the given files are 
# * fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
# * channelGrouping - The channel via which the user came to the Store.
# * date - The date on which the user visited the Store.
# * device - The specifications for the device used to access the Store.
# * geoNetwork - This section contains information about the geography of the user.
# * sessionId - A unique identifier for this visit to the store.
# * socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
# * totals - This section contains aggregate values across the session.
# * trafficSource - This section contains information about the Traffic Source from which the session originated.
# * visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
# * visitNumber - The session number for this user. If this is the first session, then this is set to 1.
# * visitStartTime - The timestamp (expressed as POSIX time).

# # WEBSITE TERMINOLOGY
# 1. Visit - This is the one piece of information that you really want to know. A visit is one individual visitor who arrives at your web site and proceeds to browse. A visit counts all visitors, no matter how many times the same visitor may have been to your site.
#  
# 
# 2. Page View - This is also called Impression.  Once a visitor arrives at your website, they will search around on a few more pages. On average, a visitor will look at about 2.5 pages. Each individual page a visitor views is tracked as a page view.
#  
# 3. Hits - The real Black Sheep in the family. The average website owner thinks that a hit means a visit but it is very different (see item 1).  A Hit actually refers to the number of files downloaded on your site, this could include photos, graphics, etc. Picture the average web page, it has photos (each photo is a file and hence a hit) and lots of buttons (each button is a file and hence a hit). On average, each page will include 15 hits.
#  
# To give you an example -  Using the average statistics , 1 Visit to an average web site will generate 3 Page Views and 45 Hits.
#  
# 4. Traffic Sources - How do visitors find your site
# 
# Direct Navigation (type URL in traffic, bookmarks, email links w/o tracking codes, etc.) 
# 
# Referral Traffic (from links across the web, social media, in trackable email, promotion & branding campaign links)
# 
# Organic Search (queries that sent traffic from any major or minor web search engines)
# 
# PPC (click through from Pay Per click sponsored ads, triggered by targeted keyphrases)

# In[ ]:


def load_df(csv_path="C:/Users/DELL/Downloads/train.csv", nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, 
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# ### Loading data with the previus function to handle json columns.. Because it takes time we save it to a csv file!

# In[ ]:


# data = load_df()
# test_data = load_df("C:/Users/DELL/Downloads/googleStoreTest.csv")


# In[ ]:


# data.to_csv(r'C:/Users/DELL/Downloads/googleStore.csv')
# test_data.to_csv(r'C:/Users/DELL/Downloads/googleStoreTest.csv')


# In[ ]:


data=pd.read_csv("../input/googlestore-loaded-json/googleStore.csv", dtype={'fullVisitorId': 'str'},low_memory=False)
test_data=pd.read_csv('../input/googlestore-loaded-json/googleStoreTest.csv', dtype={'fullVisitorId': 'str'},low_memory=False)


# In[ ]:


# allData=pd.concat([data,test_data],axis=0,keys=['x','y'])
# test and train together


# # FILLING NA 

# In[ ]:


data.isna().sum()[data.isna().sum()>0]


# # channel Grouping
# 
#  * Traffic Sources - How do visitors find your site
# 
# Direct Navigation (type URL in traffic, bookmarks, email links w/o tracking codes, etc.) 
# 
# Referral Traffic (from links across the web, social media, in trackable email, promotion & branding campaign links)
# 
# Organic Search ( traffic from any major or minor web search engines)
# 
# Paid search(click through from Pay Per click sponsored ads, triggered by targeted keyphrases)
# 
# Affiliate (from  ads on websites)

# In[ ]:



data['positiveRevenue']=data['totals.transactionRevenue'].map(lambda x: 1 if (x>0)  else 0)


# In[ ]:


plt.style.use('dark_background')


sns.set(rc={'figure.figsize':(16.7,7)})

sns.countplot(data['channelGrouping'],hue=data['positiveRevenue'])


# Referral has a good ratio of positive revenues,Organic brings the most visitors.

# # Visits

# In[ ]:


print('ratio of visits with non zero revenue/total Visits:',1-data['totals.transactionRevenue'].isna().sum()/len(data))


# In[ ]:


plt.figure(figsize=(15,6))
uniqueCostumers=data['fullVisitorId'].value_counts()
print('most of the visits are from unique costumers',len(uniqueCostumers)/len(data))
uniqueCostumers
sns.barplot(x=['all visits','unique visits'],y=[len(data),data['fullVisitorId'].nunique()])


# In[ ]:


print('only', 100*(1-data.groupby('fullVisitorId').mean()['totals.transactionRevenue'].isnull().sum()/len(uniqueCostumers)),'% of unique costumers produce revenue' ) 


# In[ ]:


data['totals.transactionRevenue'].fillna(0,inplace=True)


# In[ ]:



data['totals.transactionRevenue']=np.log(data['totals.transactionRevenue']+1)


# In[ ]:


ax=sns.distplot(data[data['totals.transactionRevenue']>0]['totals.transactionRevenue'])
print('Overall mean of visits:',data['totals.transactionRevenue'].mean(),
      
      '\n mean of positive revenue visits:',data[data['totals.transactionRevenue']>0]['totals.transactionRevenue'].mean())


#  * gini coefficient of total transaction Revenue (0,1) bigger means bigger inequality in revenue distribution

# In[ ]:


def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area
print('gini of positive Revenue',gini(data['totals.transactionRevenue'][data['positiveRevenue']>0]),'\n \n gini of total Revenue',gini(data['totals.transactionRevenue']))


# In[ ]:


def dataNumberOfVisitsGreaterThan(n):
    
    return data[data['fullVisitorId'].isin(data['fullVisitorId'].value_counts()[data['fullVisitorId'].value_counts()>n].index)]

def dataNumberOfVisitsLessThan(n):
    
    return data[data['fullVisitorId'].isin(data['fullVisitorId'].value_counts()[data['fullVisitorId'].value_counts()<n].index)]


def dataNumberOfVisits(n):
    
    return data[data['fullVisitorId'].isin(data['fullVisitorId'].value_counts()[data['fullVisitorId'].value_counts()==n].index)]


# # TRANSACTION--VISITS

#  Mean Transacion increases when visits increase both in positive-revenue and general-revenue visits
# 

# In[ ]:


from datetime import datetime
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["_weekday"] = df['date'].dt.weekday 
    df["_day"] = df['date'].dt.day 
    df["_month"] = df['date'].dt.month 
    df["_year"] = df['date'].dt.year 
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    
    return df

date_process(data)
date_process(test_data)


# # DEVICE DATA VISUALAZATION

# In[ ]:


browsers=data.groupby(data['device.browser']).filter(lambda x:x['device.browser'].size*100/len(data)>1)['device.browser'].value_counts()

pbrowsers=data[data['positiveRevenue']>0].groupby(data['device.browser']).filter(lambda x:x['device.browser'].size*100/len(data)>1)['device.browser'].value_counts()


# browsers with more than 1% usage

# In[ ]:


sns.set(palette='dark')
plt.style.use('dark_background')
fig,axes=plt.subplots(2,3,figsize=(19,12))
axes[0,1].set_title('DEVICE GRAPH',fontsize=26)
sns.barplot(browsers.index,browsers.values,ax=axes[0,0])
sns.barplot(data['device.deviceCategory'].value_counts().index,data['device.deviceCategory'].value_counts().values,ax=axes[0,1])
sns.barplot(data['device.operatingSystem'].value_counts().index[0:6],data['device.operatingSystem'].value_counts().values[0:6],ax=axes[0,2])

axes[1,1].set_title('DEVICE GRAPH FOR POSITIVE REVENUE',fontsize=26)
sns.barplot(pbrowsers.index,pbrowsers.values,ax=axes[1,0])
sns.countplot(data[data['positiveRevenue']>0]['device.deviceCategory'],ax=axes[1,1])
sns.countplot(data[data['positiveRevenue']>0]['device.operatingSystem'],ax=axes[1,2],order=data['device.operatingSystem'].value_counts().index[0:6])


# * only chrome users have positive revenue.Others have  low amount of visits
# * mobile and tablets have worse ratio from desktops
# * Macintosh  way bigger  ratio, chrome OS and linux too,while everything less is smaller with emphasis to windows

# In[ ]:


fig,axes=plt.subplots(1,3,figsize=(21,10))
axes[1].set_title('DEVICE GRAPH',fontsize=26)
sns.boxplot('device.operatingSystem','totals.transactionRevenue',data=data[(data['device.browser'].isin(browsers.index))&data['positiveRevenue']==1],ax=axes[0])
sns.boxplot('device.deviceCategory','totals.transactionRevenue',data=data[(data['device.browser'].isin(browsers.index))&data['positiveRevenue']==1],ax=axes[1])
sns.boxplot('device.browser','totals.transactionRevenue',data=data[(data['device.browser'].isin(browsers.index))&data['positiveRevenue']==1],ax=axes[2])
ylim = (11, 22)
plt.setp(axes, ylim=ylim);


# chrome OS ,desktop,and chrome are the highest of each category

# In[ ]:


sns.set(palette='bright')
crosstab_eda = pd.crosstab(columns=data['device.browser'][data['device.browser'].isin(browsers.index)], 
                           index=data['device.deviceCategory'])
crosstab_eda.plot(kind="bar",    
                 figsize=(14,7), 
                 stacked=True)  
plt.title("Most frequent Browser's by Device Category", fontsize=22) 
plt.xlabel("Device Name", fontsize=19)        ;  
plt.ylabel("Browser Count", fontsize=19)         
plt.xticks(rotation=0) ;


# In[ ]:



df_train=data
crosstab_eda = pd.crosstab(index=df_train['device.deviceCategory'], 
                           columns=df_train[df_train['device.operatingSystem']\
                                            .isin(df_train['device.operatingSystem']\
                                                  .value_counts()[:6].index.values)]['device.operatingSystem'])
crosstab_eda.plot(kind="bar",    
                 figsize=(14,7), 
                 stacked=True)  
plt.title("Most frequent OS's by Device Category", fontsize=22) 
plt.xlabel("Device Name", fontsize=19)        
plt.ylabel("Count Device x OS", fontsize=19)         
plt.xticks(rotation=0) ;


# # DATES

# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(21,10))
axes[1].set_title('Positive---0 revenue',fontsize=26)
sns.countplot(data['_year'],hue=data['positiveRevenue'])
sns.countplot(data['_year'],ax=axes[0])
min(data['date']),max(data['date'])


# 2016 data has only 4 months while 2017 8. so 2017 is a worse year 
# 

# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(20,11))
sns.countplot(data['_weekday'],ax=axes[0])
weekdayDist=data[data['positiveRevenue']==1].groupby('_weekday')['date'].count()
sns.barplot(weekdayDist.index,weekdayDist.values,ax=axes[1])


# Lowest counts on weekends!Also,weekends go even worse for positive revenue ratio compared to others.

# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(20,11))
sns.countplot(data['_month'],ax=axes[0])
axes[1].set_title('How many people buy',fontsize=19)
axes[0].set_title('How many people visit',fontsize=19)
monthDist=data[data['positiveRevenue']==1].groupby('_month')['date'].count()
sns.barplot(monthDist.index,monthDist.values,ax=axes[1])


# Novmber & October have the highest counts.But they dont go so well revenue wise. December's ratio is the largest

# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(20,11))
sns.countplot(data['_day'],ax=axes[0])
axes[1].set_title('How many people buy',fontsize=19)
axes[0].set_title('How many people visit',fontsize=19)
monthDist=data[data['positiveRevenue']==1].groupby('_day')['date'].count()
sns.barplot(monthDist.index,monthDist.values,ax=axes[1])


# In[ ]:


fig,axes=plt.subplots(2,1,figsize=(20,11))
sns.countplot(data['_visitHour'],ax=axes[0])
axes[1].set_title('How many people buy',fontsize=19)
axes[0].set_title('How many people visit',fontsize=19)
monthDist=data[data['positiveRevenue']==1].groupby('_visitHour')['date'].count()
sns.barplot(monthDist.index,monthDist.values,ax=axes[1])


# 18 to 21 highest traffic.
# 
# Huge decrease revenue-wise from 5 to 19

# In[ ]:


date_sales = ['_visitHour', '_weekday']

cm = sns.light_palette("yellow", as_cmap=True)
pd.crosstab(df_train[date_sales[0]], df_train[date_sales[1]], 
            values=df_train["totals.transactionRevenue"], aggfunc=[np.sum]).style.background_gradient(cmap = cm)


# # LOCATION
# 

# In[ ]:


country_tree = df_train["geoNetwork.country"].value_counts() 
import squarify
country_tree = round((df_train["geoNetwork.country"].value_counts()[:30]/len(df_train['geoNetwork.country']) * 100),2)
fig,axes=plt.subplots(2,1,figsize=(16,14))
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, value=country_tree.values, alpha=.7,ax=axes[0])
g.set_title("Top countries",fontsize=20)
g.set_axis_off()

country_tree = round((df_train[df_train['positiveRevenue']>0]["geoNetwork.country"].value_counts()[:10]/len(df_train[df_train['positiveRevenue']>0]['geoNetwork.country']) * 100),2)
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, value=country_tree.values, alpha=.7,ax=axes[1])
g.set_title("Top positive Revenue countries ",fontsize=20)
g.set_axis_off()


# Unites States takes 95% of positive revenues & only 40% of total visits

# In[ ]:



sns.set(palette='bright')
plt.style.use('dark_background')
plt.figure(figsize=(16,8))
sns.countplot(df_train[df_train['geoNetwork.subContinent'].isin(df_train['geoNetwork.subContinent'].value_counts()[:13].index.values)]['geoNetwork.subContinent'], palette="hls") 
plt.title("Most frequent SubContinents", fontsize=22) 
plt.xlabel("subContinent", fontsize=20) 
plt.ylabel("SubContinent Count", fontsize=20) 
plt.xticks(rotation=52);


# # TOTAL 

# A bounce occurs when a web site visitor only views a single page on a website, that is, the visitor leaves a site without visiting any other pages before a specified session-timeout occurs. 

# In[ ]:


fig,axes=plt.subplots(1,2)

sns.regplot(data['totals.pageviews'],data['totals.transactionRevenue'],ax=axes[0])

sns.regplot(data['totals.hits'],data['totals.transactionRevenue'],ax=axes[1])


#  Almost Similar plots

# In[ ]:


fig,axes=plt.subplots(1,3)

a=sns.barplot(data['positiveRevenue'],data['totals.pageviews'],ax=axes[0])
a.set_ylabel('PAGEVIEWS',fontsize=20)

a=sns.barplot(data['positiveRevenue'],data['totals.hits'],ax=axes[1])
a.set_ylabel('HITS',fontsize=20)


a=sns.barplot(data['positiveRevenue'],data['totals.bounces'],ax=axes[2])
a.set_ylabel('BOUNCES',fontsize=20);


# As expected when we have positive revenue we dont have bounce

# # Cleaning the data:

#  * Label encoder to categorical columns

# In[ ]:


cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "device.isMobile","geoNetwork.city", "geoNetwork.continent","geoNetwork.country",
            "geoNetwork.metro","geoNetwork.networkDomain", "geoNetwork.region",
            "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
for col in cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data[col].values.astype('str')))
    data[col] = lbl.transform(list(data[col].values.astype('str')))


# In[ ]:


get_ipython().run_cell_magic('time', '', "for col in cols:\n    lbl.fit(list(test_data[col].values.astype('str')))\n    test_data[col] = lbl.transform(list(test_data[col].values.astype('str')))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef transform(data):\n            \n    data['_visitStartHour'] = data['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))\n    data['_visitStartHour'] = data['_visitStartHour'].astype(int)\n        \n\n        \n    data['totals.pageviews'].fillna(1, inplace=True)\n    data['totals.newVisits'].fillna(0, inplace=True)\n    data['totals.bounces'].fillna(0, inplace=True)\n    data['totals.pageviews'] = data['totals.pageviews'].astype(int)\n    data['totals.newVisits'] = data['totals.newVisits'].astype(int)\n    data['totals.bounces'] = data['totals.bounces'].astype(int)\n\n    \n    \n    data['meanHitsHour'] = data.groupby(['_visitHour'])['totals.hits'].transform('mean')\n    data['meanHitsDay'] = data.groupby(['_day'])['totals.hits'].transform('mean')\n    data['meanHitsWeekday'] = data.groupby(['_weekday'])['totals.hits'].transform('mean')\n    data['meanHitsMonth'] = data.groupby(['_month'])['totals.hits'].transform('mean')\n    data['hitsHour'] = data.groupby(['_visitHour'])['totals.hits'].transform('sum') \n    data['hitsDay'] = data.groupby(['_day'])['totals.hits'].transform('sum')\n    data['hitsWeekday'] = data.groupby(['_weekday'])['totals.hits'].transform('sum')\n    data['hitsMonth'] = data.groupby(['_month'])['totals.hits'].transform('sum')\n    \n    \n    for i in ['device.isMobile']: #using mobile or not in particular visit vs mean\n        x = data.groupby('fullVisitorId')[i].mean()\n        data['m'] = data.fullVisitorId.map(x)\n        data['isMobileDifference']=data['device.isMobile']-data['m']\n        data.drop('m',1,inplace=True)\n        \n    \n    for i in ['totals.bounces','totals.newVisits']: ##mean \n        x = data.groupby('fullVisitorId')[i].mean()\n        data['visitorMean_' +i] = data.fullVisitorId.map(x)\n\n\n    for i in ['totals.hits', 'totals.pageviews']: ##mean ,max,min,sum\n        x = data.groupby('fullVisitorId')[i].mean()\n        maxx = data.groupby('fullVisitorId')[i].max()\n        minn = data.groupby('fullVisitorId')[i].min()\n        summ = data.groupby('fullVisitorId')[i].sum()\n        data['visitorMean_' +i] = data.fullVisitorId.map(x)\n        data['visitorMax_' +i] = data.fullVisitorId.map(maxx)\n        data['visitorMin_' +i] = data.fullVisitorId.map(minn)\n        data['visitorMean_' +i] = data.fullVisitorId.map(summ)\n\n    for i in ['visitNumber']:\n        maxx = data.groupby('fullVisitorId')[i].max()\n        minn=data.groupby('fullVisitorId')[i].min()\n        data['visitorMax_' + i] = data.fullVisitorId.map(maxx) \n#         data['visitorDif_'+i]=data.fullVisitorId.map(maxx-minn)\n        \n        \n    for i in ['date']: #date\n        maxx = data.groupby('fullVisitorId')[i].max()\n        minn=data.groupby('fullVisitorId')[i].min()\n        data['visitorDiff_'+i]=data.fullVisitorId.map((maxx-minn))\n\n\ntransform(data)\ntransform(test_data)\n")


# In[ ]:


test_data.isnull().sum().any(),data.isnull().sum().any()


# In[ ]:



for j in data.columns:
    if j not in test_data.columns:
        print(j)
        
        


# In[ ]:


def datadrop(data):
    data.drop(['sessionId','visitId','visitStartTime','date'],1,inplace=True)
    
    
    
datadrop(data)
datadrop(test_data)


const_cols = [c for c in data.columns if data[c].nunique(dropna=False)==1 ]
data.drop(const_cols,1,inplace=True)
test_data.drop(const_cols,1,inplace=True)


# In[ ]:


data['visitorDiff_date']=data['visitorDiff_date'].apply(lambda x : x.days)
test_data['visitorDiff_date']=test_data['visitorDiff_date'].apply(lambda x : x.days)


# In[ ]:


x=data.drop(['totals.transactionRevenue','positiveRevenue','fullVisitorId','Unnamed: 0','trafficSource.campaignCode'],1)
y=data['totals.transactionRevenue']


xtr,xtest,ytr,ytest=train_test_split(x,y,test_size=0.25)


# In[ ]:


xpred=test_data.drop(['fullVisitorId','Unnamed: 0'],1)


# # MODEL & PARAMETER TUNING

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
def model(model):
    reg=model
    reg.fit(xtr,ytr)

    y_pred=reg.predict(xtest)
    y_pred[y_pred<1]=0
    score=mean_squared_error(ytest, y_pred)
    print(sqrt(score))
    


# In[ ]:



from sklearn.linear_model import BayesianRidge

model(BayesianRidge())


# In[ ]:


params = {
     "objective" : "regression",
     "metric" : "rmse", 
     
     "bagging_fraction" : 0.7,
     "feature_fraction" : 0.5,
     "bagging_frequency" : 5,
     "bagging_seed" : 2018,
     "verbosity" : -1
 }

gridModel=lgb.LGBMRegressor(**params)    

gridParams = { 
     "min_child_samples" : [100,50,200],
 'learning_rate': [0.1,0.01,0.001],
 'num_leaves': [12,20,30,40,50]
}


# In[ ]:


#make mean_squared_error scorrer

# grid = GridSearchCV(gridModel, gridParams,
#                     verbose=0,
#                     cv=5, n_jobs=-1)
# grid.fit(xtr, ytr)

# Print the best parameters found
# print(grid.best_params_)


# In[ ]:


def lgbCustom(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        
        
        "num_leaves" : 50,
        "min_child_samples" : 200, #best params from grid Search!!
        "learning_rate" : 0.1,
        
        
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=300, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y[pred_test_y<1]=0
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    
    pred_val_y[pred_val_y<1]=0

    return pred_test_y, model, pred_val_y


pred, model, val = lgbCustom(xtr, ytr, xtest, ytest, xpred)


# In[ ]:


print(sqrt(mean_squared_error(val,ytest)))


# In[ ]:


submission = pd.DataFrame({'fullVisitorId':test_data['fullVisitorId'], 'PredictedLogRevenue':pred})

submission["PredictedLogRevenue"] = np.expm1(submission["PredictedLogRevenue"])

submission_sum = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
submission_sum["PredictedLogRevenue"] = np.log1p(submission_sum["PredictedLogRevenue"])


# In[ ]:


submission_sum.describe()


# In[ ]:


sns.kdeplot(data=submission_sum['PredictedLogRevenue'][submission_sum['PredictedLogRevenue']>0],shade=True)


# In[ ]:


# submission_sum.to_csv("submission.csv")
# submission_sum.head(10)


# # Feature Importance:
# 
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
lgb.plot_importance(model, max_num_features=45, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM Feature Importance", fontsize=15)
plt.show()


# if you have questions feel free to answer 
# 
# 
