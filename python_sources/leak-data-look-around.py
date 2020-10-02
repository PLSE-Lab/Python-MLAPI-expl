#!/usr/bin/env python
# coding: utf-8

# **<font size=5>Objectives:<font>**
# <font size=3>
# * Explore importance of leak data in the whole data set.  
# *     What is the % missing 'totals.transactionRevenue' in leak data row.  
# *     What is the % of frequence in leak data row.  
# *     What is the correlation of importance feature between 'totals.transactionRevenue' <font>
#     
# **<font size=3>I would try my best hope you like it<font>**
# 
# 

# In[ ]:


import pandas as pd
import numpy as np

# DRAGONS
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

# plots
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# pandas / plt options
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (14, 7)
font = {'family' : 'verdana',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

# remove warnings
import warnings
warnings.simplefilter("ignore")

# garbage collector
import gc
gc.enable()


# **<font size=5>load data</font>**

# In[ ]:


import os
print(os.listdir("../input"))

train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', dtype={'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': np.int64})
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', dtype={'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': np.int64})
train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


traincolumns_1 = train.columns
traincolumns_1


# In[ ]:


train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
dataset = pd.concat(objs=[train_store_1, train_store_2], axis=0)
dataset.info()
del dataset


# In[ ]:


for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)


# **<font size=5>Check extend feature</font>**

# In[ ]:


train = train.merge(pd.concat([train_store_1, train_store_2], sort=False), how="left", on="visitId")
test = test.merge(pd.concat([test_store_1, test_store_2], sort=False), how="left", on="visitId")

# Drop Client Id
for df in [train, test]:
    df.drop("Client Id", 1, inplace=True)
leakcolumns = [x for x in train.columns if x not in traincolumns_1]
leakcolumns


# In[ ]:


train.head()


# In[ ]:


train.info()


# 
# **<font size=5>Processed feature for further use</font>**

# In[ ]:


train['has_revenue'] = train['totals.transactionRevenue'].apply(lambda x: 1 if x > 0 else 0)

for df in [train, test]:
    df['browser.os'] = df['device.browser'] + '_' + df['device.operatingSystem']

for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60


# **<font size=5>Knowing the missing values</font>**

# In[ ]:


def missing_values(data):
    print(data.shape)
    total = data.isnull().sum().sort_values(ascending = False) # getting the sum of null values and ordering
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) #getting the percent and order of null
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Concatenating the total and percent
    print("Total columns at least one Values: ")
    print (df[~(df['Total'] == 0)]) # Returning values of nulls different of 0
    
missing_values(train) 

print("\n Total of Sales % of Total: ", round((train[train['totals.transactionRevenue'] !=         np.nan]['totals.transactionRevenue'].count() / len(train['totals.transactionRevenue']) * 100),4))


train_leak = train[train['Sessions'].isnull().values==False]

missing_values(train_leak) 

print("\n Total of Sales % of train_leak: ", round((train_leak[train_leak['totals.transactionRevenue'] !=         np.nan]['totals.transactionRevenue'].count() / len(train_leak['totals.transactionRevenue']) * 100),4))


# <font size=3>
# Whole data row 1.2744% have 'totals.transactionRevenue' value  
#     
#  Leak data row 42.4326% have 'totals.transactionRevenue' value
# </font>
# 

# In[ ]:


test.info()
missing_values(test) 


# **<font size=5>Distribuition of transactions Revenues</font>**

# In[ ]:


for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    del df
gc.collect()


# In[ ]:


def plotdisturbtion(df_train):
    # Printing some statistics of our data
    print("Transaction Revenue Min Value: ", 
          df_train[df_train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"].min()) # printing the min value
    print("Transaction Revenue Mean Value: ", 
          df_train[df_train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"].mean()) # mean value
    print("Transaction Revenue Median Value: ", 
          df_train[df_train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"].median()) # median value
    print("Transaction Revenue Max Value: ", 
          df_train[df_train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"].max()) # the max value

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    ax = sns.distplot(np.log(df_train[df_train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"] + 0.01), bins=40, kde=True)
    ax.set_xlabel('Transaction RevenueLog', fontsize=15) #seting the xlabel and size of font
    ax.set_ylabel('Distribuition', fontsize=15) #seting the ylabel and size of font
    ax.set_title("Distribuition of Revenue Log", fontsize=20) #seting the title and size of font

    plt.subplot(1,2,2)
    plt.scatter(range(df_train.shape[0]), np.sort(df_train['totals.transactionRevenue'].values))
    plt.xlabel('Index', fontsize=15) # xlabel and size of words
    plt.ylabel('Revenue value', fontsize=15) # ylabel and size of words
    plt.title("Revenue Value Distribution", fontsize=20) # Setting Title and fontsize
    plt
    
print('whole data')
plotdisturbtion(train)
print('leak data row')
plotdisturbtion(train_leak)


# **<font size=5>Distribuition of Category columns</font>**

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

def barplot_percentage(count_feat, color1= 'green', 
                       color2= 'rgb(26, 118, 255)',color3= 'red',num_bars= None):

    train_channel = 100*train[train[count_feat].isin(train[count_feat]            .value_counts()[:7].index.values)][count_feat].value_counts()/len(train)
    train_channel = train_channel.to_frame().reset_index()

    test_channel = 100*test[test[count_feat].isin(train[count_feat]            .value_counts()[:7].index.values)][count_feat].value_counts()/len(test)
    test_channel = test_channel.to_frame().reset_index()
    
    leak_channel = 100*train_leak[train_leak[count_feat].isin(train[count_feat]            .value_counts()[:7].index.values)][count_feat].value_counts()/len(train_leak)
    leak_channel = leak_channel.to_frame().reset_index()
    
    if num_bars:
        train_channel = train_channel.head(num_bars)
        test_channel = test_channel.head(num_bars)
        leak_channel = leak_channel.head(num_bars)

    trace0 = go.Bar(
        x=train_channel['index'],
        y=train_channel[count_feat],
        name='Train set',
        marker=dict(color=color1)
    )
    trace1 = go.Bar(
        x=test_channel['index'],
        y=test_channel[count_feat],
        name='Test set',
        marker=dict(color=color2,)
    )
    trace2 = go.Bar(
        x=leak_channel['index'],
        y=leak_channel[count_feat],
        name='leak data set',
        marker=dict(color=color3,)
    )

    layout = go.Layout(
        height=400,
        title='{} grouping'.format(count_feat),
        xaxis=dict(
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        yaxis=dict(
            title='Percentage of visits',
            titlefont=dict(size=16, color='rgb(107, 107, 107)'),
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    fig = go.Figure(data=[trace0, trace1,trace2], layout=layout)
    iplot(fig)
    


# In[ ]:


for x in ['geoNetwork.country','geoNetwork.region','geoNetwork.metro','channelGrouping','browser.os']:
    barplot_percentage(x)


# In[ ]:


for x in ['geoNetwork.networkDomain','trafficSource.medium','device.browser']:
    barplot_percentage(x)


# **<font size=5>Distribuition and Trend of Number columns</font>**

# In[ ]:


train.columns


# In[ ]:


print(train['has_revenue'].unique())
# print(train['has_revenue'])
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
train_leak['totals.transactionRevenue'] = train_leak['totals.transactionRevenue'].fillna(0)
train.shape


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def plotrevenues(col_1,col_2):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and revenue")
    ax = sns.scatterplot(x=col_1 , y='totals.transactionRevenue',
                     data=train,color='orange', hue='has_revenue')
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and revenue")
    ax = sns.scatterplot(x=col_2, y='totals.transactionRevenue',
                     data=train,color='orange', hue='has_revenue')
    plt
    
    cnt_col1 = train.groupby(col_1)['totals.transactionRevenue'].agg(['mean','count'])
    cnt_col2= train.groupby(col_2)['totals.transactionRevenue'].agg(['mean','count'])
    
    cnt_col1 = cnt_col1.reset_index()
    cnt_col2 = cnt_col2.reset_index()
    
    cnt_col1 = cnt_col1[cnt_col1['count']>10]
    cnt_col2 = cnt_col2[cnt_col2['count']>10]
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and mean revenue")
    ax = sns.scatterplot(x=col_1, y='mean',
                     data=cnt_col1,color='blue')
    
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and mean revenue")
    ax = sns.scatterplot(x=col_2, y='mean',
                     data=cnt_col2,color='blue')
    plt

plotrevenues('totals.pageviews','totals.hits')
# plotrevenues('next_session_1','next_session_2')
plotrevenues('visitNumber','sess_date_dow')
# plotrevenues('sess_date_hours','sess_date_dom')


# **<font size=3>Wow totals.pageviews totals.hits highly correlated with revenue</font>**

# **<font size=3>Filter feature session1 and session2</font>**

# In[ ]:


train_session_filter = train[train['next_session_1']>-500000]
train_session_filter = train_session_filter[train_session_filter['next_session_2']>-500000]
train_session_filter = train_session_filter[train_session_filter['totals.transactionRevenue'] < 100000000]
def plotrevenues_filter(col_1,col_2):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and revenue")
    ax = sns.scatterplot(x=col_1 , y='totals.transactionRevenue',
                     data=train_session_filter,color='orange', hue='has_revenue')
    
#     plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and revenue")
    ax = sns.scatterplot(x=col_2, y='totals.transactionRevenue',
                     data=train_session_filter,color='orange', hue='has_revenue')
    plt
    
    
    t_s_f = train_session_filter[[col_1,col_2,'totals.transactionRevenue']]
    
    t_s_f[col_1] = t_s_f[col_1].apply(lambda x: int(x/10))
    t_s_f[col_2] = t_s_f[col_2].apply(lambda x: int(x/10))
    
    
    cnt_col1 = t_s_f.groupby(col_1)['totals.transactionRevenue'].agg(['mean','count'])
    cnt_col2= t_s_f.groupby(col_2)['totals.transactionRevenue'].agg(['mean','count'])
    
    cnt_col1 = cnt_col1.reset_index()
    cnt_col2 = cnt_col2.reset_index()
    
    cnt_col1 = cnt_col1[cnt_col1['count']>5]
    cnt_col2 = cnt_col2[cnt_col2['count']>5]
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and mean revenue")
    ax = sns.scatterplot(x=col_1, y='mean',
                     data=cnt_col1,color='blue')
    
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and mean revenue")
    ax = sns.scatterplot(x=col_2, y='mean',
                     data=cnt_col2,color='blue')
    plt
    
plotrevenues_filter('next_session_1','next_session_2')


# **<font size=5>Distribuition and Trend of leak Number columns</font>**

# In[ ]:


for df in [train_leak]:
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Revenue"].fillna(0.0, inplace=True)
    
for df in [train_leak]:
    df["Avg. Session Duration"][df["Avg. Session Duration"] == 0] = "00:00:00"
    df["Avg. Session Duration"] = df["Avg. Session Duration"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df["Bounce Rate"] = df["Bounce Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)
    df["Goal Conversion Rate"] = df["Goal Conversion Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)


# In[ ]:


leakcolumns


# In[ ]:


def plotrevenues_leak(col_1,col_2):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and revenue")
    ax = sns.scatterplot(x=col_1 , y='totals.transactionRevenue',
                     data=train_leak_filter,color='orange', hue='has_revenue')
    
#     plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and revenue")
    ax = sns.scatterplot(x=col_2, y='totals.transactionRevenue',
                     data=train_leak_filter,color='orange', hue='has_revenue')
    plt
        
    cnt_col1 = train_leak_filter.groupby(col_1)['totals.transactionRevenue'].agg(['mean','count'])
    cnt_col2= train_leak_filter.groupby(col_2)['totals.transactionRevenue'].agg(['mean','count'])
    
    cnt_col1 = cnt_col1.reset_index()
    cnt_col2 = cnt_col2.reset_index()
    
    cnt_col1 = cnt_col1[cnt_col1['count']>5]
    cnt_col2 = cnt_col2[cnt_col2['count']>5]
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title("Number of " + col_1 + " and mean revenue")
    ax = sns.scatterplot(x=col_1, y='mean',
                     data=cnt_col1,color='blue')
    
    plt.subplot(1,2,2)
    plt.title("Number of " + col_2 + "and mean revenue")
    ax = sns.scatterplot(x=col_2, y='mean',
                     data=cnt_col2,color='blue')
    plt


# **<font size=3>Remove session > 200</font>**

# In[ ]:


train_leak_filter = train_leak[train_leak['Sessions']<200] 


# In[ ]:


plotrevenues_leak('Sessions','Avg. Session Duration')
plotrevenues_leak('Bounce Rate','Revenue')
plotrevenues_leak('Transactions','Goal Conversion Rate')


# **<font size=3>leak data all highly correlated to revenue</font>**

# **<font size=5>Process some category columns</font>**

# In[ ]:


def topcolumn_proces(colname):
    browsers_top= train[train[colname].isin(train[colname].value_counts()[:6].index.values)][colname]
    browsercolumns = browsers_top.unique()
    def source_mapping(x):
        if x in browsercolumns:
            return x
        else:
            return 'others'
    train[colname+'_new'] = train[colname].map(lambda x:source_mapping(str(x))).astype('str')
    train_leak[colname+'_new'] = train_leak[colname].map(lambda x:source_mapping(str(x))).astype('str')
for x in ['geoNetwork.country','channelGrouping','browser.os','trafficSource.medium']:
    topcolumn_proces(x)    


# **<font size=3>Filter revenue > 5000000000</font>**

# In[ ]:


train_Revenue_filter = train[train['totals.transactionRevenue']<5000000000]


# **<font size=5>Plot jointplot plot</font>**

# In[ ]:


# train_n = train[train["Bounce Rate"] > 0]
def plotjointplot(df,colname,cate_name):
    g = sns.jointplot(df[colname], df['totals.transactionRevenue'],  s=1, size=12)
    g.ax_joint.cla()
    plt.sca(g.ax_joint)
    categorycolumns = train[cate_name+'_new'].unique()
    for cate_col in categorycolumns:
        v = df[df[cate_name+'_new'] == cate_col]
        plt.scatter(v[colname], v['totals.transactionRevenue'], s=4, label='{}'.format(cate_col))
    plt.xlabel(colname,fontsize=15)
    plt.ylabel('totals.transactionRevenue',fontsize=15)
    plt.legend()
    plt


# **<font size=3>Filter pageviews hits visitnumber</font>**

# In[ ]:


train_Ru_Pg_filter = train_Revenue_filter[train_Revenue_filter['totals.pageviews']<200]
train_Ru_hits_filter = train_Revenue_filter[train_Revenue_filter['totals.hits']<250]
train_Ru_vis_filter = train_Revenue_filter[train_Revenue_filter['visitNumber']<100]

# for x in ['geoNetwork.country','channelGrouping','browser.os','trafficSource.medium']:
#     plotjointplot(train_Ru_Pg_filter,'totals.pageviews',x)
plotjointplot(train_Ru_Pg_filter,'totals.pageviews','geoNetwork.country')
plotjointplot(train_Ru_hits_filter,'totals.hits','browser.os')
plotjointplot(train_Ru_vis_filter,'visitNumber','geoNetwork.country')


# **<font size=3>Filter  leak data revenue and sessions</font>**

# In[ ]:


train_leak_filter = train_leak[train_leak['totals.transactionRevenue']<200000000]
train_leak_filter = train_leak_filter[train_leak_filter['Sessions']<200] 


# In[ ]:


for x in ['browser.os']:
    for y in leakcolumns:
        plotjointplot(train_leak_filter,y,x)


# **<font size=5>Geolocation plot to visually understand the data</font>**  
# 
# 
# **<font size=3>whole data set :  vistis</font>**

# In[ ]:


def plotmapvisit(df_train):
    # Counting total visits by countrys
    countMaps = pd.DataFrame(df_train['geoNetwork.country'].value_counts()).reset_index()
    countMaps.columns=['country', 'counts'] #renaming columns
    countMaps = countMaps.reset_index().drop('index', axis=1) #reseting index and droping the column

    data = [ dict(
            type = 'choropleth',
            locations = countMaps['country'],
            locationmode = 'country names',
            z = countMaps['counts'],
            text = countMaps['country'],
            autocolorscale = False,
            marker = dict(
                line = dict (
                    color = 'rgb(180,180,180)',
                    width = 0.5
                ) ),
            colorbar = dict(
                autotick = False,
                tickprefix = '',
                title = 'Number of Visits'),
          ) ]

    layout = dict(
        title = 'Couting Visits Per Country',
        geo = dict(
            showframe = False,
            showcoastlines = True,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    figure = dict( data=data, layout=layout )
    iplot(figure, validate=False, filename='map-countrys-count')
plotmapvisit(train)


# **<font size=3>leak data set : vistis</font>**

# In[ ]:


plotmapvisit(train_leak)


# 
# **<font size=3>whole data set :  revenues counts</font>**

# In[ ]:


def plotmaprevenues(df_train):
   # I will crete a variable of Revenues by country sum
   sumRevMaps = df_train[df_train['totals.transactionRevenue'] > 0].groupby("geoNetwork.country")["totals.transactionRevenue"].count().to_frame().reset_index()
   sumRevMaps.columns = ["country", "count_sales"] # renaming columns
   sumRevMaps = sumRevMaps.reset_index().drop('index', axis=1) #reseting index and drop index column

   data = [ dict(
           type = 'choropleth',
           locations = sumRevMaps['country'],
           locationmode = 'country names',
           z = sumRevMaps['count_sales'],
           text = sumRevMaps['country'],
           autocolorscale = False,
           marker = dict(
               line = dict (
                   color = 'rgb(180,180,180)',
                   width = 0.5
               ) ),
           colorbar = dict(
               autotick = False,
               tickprefix = '',
               title = 'Count of Sales'),
         ) ]

   layout = dict(
       title = 'Total Sales by Country',
       geo = dict(
           showframe = False,
           showcoastlines = True,
           projection = dict(
               type = 'Mercator'
           )
       )
   )

   figure = dict( data=data, layout=layout )

   iplot(figure, validate=False, filename='map-countrys-total')
plotmaprevenues(train)


# **<font size=3>leak data set :  revenues counts</font>**

# In[ ]:


plotmaprevenues(train_leak)


# **<font size=3>It is my first EDA:))))))))))</font>**

# In[ ]:




