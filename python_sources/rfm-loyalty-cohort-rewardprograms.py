#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
import datetime as dt
import sklearn
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import __version__
print(__version__)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 5)
pd.options.display.float_format = '{:20,.2f}'.format
np.set_printoptions(suppress =True) 

import os
print(os.listdir("../input"))
import tdqm
from tqdm import tqdm


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv', nrows=50).head(5)


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv', nrows=50).head(3)


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv', nrows=50).head(3)


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/train.csv', nrows=50).head(3)


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/test.csv', nrows=50).head(3)


# In[ ]:


pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv', nrows=50).head(3)


# In[ ]:


pd.read_excel('../input/elo-merchant-category-recommendation/Data_Dictionary.xlsx')


# In[ ]:


pd.read_csv('../input/ecommerce-data/data.csv', nrows=50).head(3)


# ### Reading the Data

# In[ ]:


# df_train = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv", nrows = 10000)
# df_test = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv", nrows = 10000)
# df_historical =pd.read_csv("../input/elo-merchant-category-recommendation/historical_transactions.csv",parse_dates=['purchase_date'])
# df_new =pd.read_csv("../input/elo-merchant-category-recommendation/new_merchant_transactions.csv",parse_dates=['purchase_date'], nrows = 10000)


# In[ ]:


df_train = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv")
df_test = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv")
df_historical =pd.read_csv("../input/elo-merchant-category-recommendation/historical_transactions.csv",parse_dates=['purchase_date'])
df_new =pd.read_csv("../input/elo-merchant-category-recommendation/new_merchant_transactions.csv",parse_dates=['purchase_date'])


# In[ ]:


df_historical.head(2)


# In[ ]:


len(df_historical.card_id.unique())


# In[ ]:


df_new.head(2)


# In[ ]:


df_historical=df_historical.loc[df_historical.authorized_flag=="Y",]
df_historical.purchase_amount += 0.75
df_new.purchase_amount += 0.75


# ### Checking for Purchase, Return and Rebate Transactions (if any)

# ## As the purchase_amount variable is normalized it is not possible to check if the transaction if a Purchase/Return/Rebate one

# ### Aggregation by card-id (At Customer Level). Can also compute at Item Level RFM and Store Level RFM or even website visit & activity based RFM etc depending on availability of Data

# In[ ]:


def groupby_mean(x):
    return x.mean()

def groupby_count(x):
    return x.count()

def purchase_duration(x):
    return (x.max() - x.min()).days

def avg_frequency(x):
    return (x.max() - x.min()).days/x.count()

groupby_mean.__name__ = 'avg'
groupby_count.__name__ = 'count'
purchase_duration.__name__ = 'purchase_duration'
avg_frequency.__name__ = 'purchase_frequency'

def get_max(cols):
    return max(cols[0],cols[1])


# In[ ]:


df_Agg_Monetary = df_historical.groupby('card_id').agg({'purchase_amount':sum})
df_Agg_Monetary.columns = ['Monetary']
print(df_Agg_Monetary.shape)
df_Agg_Monetary.head()


# In[ ]:


df_Agg_Frequency = df_historical.groupby('card_id').agg({'card_id': groupby_count,'purchase_date': groupby_count})
df_Agg_Frequency['Frequency'] = df_Agg_Frequency[['card_id','purchase_date']].apply(get_max,axis = 1)
print(df_Agg_Frequency.shape)
df_Agg_Frequency.head()


# In[ ]:


table_max_date = max(df_historical['purchase_date'])
print(table_max_date)

x = df_historical['purchase_date'][0]
(table_max_date-x).days


# In[ ]:


df_Agg_Recency = df_historical.groupby(['card_id']).agg({'purchase_date':max})
df_Agg_Recency['Recency'] = df_Agg_Recency['purchase_date'].apply(lambda x:(table_max_date-x).days)
print(df_Agg_Recency.shape)
df_Agg_Recency.head(10)


# In[ ]:


df_rfm = pd.merge(pd.merge(df_Agg_Recency, df_Agg_Frequency, left_index=True, right_index=True), 
                           df_Agg_Monetary, left_index=True, right_index=True)
df_rfm.drop(columns=['purchase_date_x', 'card_id', 'purchase_date_y'], inplace=True)
df_rfm.head(5)


# In[ ]:


counts, bin_edges = np.histogram(df_rfm['Recency'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# In[ ]:


counts, bin_edges = np.histogram(df_rfm['Frequency'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# In[ ]:


counts, bin_edges = np.histogram(df_rfm['Monetary'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df_rfm, size=3);
plt.show()


# In[ ]:


# fig = ff.create_scatterplotmatrix(df_rfm, height=800, width=800)
# iplot(fig, filename='Basic Scatterplot Matrix')


# ### If Geospatial Variables like City/State/County/Latitude/Longitude are given, these plots might also give some insights

# In[ ]:


df_StateRFM = pd.read_csv("/kaggle/input/statewiserfm/State_RFM.csv")
df_StateRFM


# In[ ]:


scl = [
    [0.0, 'rgb(242,240,247)'],
    [0.2, 'rgb(218,218,235)'],
    [0.4, 'rgb(188,189,220)'],
    [0.6, 'rgb(158,154,200)'],
    [0.8, 'rgb(117,107,177)'],
    [1.0, 'rgb(84,39,143)']
]
data = [go.Choropleth(
    colorscale = scl,
    autocolorscale = False,
    locations = df_StateRFM['State'],
    z = df_StateRFM['Monetary'].astype(float),
    locationmode = 'USA-states',
    marker = go.choropleth.Marker(
        line = go.choropleth.marker.Line(
            color = 'rgb(255,255,255)',
            width = 2
        )),
    colorbar = go.choropleth.ColorBar(
        title = "Dollars")
)]

layout = go.Layout(
    title = go.layout.Title(
        text = 'State wise Monetary Value '
    ),
    geo = go.layout.Geo(
        scope = 'usa',
        projection = go.layout.geo.Projection(type = 'albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'd3-cloropleth-map')


# In[ ]:


df_StateRFM['Frequency'] = np.round(df_StateRFM['Frequency'])
data = [go.Choropleth(
    colorscale = scl,
    autocolorscale = False,
    locations = df_StateRFM['State'],
    z = df_StateRFM['Frequency'].astype(float),
    locationmode = 'USA-states',
    marker = go.choropleth.Marker(
        line = go.choropleth.marker.Line(
            color = 'rgb(255,255,255)',
            width = 2
        )),
    colorbar = go.choropleth.ColorBar(
        title = "Frequency Count")
)]

layout = go.Layout(
    title = go.layout.Title(
        text = 'State wise Frequency Value '
    ),
    geo = go.layout.Geo(
        scope = 'usa',
        projection = go.layout.geo.Projection(type = 'albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'd3-cloropleth-map')


# In[ ]:


df_StateRFM['Recency'] = np.round(df_StateRFM['Recency'])
data = [go.Choropleth(
    colorscale = scl,
    autocolorscale = False,
    locations = df_StateRFM['State'],
    z = df_StateRFM['Recency'].astype(float),
    locationmode = 'USA-states',
    marker = go.choropleth.Marker(
        line = go.choropleth.marker.Line(
            color = 'rgb(255,255,255)',
            width = 2
        )),
    colorbar = go.choropleth.ColorBar(
        title = "Recency Count")
)]

layout = go.Layout(
    title = go.layout.Title(
        text = 'State wise Recency Value '
    ),
    geo = go.layout.Geo(
        scope = 'usa',
        projection = go.layout.geo.Projection(type = 'albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'd3-cloropleth-map')


# ## 1. Simple Rule Based Approach - The Business Analyst Way

# In[ ]:


df_rfm.quantile(q=[0.1,0.25,0.4,0.5,0.75,0.9])


# ### Thresholds for R | F | M

# In[ ]:


threshold_M = int(df_rfm.median()['Monetary'])+1
print(threshold_M)
threshold_F = df_rfm.Frequency.quantile(0.75)
print(threshold_F)
threshold_R = df_rfm.Recency.quantile(0.40)
print(threshold_R)


# In[ ]:


df_rfm['threshold_R'] = df_rfm['Recency'].apply(lambda x: x < threshold_R)
df_rfm['threshold_F'] = df_rfm['Frequency'].apply(lambda x: x > threshold_F)
df_rfm['threshold_M'] = df_rfm['Monetary'].apply(lambda x: x > threshold_M)
df_rfm[['threshold_R', 'threshold_F', 'threshold_M']] = df_rfm[['threshold_R', 'threshold_F', 'threshold_M']].apply(lambda x: x.astype(int), axis=1)
df_rfm['IsLoyal'] = 'NA'
df_rfm['Segment'] = 'NA'
df_rfm.head(10)


# In[ ]:


def Loyalty_assign(x):
    if((x[5]==1) & (x[4]==1)):
        return 'Loyal'
       
    elif((x[3]==1) & (x[4]==0)):
        return 'Loyal'
         
    else:
        return 'Not Loyal'
    
def Segment_assign(x):
    if((x[5]==1) & (x[3]==1) & (x[4]==1)):
        return 'Champions'
       
    elif((x[5]==1) & (x[3]==1) & (x[4]==0)):
        return 'Future Champions'
         
    elif((x[5]==1) & (x[3]==0) & (x[4]==1)):
        return 'Very Valuable'
         
    elif((x[5]==1) & (x[3]==0) & (x[4]==0)):
        return 'Hibernating'
         
    elif((x[5]==0) & (x[3]==1) & (x[4]==1)):
        return 'Active'
         
    elif((x[5]==0) & (x[3]==1) & (x[4]==0)):
        return 'About to Sleep'
         
    elif((x[5]==0) & (x[3]==0)):
        return 'Lost'


# In[ ]:


df_rfm['Segment'] = df_rfm.apply(Segment_assign, axis=1)
df_rfm['IsLoyal'] = df_rfm.apply(Loyalty_assign, axis=1)
df_rfm.head()


# In[ ]:


df_rfm['IsLoyal'].value_counts()


# In[ ]:


df_rfm['Segment'].value_counts()


# In[ ]:


df_rfm.reset_index(inplace=True)
df_rfm.head(1)


# In[ ]:


df_Segment = df_rfm.groupby('Segment', as_index=False).agg({'Monetary':sum, 'card_id':groupby_count, 'Frequency':sum})
df_Segment.columns = ['Segment','Monetary', 'No_Cards', 'Frequency']
df_Segment


# In[ ]:


df_Loyal = df_rfm.groupby('IsLoyal', as_index=False).agg({'Monetary':sum, 'card_id':groupby_count, 'Frequency':sum})
df_Loyal.columns = ['Loyality','Monetary', 'No_Cards', 'Frequency']
df_Loyal


# In[ ]:


groups = df_Segment['Segment'].values.tolist()
amount = df_Segment['Monetary'].values.tolist()
#colors = ['red', 'yellow', 'green', 'orange']

trace = go.Pie(labels=groups, values=amount, hoverinfo='label+percent', textinfo='value', textfont=dict(size=25),
       pull=.4,hole=.2,marker=dict(line=dict(color='#000000', width=3)))

iplot([trace])


# In[ ]:


groups = df_Loyal['Loyality'].values.tolist()
amount = df_Loyal['Monetary'].values.tolist()
#colors = ['red', 'yellow', 'green', 'orange']

trace = go.Pie(labels=groups, values=amount, hoverinfo='label+percent', textinfo='value', textfont=dict(size=25),
       pull=.4,hole=.2,marker=dict(line=dict(color='#000000', width=3)))

iplot([trace])


# In[ ]:


groups = df_Segment['Segment'].values.tolist()
No_Company = df_Segment['No_Cards'].values.tolist()
Distinct_Frequency = df_Segment['Frequency'].values.tolist()
#colors = ['blue','red', 'yellow', 'pink','violet','green', 'orange']

#trace2 = go.Bar(x=groups,y=No_Company,name='Companies', marker=dict(color=colors))

trace1 = go.Bar(x=groups,y=Distinct_Frequency,name='Frequency')
trace2 = go.Bar(x=groups,y=No_Company,name='Cards/Customers')
data = [trace1, trace2]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# In[ ]:


groups = df_Loyal['Loyality'].values.tolist()
No_Company = df_Loyal['No_Cards'].values.tolist()
Distinct_Frequency = df_Loyal['Frequency'].values.tolist()
#colors = ['blue','red', 'yellow', 'pink','violet','green', 'orange']

#trace2 = go.Bar(x=groups,y=No_Company,name='Companies', marker=dict(color=colors))

trace1 = go.Bar(x=groups,y=Distinct_Frequency,name='Frequency')
trace2 = go.Bar(x=groups,y=No_Company,name='Cards/Customers')
data = [trace1, trace2]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# ## 2. Clustering Algorithms Based Segmentation

# In[ ]:


df_rfm.set_index('card_id', inplace=True)
df_rfm.head(1)


# In[ ]:


rank_df = df_rfm[['Recency','Frequency', 'Monetary']].rank(method='first')
rank_df.head(2)


# In[ ]:


normalized_df = (rank_df - rank_df.mean()) / rank_df.std()
normalized_df.head(2)


# In[ ]:


from sklearn.cluster import KMeans
data = normalized_df[['Recency', 'Frequency', 'Monetary']]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


print(normalized_df.shape)


# In[ ]:


# for n_cluster in tqdm([2,3,4,5]):
#     kmeans = KMeans(n_clusters=n_cluster, max_iter=100).fit(normalized_df[['Recency', 'Frequency', 'Monetary']])
    
#     silhouette_avg = silhouette_score(normalized_df[['Recency', 'Frequency', 'Monetary']], kmeans.labels_)
    
#     print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))


# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4,max_iter=1000).fit(normalized_df[['Recency', 'Frequency', 'Monetary']])
print(kmeans.labels_)
print(kmeans.cluster_centers_)


# In[ ]:


df_kmeans_rfm = df_rfm[['Recency','Frequency', 'Monetary']].copy()
df_kmeans_rfm['Cluster'] = kmeans.labels_
print(df_kmeans_rfm['Cluster'].value_counts())
df_kmeans_rfm.head(5)


# In[ ]:


# trace1 = go.Scatter3d(
#     x=normalized_df['Recency'],
#     z=normalized_df['Monetary'],
#     y=normalized_df['Frequency'],
#     mode='markers',
#     marker=dict(size=12,color=df_kmeans_rfm['Cluster'],))

# data = [trace1]

# layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))

# fig = go.Figure(data=data, layout=layout)
# iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:


colors = ['red', 'yellow', 'green', 'orange']

for i in ['Monetary']:
    trace = go.Pie(labels=df_kmeans_rfm['Cluster'], values=df_kmeans_rfm[i], 
           hoverinfo='label+percent', textinfo='value', textfont=dict(size=15),
           marker=dict(colors=colors, line=dict(color='#000000', width=3)))
    iplot([trace])


# In[ ]:





# ### This is just the first draft version. Will include some more of my own code with lots of updates in coming weeks

# In[ ]:





# In[ ]:




