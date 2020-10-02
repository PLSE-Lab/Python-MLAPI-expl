#!/usr/bin/env python
# coding: utf-8

# # Overview

# This kernel was made to provide a repository of applying clustering and forecasting concept. Hopefully, it will be usefull for the all readers. Thank you.
# 
# Table of contents:
# 
#     Data overview
#     Loading library and data
#     Segmentation
#         Pre-processing data for segmentation
#         Determining the number of cluster
#         Clustering
#         Visualization of segments
#         Analyzing each segment
#     Forecasting Single time-series
#         Prepocessing Data for forecasting
#         Forecasting electricity usage of each segment using Fbprophet
#         Forecasting electricity usage of all costumers using Fbprophet

# # Loading Library and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)


# In[ ]:


data_ori = pd.read_csv('../input/daily_electricity_usage.csv')
data_ori['date'] = pd.to_datetime(data_ori['date'])


# In[ ]:


data_ori.head()


# The given dataset contains 6445 IDs from ID 1000 to ID 7444. The dataset provide daily electricity usage from July 14, 2009 to December 31, 2010. The costumers are not only from housing but also companys.

# # 1. Preprocessing Data

# ## Splitting data per costumer 

# In[ ]:


data = pd.DataFrame({'date':pd.date_range('2009-07-14',periods=536,freq='D',)})
for i in range(1000,7445):
    S=data_ori[data_ori['Meter ID']==i][['date','total daily KW']]
    data=pd.merge(data,S,how='left',on='date')
for i in range(1,6446):
    data.columns.values[i]="ID"+str(999+i)


# In[ ]:


data.head()


# ## Handling missing data

# In[ ]:


data.isnull().sum().sum()


# In[ ]:


data = data.fillna(data.mean())


# The dataset contains 163.262 missing data because:
# 1. there are new costumers in the middle of observation period;
# 2. there are some Meter IDs between 1000-7444 not observed; or
# 3. there are some meter IDs that stop to be member in the middle of observation period.

# # 1. Segmentation

# ## Preparing features for Segmentation 

# In[ ]:


data.date = pd.to_datetime(data.date)
data['day'] = data['date'].apply(lambda x:x.weekday())
x_call = data.columns[1:-1]


# In[ ]:


data_fix = pd.DataFrame({'Meter ID':range(1000,7445,1),'total KW':np.sum(data[x_call]).values})
data_fix['average per day']=data[x_call].mean().values
data_fix['% Monday']=data[data['day']==0][x_call].sum().values/data_fix['total KW']*100
data_fix['% Tuesday']=data[data['day']==1][x_call].sum().values/data_fix['total KW']*100
data_fix['% Wednesday']=data[data['day']==2][x_call].sum().values/data_fix['total KW']*100
data_fix['% Thursday']=data[data['day']==3][x_call].sum().values/data_fix['total KW']*100
data_fix['% Friday']=data[data['day']==4][x_call].sum().values/data_fix['total KW']*100
data_fix['% Saturday']=data[data['day']==5][x_call].sum().values/data_fix['total KW']*100
data_fix['% Sunday']=data[data['day']==6][x_call].sum().values/data_fix['total KW']*100
data_fix['% weekday']=data[(data['day']!=5)&(data['day']!=6)][x_call].sum().values/data_fix['total KW']*100
data_fix['% weekend']=data[(data['day']==5)|(data['day']==6)][x_call].sum().values/data_fix['total KW']*100


# In[ ]:


data_fix=data_fix.fillna(0)
data_fix.head()


# We built 11 variables to detect the consumption behavior of every costumers. Those are:
# 1. Total consumption in the observation period (total KW);
# 2. The average of daily electricity usage (average per day);
# 3. The percentage of total consumption on Monday (% Monday);
# 4. The percentage of total consumption on Tuesday (% Tuesday);
# 5. The percentage of total consumption on Wednesday (% Wednesday);
# 6. The percentage of total consumption on Thursday (% Thursday);
# 7. The percentage of total consumption on Friday (% Friday);
# 8. The percentage of total consumption on Saturday (% Saturday);
# 9. The percentage of total consumption on Sunday (% Sunday);
# 10. The percentage of total consumption on Weekday (% weekday); and
# 11. The percentage of total consumption on Weekend (% weekend).

# ## Standardization Data 

# In[ ]:


from sklearn.preprocessing import StandardScaler
x_calls = data_fix.columns[1:]
scaller = StandardScaler()
matrix = pd.DataFrame(scaller.fit_transform(data_fix[x_calls]),columns=x_calls)
matrix['Meter ID'] = data_fix['Meter ID']
print(matrix.head())


# We keep the outlier so the costumers from big company or too small housing not be eliminated.

# ## Correlation 

# In[ ]:


corr = matrix[x_calls].corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax=ax.matshow(corr,vmin=-1,vmax=1)
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.xticks(rotation=90)
plt.colorbar(cax)


# As we can guess "total KW" has strong positive correlation with "average per day". Beside that, "% Saturday" and "% Sunday" also have strong positive correlation with "% weekend" and negative correlation with "% weekday". So does "% Monday" until "% Friday" have positive corralation with "% weekday" and "% weekend".

# ## Determining the number of cluster

# In[ ]:


def plot_BIC(matrix,x_calls,K):
    from sklearn import mixture
    BIC=[]
    for k in K:
        model=mixture.GaussianMixture(n_components=k,init_params='kmeans')
        model.fit(matrix[x_calls])
        BIC.append(model.bic(matrix[x_calls]))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(K,BIC,'-cx')
    plt.ylabel("BIC score")
    plt.xlabel("k")
    plt.title("BIC scoring for K-means cell's behaviour")
    return(BIC)


# In[ ]:


K = range(2,31)
BIC = plot_BIC(matrix,x_calls,K)


# By Bayessian Information Criterion (BIC), we decided to segmentate the costumers to be 5 segments.

# ## Clustering 

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
cluster = KMeans(n_clusters=5,random_state=217)
matrix['cluster'] = cluster.fit_predict(matrix[x_calls])
print(matrix.cluster.value_counts())


# In[ ]:


d=pd.DataFrame(matrix.cluster.value_counts())
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(d.index,d['cluster'],align='center',alpha=0.5)
plt.xlabel('Cluster')
plt.ylabel('number of data')
plt.title('Cluster of Data')


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances
distance = euclidean_distances(cluster.cluster_centers_, cluster.cluster_centers_)
print(distance)


# The first segment (Cluster 0) contains 95 costumers, the second (Cluster 1) 3747 costumers, the third (Cluster 2) 10 costumers, the fourth (Cluster 3) 2208 costumers, and the fifth (Cluster 4) 385 costumers.

# ## Visualization Segment

# In[ ]:


# Reduction dimention of the data using PCA
pca = PCA(n_components=3)
matrix['x'] = pca.fit_transform(matrix[x_calls])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_calls])[:,1]
matrix['z'] = pca.fit_transform(matrix[x_calls])[:,2]

# Getting the center of each cluster for plotting
cluster_centers = pca.transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x', 'y', 'z'])
cluster_centers['cluster'] = range(0, len(cluster_centers))
print(cluster_centers)


# In[ ]:


# Plotting for 2-dimention
fig, ax = plt.subplots(figsize=(8, 6))
scatter=ax.scatter(matrix['x'],matrix['y'],c=matrix['cluster'],s=21,cmap=plt.cm.Set1_r)
ax.scatter(cluster_centers['x'],cluster_centers['y'],s=70,c='blue',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)
plt.title('Data Segmentation')


# In[ ]:


# Plotting for 3-Dimention
fig, ax = plt.subplots(figsize=(8, 6))
ax=fig.add_subplot(111, projection='3d')
scatter=ax.scatter(matrix['x'],matrix['y'],matrix['z'],c=matrix['cluster'],s=21,cmap=plt.cm.Set1_r)
ax.scatter(cluster_centers['x'],cluster_centers['y'],cluster_centers['z'],s=70,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.colorbar(scatter)
plt.title('Data Segmentation')


# By the plots above, we can see that all segments are separated well from each other. It means that BIC method works good for this project.

#  ## The behavior of each segment:

# In[ ]:


data_fix['cluster']=matrix['cluster']
print(data_fix[data_fix.columns[1:]].groupby(['cluster']).agg([np.mean]))


# In[ ]:


list(data_fix[data_fix.cluster==2]['Meter ID'])


# 1. Cluster 0 contains only 95 IDs but the average of daily usage and total KW is so high, 381.55 kWh and 304513.39 kWh respectively. They also have activities in all days, the highest in Thursday and Friday. We can guess that Cluster 0 comes from big companies.
# 
# 2. Cluster 1 is a biggest segment in this project contains 3747 IDs. The average of total electricity usage in observation period is 15161.06 kWh and the average of daily electricity usage is 28.28 kWh. From the percentage of daily usage in weekend or weekdays, we can see that there is no significant different between them. The consumption is about 14%. By this behaviour we can guess that the IDs in Cluster 0 comes from housing or small company that has same activities in all days.
# 
# 3. Cluster 2 contains 10 IDs that is not observed in this project. It can be seen from the total of electricity usage that is 0 kWh. The IDs not observed in this project are 2083, 2691, 3141, 3348, 4096, 4113, 4447, 5855, 6596, dan 6713.
# 
# 4. Cluster 4 has 2208 IDs and similar behavior with Cluster 1. It is confirmed by the distance between those centroids. Not only that, the similar behavior also can be seen from the daily electricity usage and the total consumption 26.4 kWh and 13955.61 kWh respectively. However, the percentage of daily using is little bit different eith Cluster 0, in weekdays the percentage is about 13% per day and it increases in weekend becomes about 16%. We can guess that those IDs are constumers comes from housing or small companies who have more activities in weekend rather than weekdays.
# 
# 5. The IDs in Cluster 5 comes from middle companies who only have activities in weekdays. We can see from the behavior of daily electricity usage where there is significant different between weekdays and weekend. In weekdays, the percentage of electricity usage is about 17% and it become slighly decrease in Monday and Friday. However,it got dramatically decrease in weekend about only 7%.

# # 2. Forecasting Using Fbprophet

# ## Preprocessing data

# In this stage for this kernel, we try to forecast for a year ahead only the electricity usage of each segment, except Cluster 2, and the total electricity usage of all costumers using a simple and very nice library developed by Facebook name[](http://)d Fbprophet. The theory with a very good description of the math/statistical approach behind the library can be seen in https://facebook.github.io/prophet/.

# In[ ]:


data_cluster=data_fix[['Meter ID','cluster']]
data_forc=pd.DataFrame({'ds':pd.to_datetime(data['date'])})


# In[ ]:


for k in range(len(cluster_centers)):
    data_clus=data_cluster[data_cluster['cluster']==k]
    del data_clus['cluster']
    s1="cluster "+str(k)
    data_forc[s1]=0
    for i in list(data_clus.iloc[:,0]):
        s2="ID"+str(i)
        data_forc[s1]+=data[s2]
data_forc=data_forc.fillna(0)


# In[ ]:


data_forc_0=data_forc[['ds','cluster 0']]
data_forc_0.columns=['ds','y']

data_forc_1=data_forc[['ds','cluster 1']]
data_forc_1.columns=['ds','y']

data_forc_2=data_forc[['ds','cluster 2']]
data_forc_2.columns=['ds','y']

data_forc_3=data_forc[['ds','cluster 3']]
data_forc_3.columns=['ds','y']

data_forc_4=data_forc[['ds','cluster 4']]
data_forc_4.columns=['ds','y']

data_forc_all=pd.DataFrame({'ds':data_forc['ds']})
data_forc_all['y']=data_forc['cluster 0']+data_forc['cluster 1']+data_forc['cluster 2']+data_forc['cluster 3']+data_forc['cluster 4']


# In[ ]:


def plot_data(data_forc):
    timeseries=data_forc.copy()
    timeseries.columns=['date','Total Daily KW']
    timeseries = timeseries.set_index('date') 
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(timeseries.index,timeseries['Total Daily KW'],c='black',s=2)


# ## Developing Function for Modelling 

# In[ ]:


import fbprophet
from sklearn.metrics import mean_squared_error, r2_score
def predic_fbp(data_forc,n_days):
    ny=pd.DataFrame({'holiday':"New Year's Day",'ds':pd.to_datetime(['2010-01-01','2011-01-01','2012-01-01']),
                     'lower_window':-1,'upper_window':1,})
    ch=pd.DataFrame({'holiday':"Christmas",'ds':pd.to_datetime(['2009-12-25','2010-12-25','2011-12-25','2012-12-25']),
                     'lower_window':0,'upper_window':1,})
    holidays=pd.concat([ny,ch])
    model = fbprophet.Prophet(daily_seasonality=False,weekly_seasonality=True,
                yearly_seasonality=True,changepoint_prior_scale=0.05,changepoints=None,
                holidays=holidays,interval_width=0.95)
    model.add_seasonality(name='monthly',period=30.5,fourier_order=5)
    size = len(data_forc) - n_days
    train, test = data_forc[0:size], data_forc[size:]
    test_=test.set_index('ds')
    model.fit(train)
    predics=model.predict(data_forc)
    test=pd.merge(test,predics[['ds','yhat','yhat_lower','yhat_upper']],how='left',on='ds')
    train=pd.merge(train,predics[['ds','yhat','yhat_lower','yhat_upper']],how='left',on='ds')
    RMSE=np.sqrt(mean_squared_error(test['y'], test['yhat']))
    print('RMSE = %.2f' % RMSE)
    R2=r2_score(test['y'], test['yhat'])
    print('R Square = %.2f'% R2)
    future = model.make_future_dataframe(periods=365+n_days, freq='D')
    future=model.predict(future)
    fig=model.plot(predics)
    plt.scatter(test_.index,test_['y'],c='black',s=7)
    fig2=model.plot(future)
    plt.scatter(test_.index,test_['y'],c='black',s=7)
    fig3=model.plot_components(future)
    return(train,test,predics,future,RMSE,R2)


# In forecasting, we use 446 first days, from July 14 2009 to October 02 2010, as a training data and 90 last day , from October 03 2010 to December 31 2010, as a validation data. We also added the holiday effects, Chistmas at December 25-26 and New Year at December 31 - January 02, on the model.
# 
# For accuracy, we use Root Mean Square Error (RMSE) and R-Squared (R2) to asses the model.

# ## The Forecasting of Cluster 0

# In[ ]:


plot_data(data_forc_0)


# In[ ]:


train_0,test_0,predics_0,future_0,RMSE_0,R2_0=predic_fbp(data_forc_0,90)


# The trend of electricity usage in this cluster is linearly decreasing with the highest usage in December and January and the lowest in June. R Squared of this prediction is not really good only 64% but the RMSE is good with 1637.01.

# ## The Forecasting of Cluster 1

# In[ ]:


plot_data(data_forc_1)


# In[ ]:


train_1,test_1,predics_1,future_1,RMSE_1,R2_1=predic_fbp(data_forc_1,90)


# The trend of electricity usage in this cluster is also decreasing with the same highest usage with Cluster 0 in December and January and the lowest in about July. R Squared of this prediction is good 84% and the RMSE is about 5531.48.

# ## The Forecasting of Cluster 3

# In[ ]:


plot_data(data_forc_3)


# In[ ]:


train_3,test_3,predics_3,future_3,RMSE_3,R2_3=predic_fbp(data_forc_3,90)


# The trend of electricity usage in this cluster is also linearly decreasing with the highest usage in January and the lowest in about June. R Squared of this prediction is good 82% and the RMSE is 4439.82.

# ## The Forecasting of Cluster 4

# In[ ]:


plot_data(data_forc_4)


# In[ ]:


train_4,test_4,predics_4,future_4,RMSE_4,R2_4=predic_fbp(data_forc_4,90)


# The trend of electricity usage in this cluster is also linearly decreasing with the highest usage in January and the lowest in about August. R Squared of this prediction is good enough 77% and the RMSE is 4589.76.

# ## The Forecasting of All Costumers

# In[ ]:


plot_data(data_forc_all)


# In[ ]:


train_all,test_all,predics_all,future_all,RMSE_all,R2_all=predic_fbp(data_forc_all,90)


# Overall, the trend of electricity usage is linearly decreasing with the highest usage in January and the lowest in about June. R Squared of this prediction is good enough 79% but the RMSE is 12125.71.
